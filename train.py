import copy
import json
import os
import random
import time
from collections import deque
import wandb

import gym
import pandas as pd
import torch
import numpy as np

import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env, EED_SPT, act_EDD_SPT


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    setup_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float32)
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    env_valid_paras["DDT_a"] = False
    env_valid_paras["seed"] = 7
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 1 + model_paras["out_size_ope"] * 1
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.4)
    opes_per_job_max = int(num_mas * 1.2)

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation
    if train_paras["load_model"]:
        mod_files = os.listdir('./load_model/')[:]
        if device.type == 'cuda':
            model_CKPT = torch.load('./load_model/' + mod_files[0])
        else:
            model_CKPT = torch.load('./load_model/' + mod_files[0], map_location='cpu')

        model.policy.load_state_dict(model_CKPT)
        model.policy_old.load_state_dict(model_CKPT)

        # model.reset_actor()

    maxlen = 1  # Save the best model
    best_models = deque()
    tardy_best = float('inf')

    use_wandB = train_paras["use_wandB"]
    if use_wandB:
        run = wandb.init(project=train_paras["project_name"], config=train_paras)
        run.name = f'M{env_paras["num_mas"]}_L{env_paras["num_jobs"]}_BS{env_paras["batch_size"]}_U{env_paras["ma_util"]}_DDT{env_paras["DDT_high"]}_SEED{env_paras["seed"]}_INIT{env_paras["init_jobs"]}_T{env_paras["time_period"]}'

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    os.makedirs(save_path)

    # Training curve storage path (average of validation set) TARDINESS
    writer_ave_tardy = pd.ExcelWriter('{0}/training_ave_tardy_{1}.xlsx'.format(save_path, str_time))
    # Training curve storage path (value of each validating instance) TARDINESS
    writer_100_tardy = pd.ExcelWriter('{0}/training_100_tardy_{1}.xlsx'.format(save_path, str_time))

    valid_results_tardy = []
    valid_results_tardy_100 = []
    st = train_paras["save_timestep"]
    en = train_paras["max_iterations"] + train_paras["save_timestep"]
    data_file = pd.DataFrame(np.arange(st, en, st), columns=["iterations"])
    data_file.to_excel(writer_ave_tardy, sheet_name='Sheet1', index=False)
    data_file = pd.DataFrame(np.arange(st, en, st), columns=["iterations"])
    data_file.to_excel(writer_100_tardy, sheet_name='Sheet1', index=False)

    # Start training iteration
    start_time = time.time()
    # \mathcal{B} instances use consistent operations to speed up training
    case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, fully_flexible=True,
                         flag_same_opes=False, flag_doc=False, flag_same_proc=True)
    file_path = "./data_dev/5mas10prodHurink/"
    case = sorted(os.listdir(file_path))
    for i in range(len(case)):
        case[i] = file_path + case[i]
    print(f'valid_data_files {case}')
    env = gym.make('fjsp-v0', case=case, env_paras=env_paras, data_source='file')
    # env = gym.make('fjsp-v0', case=case, env_paras=env_paras)

    print('num_job: ', num_jobs, '\tnum_mas: ', num_mas)

    eps = 1.0
    for i in range(1, train_paras["max_iterations"]+1):
        if (i % train_paras["update_timestep"] == 0) and True:  # every x iterations don't reset rng
            print(f'updating selection')
            eps = eps * 0.95
            env.training_reset(reset_rng=False)
            env_spt = copy.deepcopy(env)
        else:  # every other iteration reset it, which means instance stays same of x iters then updates but lib the same
            env.training_reset(reset_rng=True)
            env_spt = copy.deepcopy(env)
        # Replace training instances every x iteration (x = 20 in paper)
        if (i  % train_paras["parallel_iter"] == 0) and False:  # every 40 iterations update the job library
            # replace library of jobs
            print(f'updating lib')
            env_paras["seed"] = env_paras["seed"] + 1
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            env_spt = copy.deepcopy(env)

        # Get state and completion signal
        state = env.state
        done = False
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                h_mas_glob, h_proc_glob, h_opes_glob, ope_ma = model.policy_old.get_global_features(state, memories,
                                                                                                    flag_train=True)
                shared_feats = (h_mas_glob, h_proc_glob, h_opes_glob)
                group_actions = model.policy_old.get_actions(num_mas, state, memories, shared_feats, ope_ma,
                                                             flag_train=True, flag_greedy=False, negotiate_rule=env_paras["negotiate_rule"])

            state, rewards, dones = env.step(group_actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            memories.clear_action_probs()
            memories.clear_attention()
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time() - last_time)
        tardiness = copy.deepcopy((env.true_tardiness_batch).mean())
        tot_jobs_completed = copy.deepcopy((env.tot_scheduled_jobs).to(torch.float).mean())

        if (((i == 1) or (i % train_paras["update_timestep"] == 0) or (i % train_paras["parallel_iter"] == 0)) and use_wandB):
            eed_result_tardy2, eed_result_tardy_1002, eed_result_jobs2, eed_result_jobs_1002 = EED_SPT(env_paras,
                                                                                                           env_spt)

        if use_wandB:
            wandb.log({'tardiness': tardiness.item(), 'jobs completed': tot_jobs_completed.item(),
                       'mean': tardiness.item() / (tot_jobs_completed.item() + 1e-5), 'EED tardiness': eed_result_tardy2.item(),
                       'EED jobs completed': eed_result_jobs2.item(),
                       'EED mean': eed_result_tardy2.item() / (eed_result_jobs2.item()+ 1e-5)}, step=i)

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_paras["update_freq"] == 0:
            loss, actor_loss, critic_loss, wait_entropy, job_entropy, wait_policy, job_policy, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()
            if use_wandB:
                wandb.log({'reward': reward}, step=i)
                wandb.log({'LOSS': loss, 'Actor Loss':actor_loss, 'Critic Loss': critic_loss}, step=i)
                wandb.log({'Wait Entropy Loss': wait_entropy, 'Job Entropy Loss': job_entropy}, step=i)
                wandb.log({'Wait Loss': wait_policy, 'Job Loss': job_policy}, step=i)
        if (i % train_paras["save_timestep"] == 0) and True or (i == 1) and True:
            print('\nStart validating')
            # Record the average results and the results on each instance
            env_valid.reset()
            vali_result_tardy, vali_result_tardy_100, vali_result_jobs, vali_result_jobs_100, vali_result_tardy_effective = validate(env_valid_paras, env_valid, model.policy_old)
            env_valid.reset()
            if i == 1:
                vali_result_tardy2, vali_result_tardy_1002, vali_result_jobs2, vali_result_jobs_1002 = EED_SPT(env_valid_paras, env_valid)
            valid_results_tardy.append(vali_result_tardy.item())
            valid_results_tardy_100.append(vali_result_tardy_100)

            # Save the best model
            # mean = vali_result_tardy.item() / (vali_result_jobs.item()+ 1e-5)
            if (vali_result_tardy_effective < tardy_best) and (vali_result_jobs > 0) and i > 1000:
                tardy_best = vali_result_tardy_effective
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            if use_wandB:
                wandb.log({'valid tardiness': vali_result_tardy.item(),
                           'valid jobs completed': vali_result_jobs.item(),
                           'valid mean': vali_result_tardy.item() / (vali_result_jobs.item()+ 1e-5),
                           'EED vali tardiness': vali_result_tardy2.item(),
                           'EED vali jobs completed': vali_result_jobs2.item(),
                           'EED vali mean': vali_result_tardy2.item() / (vali_result_jobs2.item()+ 1e-5)}, step=i)


    # Save the data of training curve to files
    data = pd.DataFrame(np.array(valid_results_tardy).transpose(), columns=["res"])
    data.to_excel(writer_ave_tardy, sheet_name='Sheet1', index=False, startcol=1)
    writer_ave_tardy.close()
    cols = train_paras["max_iterations"] // train_paras["save_timestep"]
    column = [i_col for i_col in range(env_valid_paras["batch_size"])]
    data = pd.DataFrame(np.array(torch.stack(valid_results_tardy_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100_tardy, sheet_name='Sheet1', index=False, startcol=1)
    writer_100_tardy.close()

    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()