import copy
import json
import os
import random
import time as time
import csv

import gym
import pandas as pd
import torch
import numpy as np

import PPO_model
from env.load_data import nums_detec


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dicts(no, headings):
    list_of_dicts = []
    for _ in range(no):
        data_dict = {heading: [] for heading in headings}
        list_of_dicts.append(data_dict)
    return list_of_dicts


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float32)
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    ind = 0

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_test_paras = load_dict["env_paras"]
    env_test_paras["continuous"] = True
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    env_test_paras["device"] = device
    model_paras["device"] = device
    env_test_paras["batch_size"] = 1
    env_test_paras["time_period"] = test_paras["time_period"]
    env_test_paras["seed"] = test_paras["test_seed"]
    env_test_paras["DDT_a"] = False
    env_test_paras["init_jobs"] = None
    data_headings = test_paras["data_headings"]
    rules = test_paras["rules"]
    mas_rules = test_paras["mas_rules"]
    models_to_test = test_paras["model"]
    model_location = test_paras["model_location"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 1 + model_paras["out_size_ope"] * 1
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    mod_files = sorted(os.listdir(model_location)[:])
    flag_greedy = test_paras["flag_greedy"]
    filter_machines = test_paras["flag_filter_machines"]
    num_filter = test_paras["flag_num_to_filter"]

    # load trained models
    memories_list = []
    model_list = []
    for model_no in models_to_test:
        memories_list.append(PPO_model.Memory())
        model_list.append(PPO_model.PPO(model_paras, train_paras))

        if device.type == 'cuda':
            model_CKPT = torch.load(model_location + mod_files[model_no])
        else:
            model_CKPT = torch.load(model_location + mod_files[model_no], map_location='cpu')
        print('\nloading checkpoint:', mod_files[model_no])
        model_list[model_no].policy.load_state_dict(model_CKPT)
        model_list[model_no].policy_old.load_state_dict(model_CKPT)
        print(f'model_{model_no}: {mod_files[model_no]}')


    for data_path in test_paras["data_path"]:
        counter = 1
        data_path_fin = ".{0}{1}/".format(test_paras["data_path_loc"], data_path)
        test_files = sorted(os.listdir(data_path_fin))
        test_files.sort(key=lambda x: x[:-4])
        test_files = test_files[:]
        for ma_util in test_paras["ma_util"]:
            env_test_paras["ma_util"] = ma_util
            for tight_percent in test_paras["tight_percent"]:
                if counter in test_paras["tests_to_exclude"]:
                    print(f'skipping test: {counter}')
                    counter += 1
                    continue
                env_test_paras["tight_percent"] = tight_percent
                env_test_paras["DDT_high"] = test_paras["DDT_high"]
                env_test_paras["DDT_low"] = test_paras["DDT_low"]

                writers = []
                save_path = f'{test_paras["save_path"]}{data_path}EX{counter:02}'

                env_test_paras["seed"] = test_paras["test_seed"]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # instance names
                file_name = [test_file for test_file in test_files]
                data_file = pd.DataFrame(columns=file_name)
                # time periods
                time_periods = list(range(env_test_paras["time_period"], env_test_paras["time_period"]*(test_paras["num_periods"]+1), env_test_paras["time_period"]))
                data_time_file = pd.DataFrame(time_periods)

                for i_drl in models_to_test:
                    writers.append(pd.ExcelWriter(
                        '{0}/test_results_DP_{1}_rule_{2}_{3}.xlsx'.format(save_path, data_path, f'DRL_{mod_files[i_drl]}', counter)))
                    # write instance names to each sheet
                    for heading in data_headings:
                        data_file.to_excel(writers[-1], sheet_name=heading, index=False, startcol=1)
                        data_time_file.to_excel(writers[-1], sheet_name=heading, index=False, header=["time"])
                for rule in rules:
                    for mas_rule in mas_rules:
                        # create file for each rule
                        job_mas_rule = f'{rule}_{mas_rule}'
                        writers.append(pd.ExcelWriter('{0}/test_results_DP_{1}_rule_{2}_{3}.xlsx'.format(save_path, data_path, job_mas_rule, counter)))
                        # write instance names to each sheet
                        for heading in data_headings:
                            data_file.to_excel(writers[-1], sheet_name=heading, index=False, startcol=1)
                            data_time_file.to_excel(writers[-1], sheet_name=heading, index=False, header=["time"])

                tot_no_envs = len(models_to_test) + len(rules) * len(mas_rules)
                start = time.time()
                counter += 1

                for i_ins in range(len(test_files)):
                    # load from test_file
                    step_time_last = time.time()
                    test_file = data_path_fin + test_files[i_ins]
                    with open(test_file) as file_object:
                        line = file_object.readlines()
                        ins_num_jobs, ins_num_mas, _ = nums_detec(line)
                    env_test_paras["num_jobs"] = ins_num_jobs
                    env_test_paras["num_mas"] = ins_num_mas
                    # print(f'jobs in lib {ins_num_jobs}')
                    # print(f'no. machines {ins_num_mas}')

                    # Create environment object
                    env = gym.make('fjsp-v0', case=[test_file], env_paras=env_test_paras, data_source='file')
                    env_test_paras["seed"] = env_test_paras["seed"] + 1
                    # list to store env for each rule
                    all_envs = []
                    for _ in range(tot_no_envs):
                        all_envs.append(copy.deepcopy(env))
                    print("Create env[{0}]".format(i_ins))

                    # dictionaries to store data for each rule
                    data_list = create_dicts(tot_no_envs, data_headings)

                    # Schedule an instance/environment
                    for T in range(test_paras["num_periods"]):
                        # get results from model
                        for i_model in models_to_test:
                            scheduled_data = schedule(all_envs[i_model], model_list[i_model], memories_list[i_model],
                                                      flag_greedy=flag_greedy, index=ind,
                                                      flag_filter_machines=filter_machines, flag_num_to_filter=num_filter,
                                                      negotiate_rule=env_test_paras["negotiate_rule"])

                            all_envs[i_model].half_reset()
                            # store data in dict
                            k = 0
                            for heading in data_headings:
                                data_list[i_model][heading].append(scheduled_data[k])
                                k += 1

                        # get results from other rules
                        env_count = len(models_to_test)
                        for rule in rules:
                            for mas_rule in mas_rules:
                                scheduled_data = HeuristicRule(env_test_paras, all_envs[env_count], rule=rule, mas_rule=mas_rule)
                                all_envs[env_count].half_reset()
                                k = 0
                                for heading in data_headings:
                                    data_list[env_count][heading].append(scheduled_data[k])
                                    k += 1
                                env_count += 1

                    ind += 1

                    # write data to file
                    for i_env in range(tot_no_envs):
                        for key, value in data_list[i_env].items():
                            # print(f'Writing data for env {i_env}, heading {key} {value}')
                            data = pd.DataFrame([val.item() for val in value])
                            data.to_excel(writers[i_env], sheet_name=key, index=False, header=False, startcol=i_ins + 1, startrow=1)

                    print("finish env {0}".format(i_ins))
                    print("env_spend_time: ", time.time() - step_time_last)
                    # print(f'util {env.state.feat_mas_batch[:, 2, :]}')
                print("total_spend_time: ", time.time() - start)
                for writer in writers:
                    writer.close()


def schedule(env, model, memories, flag_greedy=False, index=0, flag_filter_machines=False, flag_num_to_filter=5,
             negotiate_rule="SPT"):
    # Get state and completion signal
    state = env.state
    num_mas = env.num_mas
    done = False  # Unfinished at the beginning
    last_time = time.time()

    # directory_opes = 'feature_logs/6920/opes_high_utils_impact/'
    # directory_mas = 'feature_logs/6920/mas_high_utils_impact/'
    # os.makedirs(directory_opes, exist_ok=True)
    # doc_opes = open(directory_opes + '{0}.txt'.format(index), 'a')
    #
    # os.makedirs(directory_mas, exist_ok=True)
    # doc_mas = open(directory_mas + '{0}.txt'.format(index), 'a')

    # print(f'time: {env.time}')
    # print(f'num jobs in sys: {env.num_jobs_system}')
    # print(f'tardiness: {(env.feat_opes_batch[:, 4, :] - env.feat_opes_batch[:, 6, :]).gather(1, env.end_ope_biases_batch)}')
    # print(f'deadlines: {env.deadlines_batch}')
    # print(f'completion times: {env.feat_opes_batch[:, 4, :].gather(1, env.end_ope_biases_batch)}')
    # print(f'num late: {env.num_jobs_system_late}')
    dupes = 0
    decisions = 0
    while ~done:
        with torch.no_grad():
            h_mas_glob, h_proc_glob, h_opes_glob, ope_ma = model.policy_old.get_global_features(state, memories,
                                                                                                flag_train=False, flag_filter_machines=flag_filter_machines, flag_num_to_filter=flag_num_to_filter)
            shared_feats = (h_mas_glob, h_proc_glob, h_opes_glob)
            # Convert the tensors to NumPy arrays
            shared_feats_mas = (torch.round(shared_feats[0]*100)/100).cpu().numpy().reshape(-1, shared_feats[0].shape[-1])
            shared_feats_opes = (torch.round(shared_feats[2]*100)/100).cpu().numpy().reshape(-1, shared_feats[2].shape[-1])

            # print(shared_feats_mas, file=doc_mas)
            # print(shared_feats_opes, file=doc_opes)

            actions, dup = model.policy_old.get_actions(num_mas, state, memories, shared_feats, ope_ma, flag_train=False,
                                                   flag_greedy=flag_greedy, negotiate_rule=negotiate_rule)
            dupes += int(dup)
            decisions += 1
        ma = 0
        for ma_action in actions:
            if ma_action[2, :] == -1:
                print(f'ma {ma}: {ma_action[2, :]}')
            ma += 1
        state, rewards, dones = env.step(actions)  # environment transit
        # print(f'time: {env.time}')
        # print(f'num jobs in sys: {env.num_jobs_system}')
        # print(f'tardiness: {(env.feat_opes_batch[:, 4, :] - env.feat_opes_batch[:, 6, :]).gather(1, env.end_ope_biases_batch)}')
        # print(f'deadlines: {env.deadlines_batch}')
        # print(f'completion times: {env.feat_opes_batch[:, 4, :].gather(1, env.end_ope_biases_batch)}')
        # print(f'num late: {env.num_jobs_system_late}')

        done = dones.all()
        memories.clear_action_probs()
        memories.clear_attention()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    print("spend_time: ", spend_time)

    tardiness = copy.deepcopy(env.true_tardiness_batch).squeeze()
    cumul_tardiness = copy.deepcopy(env.true_tardiness_batch_cumul).squeeze()
    num_jobs_completed = copy.deepcopy(env.tot_scheduled_jobs).squeeze()
    num_jobs_late = copy.deepcopy(env.tot_scheduled_jobs_late).squeeze()
    num_jobs_sys = copy.deepcopy(env.num_jobs_system).squeeze()
    num_jobs_sys_late = copy.deepcopy(env.num_jobs_system_late).squeeze()
    est_sys_tardiness = copy.deepcopy(env.tardiness_batch).squeeze()
    time_to_deadline = env.time.unsqueeze(1) - env.deadlines_batch
    time_to_deadline = torch.max(torch.zeros_like(time_to_deadline), time_to_deadline)
    sum_time_to_deadline = torch.sum(time_to_deadline)
    # doc_opes.close()
    # doc_mas.close()

    return tardiness, cumul_tardiness, \
           num_jobs_completed, num_jobs_late, \
           num_jobs_sys, num_jobs_sys_late, \
           est_sys_tardiness, sum_time_to_deadline, torch.tensor(dupes), torch.tensor(decisions)


def HeuristicRule(env_paras, env, rule="EED", mas_rule="SPT"):
    '''
      Validate the policy during training, and the process is similar to test
      '''
    num_mas = env_paras["num_mas"]
    state = env.state
    done = False
    last_time = time.time()
    while ~done:
        with torch.no_grad():
            group_actions = assignments(state, num_mas, rule, mas_rule)

        state, rewards, dones = env.step(group_actions)
        done = dones.all()

    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    print(f'Rule {rule}_{mas_rule} time: {spend_time}')
    # print(f'util {env.state.feat_mas_batch[:, 2, :]}')

    tardiness = copy.deepcopy(env.true_tardiness_batch).squeeze()
    cumul_tardiness = copy.deepcopy(env.true_tardiness_batch_cumul).squeeze()
    num_jobs_completed = copy.deepcopy(env.tot_scheduled_jobs).squeeze()
    num_jobs_late = copy.deepcopy(env.tot_scheduled_jobs_late).squeeze()
    num_jobs_sys = copy.deepcopy(env.num_jobs_system).squeeze()
    num_jobs_sys_late = copy.deepcopy(env.num_jobs_system_late).squeeze()
    est_sys_tardiness = copy.deepcopy(env.tardiness_batch).squeeze()
    time_to_deadline = env.time.unsqueeze(1) - env.deadlines_batch
    time_to_deadline = torch.max(torch.zeros_like(time_to_deadline), time_to_deadline)
    sum_time_to_deadline = torch.sum(time_to_deadline)
    return tardiness, cumul_tardiness, \
           num_jobs_completed, num_jobs_late, \
           num_jobs_sys, num_jobs_sys_late, \
           est_sys_tardiness, sum_time_to_deadline

def assignments(state, num_mas, job_rule="EDD", mas_rule="SPT"):
    batch_idxes = state.batch_idxes

    eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1, state.ope_step_batch[..., :, None].expand(-1, -1,  state.ope_ma_adj_batch.size(-1))[batch_idxes])
    # Matrix indicating whether machine is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(eligible_proc)
    ma_procing_batch = copy.deepcopy(state.mask_ma_procing_batch)
    # Matrix indicating whether job is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                     state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(eligible_proc)
    # shape: [len(batch_idxes), num_jobs, num_mas]
    eligible = job_eligible & ma_eligible & (eligible_proc == 1)

    group_actions = []
    for ma in range(num_mas):
        mas = torch.ones(state.batch_idxes.size()).long() * ma
        jobs = torch.ones(state.batch_idxes.size()).long() * -2
        opes = torch.ones(state.batch_idxes.size()).long() * -2
        actions = torch.stack((opes, mas, jobs), dim=1).t()
        group_actions.append(actions)

    deadlines = state.deadlines_batch[batch_idxes, :]
    completion_times = state.feat_opes_batch[:, 4, :].gather(1, state.end_ope_biases_batch)
    sorted_indices_job = torch.arange(job_eligible.size(1)).unsqueeze(0)

    if job_rule == "EDD":
        # Sort the deadlines along the second dimension (i.e., by job)
        sorted_indices_job = torch.argsort(deadlines, dim=1, stable=True)
    elif job_rule == "FIFO":
        sorted_indices_job = torch.arange(job_eligible.size(1)).unsqueeze(0)
    elif job_rule == "MST":
        slack_time = deadlines - completion_times #todo: CHECK
        # Sort the jobs by slack time
        sorted_indices_job = torch.argsort(slack_time, dim=1, stable=True)
    elif job_rule == "CR":
        n = deadlines - state.time_batch.expand_as(deadlines)
        d = completion_times - state.time_batch.expand_as(completion_times)
        critical_ratio = torch.div(n, d)
        # Sort the jobs by critic ratio
        sorted_indices_job = torch.argsort(critical_ratio, dim=1, stable=True)
    elif job_rule == "LWKR":
        LWKR = completion_times - state.time_batch.expand_as(completion_times)
        sorted_indices_job = torch.argsort(LWKR, dim=1, stable=True)
    elif job_rule == "LOR":
        LOR = state.feat_opes_batch[:, 3, :].gather(1, state.end_ope_biases_batch)
        sorted_indices_job = torch.argsort(LOR, dim=1, stable=True)

    count_assigned = [0 for _ in batch_idxes]

    for i in batch_idxes:
        # for each job in order of due date
        for j in range(sorted_indices_job.size()[1]):
            # if all machines are processing break loop
            if torch.all(ma_procing_batch[i]):
                break

            job_idx = sorted_indices_job[i, j]
            # get op and ope proc times
            ope_step_batch = state.ope_step_batch
            ope = ope_step_batch[i, job_idx]
            sorted_indices_ma = torch.arange(ma_procing_batch.size(1)).unsqueeze(0)

            if mas_rule == "SPT":
                # sort machine by SPT
                ope_proc_time = state.proc_times_batch[i, ope, :]
                sorted_indices_ma = torch.argsort(ope_proc_time, stable=True)
            elif mas_rule == "EET":
                avail_time = state.feat_mas_batch[i, 1, :]
                sorted_indices_ma = torch.argsort(avail_time, stable=True)
            elif mas_rule == "LLM":
                util = state.feat_mas_batch[i, 2, :]
                sorted_indices_ma = torch.argsort(util, stable=True)

            # for each sorted machine
            for ma in sorted_indices_ma:
                if group_actions[ma][0, i] >= 0:  # if machine has already been assigned skip
                    continue
                if eligible[i, job_idx, ma]:
                    count_assigned[i] += 1
                    group_actions[ma][0, i] = ope
                    group_actions[ma][2, i] = job_idx
                    ma_procing_batch[i, ma] = True
                    break

    return group_actions


if __name__ == '__main__':
    main()
