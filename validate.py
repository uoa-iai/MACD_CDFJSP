import gym
from numpy import sort

import PPO_model
import torch
import time
import os
import copy

def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    # file_path = "./data_dev/validation/validation_set_no_flex/{0}{1}/".format(env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2))
    file_path = "./data_dev/validation/5mas10prodHurink/"
    valid_data_files = sorted(os.listdir(file_path))
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    print(f'valid_data_files {valid_data_files}')
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env

def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    num_mas = env_paras["num_mas"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    while ~done:
        with torch.no_grad():
            h_mas_glob, h_proc_glob, h_opes_glob, ope_ma = model_policy.get_global_features(state, memory,
                                                                                            flag_train=False)
            shared_feats = (h_mas_glob, h_proc_glob, h_opes_glob)
            actions = model_policy.get_actions(num_mas, state, memory, shared_feats, ope_ma, flag_train=False,
                                               flag_greedy=False, negotiate_rule=env_paras["negotiate_rule"])
            state, rewards, dones = env.step(actions)  # environment transit
            done = dones.all()
            memory.clear_action_probs()

    tardiness = copy.deepcopy((env.true_tardiness_batch).mean())
    tot_scheduled_jobs = copy.deepcopy((env.tot_scheduled_jobs).to(torch.float).mean())
    tardiness_batch = copy.deepcopy(env.true_tardiness_batch)
    tot_scheduled_jobs_batch = copy.deepcopy(env.tot_scheduled_jobs)
    effective_tardiness = copy.deepcopy((env.tardiness_batch).mean())

    env.reset()
    print('validating time: ', time.time() - start, '\n')

    return tardiness, tardiness_batch, tot_scheduled_jobs, tot_scheduled_jobs_batch, effective_tardiness


def EED_SPT(env_paras, env, testing=False):
    '''
      Validate the policy during training, and the process is similar to test
      '''
    num_mas = env_paras["num_mas"]
    state = env.state
    done = False
    while ~done:
        with torch.no_grad():
            group_actions = act_CR_SPT(state, num_mas)
            # print(f'group actions {group_actions}')
        state, rewards, dones = env.step(group_actions)
        done = dones.all()
        #print(f'env.num_jobs_system {env.num_jobs_system[5]}')

    tardiness = copy.deepcopy((env.true_tardiness_batch).mean())
    tot_scheduled_jobs = copy.deepcopy((env.tot_scheduled_jobs).to(torch.float).mean())
    tardiness_batch = copy.deepcopy(env.true_tardiness_batch)
    tot_scheduled_jobs_batch = copy.deepcopy(env.tot_scheduled_jobs)

    if not testing:
        env.reset()
        return tardiness, tardiness_batch, tot_scheduled_jobs, tot_scheduled_jobs_batch
    else:
        tardiness = copy.deepcopy(env.true_tardiness_batch).squeeze()
        cumul_tardiness = copy.deepcopy(env.true_tardiness_batch_cumul).squeeze()
        num_jobs_completed = copy.deepcopy(env.tot_scheduled_jobs).squeeze()
        num_jobs_late = copy.deepcopy(env.tot_scheduled_jobs_late).squeeze()
        num_jobs_sys = copy.deepcopy(env.num_jobs_system).squeeze()
        num_jobs_sys_late = copy.deepcopy(env.num_jobs_system_late).squeeze()
        return tardiness, cumul_tardiness, \
               num_jobs_completed, num_jobs_late, \
               num_jobs_sys, num_jobs_sys_late

def act_EDD_SPT(state, num_mas):
    batch_idxes = state.batch_idxes

    eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1, state.ope_step_batch[..., :, None].expand(-1, -1,  state.ope_ma_adj_batch.size(-1))[batch_idxes])
    # Matrix indicating whether machine is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(eligible_proc)
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
    # Sort the deadlines along the second dimension (i.e., by job)
    sorted_deadlines, sorted_indices_job = torch.sort(deadlines, dim=1)

    count_assigned = [0 for _ in batch_idxes]

    for i in batch_idxes:
        for j in range(sorted_indices_job.size()[1]):
            job_idx = sorted_indices_job[i, j]

            ope_step_batch = state.ope_step_batch
            ope = ope_step_batch[i, job_idx]
            ope_proc_time = state.proc_times_batch[i, ope, :]
            sorted_indices_ma = torch.argsort(ope_proc_time)

            for ma in sorted_indices_ma:
                if group_actions[ma][0, i] >= 0:  # if machine has already been assigned skip
                    continue
                if eligible[i, job_idx, ma]:
                    count_assigned[i] += 1
                    assigned_ma = ma
                    group_actions[ma][0, i] = ope
                    group_actions[ma][2, i] = job_idx
                    break
            if count_assigned[i] >= num_mas:
                break

    return group_actions

def act_CR_SPT(state, num_mas):
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

    n = deadlines - state.time_batch.unsqueeze(1).expand_as(deadlines)
    d = completion_times - state.time_batch.unsqueeze(1).expand_as(completion_times)
    critical_ratio = torch.div(n, d)
    # Sort the jobs by critic ratio
    sorted_indices_job = torch.argsort(critical_ratio, dim=1, stable=True)

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

            # sort machine by SPT
            ope_proc_time = state.proc_times_batch[i, ope, :]
            sorted_indices_ma = torch.argsort(ope_proc_time)

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