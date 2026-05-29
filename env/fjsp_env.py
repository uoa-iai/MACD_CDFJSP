import random

import gym
import torch

from dataclasses import dataclass

from env.case_generator import QueueModel
from env.load_data import load_fjs, nums_detec, load_for_l, num_ma_detec, load_fjs_new, lib_detec
import numpy as np
import copy
import torch.nn.functional as F


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None

    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    batch_idxes: torch.Tensor = None

    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None

    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None

    cal_cumul_adj_batch: torch.Tensor = None
    deadlines_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    nums_ope_batch: torch.Tensor = None
    nums_ope_batch_dynamic: torch.Tensor = None
    num_ope_biases_batch: torch.Tensor = None

    num_jobs_system: torch.Tensor = None

    min_proc: torch.Tensor = None
    max_proc: torch.Tensor = None
    max_flex: torch.Tensor = None
    min_opes: torch.Tensor = None
    max_opes: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time,
               deadlines_batch,
               cal_cumul_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, opes_appertain_batch, end_ope_biases_batch,
               nums_opes_batch, nums_ope_batch, nums_ope_batch_dynamic, num_ope_biases_batch, num_jobs_system, min_proc, max_proc, max_flex, min_opes, max_opes):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch
        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time

        self.deadlines_batch = deadlines_batch
        self.cal_cumul_adj_batch = cal_cumul_adj_batch
        self.ope_pre_adj_batch = ope_pre_adj_batch
        self.ope_sub_adj_batch = ope_sub_adj_batch
        self.opes_appertain_batch = opes_appertain_batch
        self.end_ope_biases_batch = end_ope_biases_batch
        self.nums_opes_batch = nums_opes_batch
        self.nums_ope_batch = nums_ope_batch
        self.nums_ope_batch_dynamic = nums_ope_batch_dynamic
        self.num_ope_biases_batch = num_ope_biases_batch

        self.num_jobs_system = num_jobs_system

        self.min_proc = min_proc
        self.max_proc = max_proc
        self.max_flex = max_flex
        self.min_opes = min_opes
        self.max_opes = max_opes


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)


def get_rngs(seed_val, num_batches):
    '''
    Return three lists of RNGs for each batch with different seed values
    '''

    # Create empty lists to store random number generators for each type of data
    rngs_job_idx = []  # For indexing job library
    rngs_job_arr = []  # For job arrival
    rngs_ddt = []  # For due date tightness
    rngs_ddt_select = []  # For due date tightness range selection
    rngs_init_jobs = []  # For no. init jobs

    # Loop over the number of batches
    for i in range(num_batches):
        # Create three separate random number generators using different seed values
        rng_job_idx = np.random.default_rng(seed_val + i)  # For indexing job library
        rng_job_arr = np.random.default_rng(seed_val + i)  # For job arrival
        rng_ddt = np.random.default_rng(seed_val + i)  # For due date tightness
        rng_ddt_select = np.random.default_rng(seed_val + i)  # For due date tightness range selection
        rng_init_jobs = np.random.default_rng(seed_val + i)  # For no. of init jobs

        # Append the random number generators to the corresponding lists
        rngs_job_idx.append(rng_job_idx)
        rngs_job_arr.append(rng_job_arr)
        rngs_ddt.append(rng_ddt)
        rngs_ddt_select.append(rng_ddt_select)
        rngs_init_jobs.append(rng_init_jobs)

    # Return the lists of random number generators
    return rngs_job_idx, rngs_job_arr, rngs_ddt, rngs_ddt_select, rngs_init_jobs


class FJSPEnv(gym.Env):
    '''
    FJSP environment
    '''

    def __init__(self, case, env_paras, data_source='case'):
        '''
        :param case: The instance generator or the addresses of the instances
        :param env_paras: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files
        '''

        # load paras
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs in library of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines in environment
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch
        seed_val = env_paras["seed"]  # seed for repeatable values
        self.ma_util = env_paras["ma_util"]  # set machine utilisation
        self.DDT_high = env_paras["DDT_high"]  # set upper bound of due date tightness
        self.DDT_low = env_paras["DDT_low"]  # set lower bound of due date tightness
        self.tight_percent = env_paras["tight_percent"]
        self.time_limit = env_paras["time_period"]  # max time before terminating simulation
        self.time_period = env_paras["time_period"]  # take measurements every T time steps
        self.periods_counted = 1  # count periods for continuous testing case so can terminate after x periods
        self.continuous = env_paras["continuous"]  # true if no terminal condition, false if terminal condition

        # load instance
        self.num_data = 10  # The amount of data extracted from instance
        tensors = [[] for _ in range(self.num_data)]

        self.num_opes = 0  # max num of opes in overall system
        self.max_jobs = 0  # max num of jobs in overall system
        self.num_opes_system = torch.zeros(self.batch_size)  # num opes in each batch
        self.num_jobs_system = torch.zeros(self.batch_size)  # num of jobs in each batch
        self.num_jobs_system_late = torch.zeros(self.batch_size)
        self.tot_jobs_added = torch.zeros(self.batch_size)  # keeps track of tot no. jobs added for each case
        self.tot_ops_added = torch.zeros(self.batch_size)  # keeps track of tot no. ops added for each case
        self.tot_scheduled_jobs = torch.zeros(self.batch_size).int()  # Count scheduled jobs
        self.tot_scheduled_jobs_late = torch.zeros(self.batch_size).int()
        self.tot_scheduled_ops = torch.zeros(self.batch_size).int()  # Count scheduled operations

        self.arrival_times = [[] for _ in range(self.batch_size)]  # arrival times for upcoming jobs
        self.arrival_rates = []  # arrival rates
        self.inital_jobs = env_paras["init_jobs"]  # jobs at steady state/or initial number of jobs in system
        self.inital_jobs_max = env_paras["init_jobs"]

        if env_paras["init_jobs"] is None:
            self.inital_jobs = 1
        self.library = []  # keeps track of job library of each instance

        # set seeds
        self.rngs_job_idx, self.rngs_job_arr, self.rngs_ddt, self.rngs_ddt_select, self.rngs_init_jobs = get_rngs(seed_val, self.batch_size)
        added_jobs = [[] for _ in range(self.batch_size)]
        # self.inital_jobs = [random.randint(1, env_paras["init_jobs"]) for _ in range(self.batch_size)]

        self.min_proc = torch.zeros(self.batch_size)
        self.max_proc = torch.zeros(self.batch_size)
        self.max_flex = torch.zeros(self.batch_size)
        self.min_opes = torch.zeros(self.batch_size)
        self.max_opes = torch.zeros(self.batch_size)

        if data_source == 'case':  # Generate instances through generators
            arrival_rate, L = case.get_arrival_rate()  # get arrival rate of case generator
            self.arrival_rates = [arrival_rate for _ in range(self.batch_size)]
            for i in range(self.batch_size):
                self.library.append(case.get_case(seed_val, i)[0])  # Generate an instance and save it
                self.initialise_arrival_times(i)
                num_jobs, num_mas, num_opes = nums_detec(self.library[i])
                min_proc, max_proc, max_flex, min_opes, max_opes = lib_detec(self.library[i], self.num_mas, num_opes, self.num_jobs)
                self.min_proc[i] = min_proc
                self.max_proc[i] = max_proc
                self.max_flex[i] = max_flex
                self.min_opes[i] = min_opes
                self.max_opes[i] = max_opes

                added_jobs[i].append(self.library[i][0])
                while (self.arrival_times[i][0] <= 0):
                    job_idx = self.rngs_job_idx[i].integers(1, self.num_jobs)
                    added_jobs[i].append(self.library[i][job_idx])
                    self.arrival_times[i].pop(0)
                    self.tot_jobs_added[i] += 1
                num_jobs, num_mas, num_opes = nums_detec(added_jobs[i])
                self.num_opes_system[i] += num_opes
                self.num_jobs_system[i] += num_jobs

                self.tot_ops_added[i] += num_opes
                self.next_arrival_times(i)
        else:  # Load instances from files
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    self.library.append(line)
                num_jobs, num_mas, num_opes = nums_detec(self.library[i])
                min_proc, max_proc, max_flex, min_opes, max_opes = lib_detec(self.library[i], self.num_mas, num_opes,
                                                                             self.num_jobs)
                self.min_proc[i] = min_proc
                self.max_proc[i] = max_proc
                self.max_flex[i] = max_flex
                self.min_opes[i] = min_opes
                self.max_opes[i] = max_opes

                added_jobs[i].append(self.library[i][0])
                service_rate, mas_per_ope, opes_per_job = load_for_l(self.library[i])
                arrival_rate = self.ma_util * self.num_mas * service_rate
                # print(f'ma util {self.ma_util}')
                # print(f'mas {self.num_mas}')
                self.arrival_rates.append(arrival_rate)
                QM = QueueModel(arrival_rate, service_rate, self.num_mas, self.ma_util)
                L = QM.number_in_sys()
                print(f'L {L}')
                self.initialise_arrival_times(i)
                while (self.arrival_times[i][0] <= 0):
                    job_idx = self.rngs_job_idx[i].integers(1, self.num_jobs)
                    # print(f'job_idx {job_idx}')
                    added_jobs[i].append(self.library[i][job_idx])
                    self.arrival_times[i].pop(0)
                    self.tot_jobs_added[i] += 1
                num_jobs, num_mas, num_opes = nums_detec(added_jobs[i])
                self.num_opes_system[i] += num_opes
                self.num_jobs_system[i] += num_jobs

                self.tot_ops_added[i] += num_opes
                self.next_arrival_times(i)

        time_in_period = int(max(self.max_proc) * max(self.max_opes) / num_mas * 10.0)
        env_paras["time_period"] = time_in_period
        self.time_limit = env_paras["time_period"]  # max time before terminating simulation
        self.time_period = env_paras["time_period"]  # take measurements every T time steps

        # Records the maximum number of operations in the parallel instances
        self.num_opes = int(max(self.num_opes_system))
        self.max_jobs = int(max(self.num_jobs_system))

        self.DDT_a = env_paras["DDT_a"]
        # load feats
        for i in range(self.batch_size):
            if self.DDT_a:
                random_number = self.rngs_ddt_select[i].random()
                if random_number < self.tight_percent:
                    load_data = load_fjs(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs, self.rngs_ddt[i],
                                         DDT_high=self.DDT_low+0.5, DDT_low=self.DDT_low)
                else:
                    load_data = load_fjs(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs, self.rngs_ddt[i],
                                         DDT_high=self.DDT_high, DDT_low=self.DDT_low+0.5)
            else:
                load_data = load_fjs(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs, self.rngs_ddt[i],
                                     DDT_high=self.DDT_high, DDT_low=self.DDT_low)

            for j in range(self.num_data - 1):
                tensors[j].append(load_data[j])

        # feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        self.nums_ope_batch_dynamic = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)
        # shape: (batch_size, num_jobs), the deadline for each job
        self.deadlines_batch = torch.stack(tensors[8], dim=0).long()
        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment

        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch < 0, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch >= self.num_opes, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch < 0, self.num_opes - 1, self.ope_step_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch >= self.num_opes, self.num_opes - 1, self.ope_step_batch)
        self.opes_appertain_batch = torch.where(self.opes_appertain_batch < 0, 0, self.opes_appertain_batch)
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
                Job Deadline
            ma:
                Number of neighboring operations
                Available time
                Utilization
        '''
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 6, :] = convert_feat_job_2_ope(self.deadlines_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                 fill_value=False)

        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)

        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))
        self.machines_batch[:, :, 3] = self.machines_batch[:, :, 3] - 1
        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]

        mask = self.feat_opes_batch[:, 6, :] != 0
        tardy_ratio = self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]
        tardy_ratio[~mask] = 0
        tardy_ratio = tardy_ratio.gather(1, self.end_ope_biases_batch)

        completion_times = self.feat_opes_batch[:, 4, :]
        completion_times[~mask] = 0
        completion_times = completion_times.gather(1, self.end_ope_biases_batch)

        tardy_ratio_clip = torch.max(torch.zeros_like(tardy_ratio), tardy_ratio)
        tardy_ratio_clip_batch = torch.sum(tardy_ratio_clip, dim=1)
        completion_batch = torch.sum(completion_times, dim=1)
        self.tardiness_batch = tardy_ratio_clip_batch
        self.completion_batch = completion_batch
        self.true_tardiness_batch_cumul = torch.zeros(self.batch_size)
        self.true_completion_batch_cumul = torch.zeros(self.batch_size)
        self.true_tardiness_batch = torch.zeros(self.batch_size)

        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
        self.old_reward = torch.zeros(self.batch_size)

        for i in self.batch_idxes:
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
            self.mask_job_procing_batch[i, int(self.num_jobs_system[i]):] = True
            self.mask_job_finish_batch[i, int(self.num_jobs_system[i]):] = True

        self.prev_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch,
                              feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch,
                              ope_ma_adj_batch=self.ope_ma_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              ope_step_batch=self.ope_step_batch,
                              time_batch=self.time,
                              deadlines_batch=self.deadlines_batch,
                              cal_cumul_adj_batch=self.cal_cumul_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch,
                              ope_sub_adj_batch=self.ope_sub_adj_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              nums_opes_batch=self.nums_opes,
                              nums_ope_batch=self.nums_ope_batch,
                              nums_ope_batch_dynamic=self.nums_ope_batch_dynamic,
                              num_ope_biases_batch=self.num_ope_biases_batch,
                              num_jobs_system = self.num_jobs_system,
                              min_proc = self.min_proc,
                              max_proc = self.max_proc,
                              max_flex = self.max_flex,
                              min_opes = self.min_opes,
                              max_opes = self.max_opes
        )

        # Save initial data for reset - only includes dynamic features
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

        self.old_min_proc = copy.deepcopy(self.min_proc)
        self.old_max_proc = copy.deepcopy(self.max_proc)
        self.old_max_flex = copy.deepcopy(self.max_flex)
        self.old_min_opes = copy.deepcopy(self.min_opes)
        self.old_max_opes = copy.deepcopy(self.max_opes)

        self.old_num_opes_system = copy.deepcopy(self.num_opes_system)
        self.old_num_jobs_system = copy.deepcopy(self.num_jobs_system)

        self.old_tot_jobs_added = copy.deepcopy(self.tot_jobs_added)
        self.old_tot_ops_added = copy.deepcopy(self.tot_ops_added)
        self.old_arrival_times = copy.deepcopy(self.arrival_times)

        self.old_ope_pre_adj_batch = copy.deepcopy(self.ope_pre_adj_batch)
        self.old_ope_sub_adj_batch = copy.deepcopy(self.ope_sub_adj_batch)
        self.old_opes_appertain_batch = copy.deepcopy(self.opes_appertain_batch)

        self.old_ope_step_batch = copy.deepcopy(self.ope_step_batch)
        self.old_end_ope_biases_batch = copy.deepcopy(self.end_ope_biases_batch)
        self.old_nums_ope_batch = copy.deepcopy(self.nums_ope_batch)
        self.old_nums_ope_batch_dynamic = copy.deepcopy(self.nums_ope_batch_dynamic)
        self.old_deadlines_batch = copy.deepcopy(self.deadlines_batch)
        self.old_num_ope_biases_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.old_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)
        self.old_mask_job_finish_batch = copy.deepcopy(self.mask_job_finish_batch)

        # get rng states
        self.old_rngs_job_idx_state = [rng_job_idx_state.__getstate__() for rng_job_idx_state in self.rngs_job_idx]
        self.old_rngs_job_arr_state = [rng_job_arr_state.__getstate__() for rng_job_arr_state in self.rngs_job_arr]
        self.old_rngs_ddt_state = [rng_ddt_state.__getstate__() for rng_ddt_state in self.rngs_ddt]
        self.old_rngs_ddt_select_state = [rng_ddt_select_state.__getstate__() for rng_ddt_select_state in self.rngs_ddt_select]
        self.old_rngs_init_jobs_state = [rng_init_job_state.__getstate__() for rng_init_job_state in self.rngs_init_jobs]

    def step(self, group_actions):
        '''
        Environment transition function
        '''

        self.reward_batch = torch.zeros(self.batch_size)
        self.future_reward_batch = torch.zeros(self.batch_size)
        feat_opes_batch_copy = copy.deepcopy(self.feat_opes_batch)

        for ma in range(self.num_mas):
            actions = group_actions[ma]
            if actions is None:
                continue
            jobs_working = copy.deepcopy(actions[2, :])
            if torch.all(jobs_working < 0):
                continue
            non_continue_idxs = torch.nonzero(jobs_working > -1).squeeze(-1)
            jobs_temp = torch.max(torch.zeros_like(jobs_working), jobs_working)
            mask_processing_jobs = self.mask_job_procing_batch[non_continue_idxs, jobs_temp[non_continue_idxs]]
            jobs_working[non_continue_idxs] = ~mask_processing_jobs * jobs_working[non_continue_idxs] + mask_processing_jobs * -1
            # filter out batches where the action is "do nothing"
            self.batch_idxes = torch.nonzero(jobs_working > -1).squeeze(-1)
            batch_idxes_wait = torch.nonzero(jobs_working == -1).squeeze(-1)
            opes = actions[0, :][self.batch_idxes]
            mas = actions[1, :][self.batch_idxes]
            jobs = actions[2, :][self.batch_idxes]

            if self.batch_size == 100:
                if actions[2, :][40] != -2:
                    # print(f'actions[2, :][self.batch_idxes] {actions[2, :][40]}')
                    pass

            opes_wait = actions[0, :][batch_idxes_wait]
            mas_wait = actions[1, :][batch_idxes_wait]
            jobs_wait = actions[2, :][batch_idxes_wait]

            self.tot_scheduled_ops[self.batch_idxes] += 1

            j = 0
            for i in self.batch_idxes:
                # print(f'mas {ma} assigned to op {opes[j]} of job {jobs[j]}')
                try:
                    if self.feat_opes_batch[i, 0, opes[j]] == 1:
                        print(f'i {i}')
                        print(f'Scheduling already scheduled operation')
                        print(f'opes {opes[j]}')
                        print(f'feats {self.feat_opes_batch[i, 0, :]}')
                        raise Exception
                except:
                    print(f'time {self.time}')
                j += 1

            # Removed unselected O-M arcs of the scheduled operations
            remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
            remain_ope_ma_adj[self.batch_idxes, mas] = 1
            self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
            self.proc_times_batch *= self.ope_ma_adj_batch

            # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
            proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
            proc_times_wait = self.proc_times_batch[batch_idxes_wait, opes_wait, mas_wait]
            self.future_reward_batch[batch_idxes_wait] += feat_opes_batch_copy[batch_idxes_wait, 2, opes_wait] - proc_times_wait
            if (proc_times == 0).any():
                print(f'You are assigning an incompatible op to machine {ma}')
                raise Exception
            # if (proc_times_wait == 0).any():
            #     print(f'You are assigning an incompatible op to machine {ma} in the future')
            #     raise Exception
            self.feat_mas_batch[self.batch_idxes, 4, mas] = proc_times
            self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
                (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                 torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                 proc_times), dim=1)
            last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1,
                                    opes - 1)
            self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

            # Update 'Number of unscheduled operations in the job'
            start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
            end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
            self.nums_ope_batch_dynamic[self.batch_idxes, jobs] -= 1
            for i in range(self.batch_idxes.size(0)):
                self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i] + 1] -= 1
            # Update 'Start time' and 'Job completion time'
            self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
            is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
            mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
            start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
            un_scheduled = 1 - is_scheduled  # unscheduled opes
            estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                       self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() \
                             * un_scheduled  # estimate start time of unscheduled opes
            start_ope = self.ope_step_batch[self.batch_idxes, jobs]
            end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
            # for i in range(self.batch_idxes.size(0)):
            #     self.feat_opes_batch[self.batch_idxes[i], 5, start_ope[i]:end_ope[i] + 1] -= 1
            self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
            end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                              self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[
                                                                                      self.batch_idxes, :])
            self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[
                                                                                                  self.batch_idxes, :])
            self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
            self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
            self.machines_batch[self.batch_idxes, mas, 2] += proc_times
            self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

            # Update feature vectors of machines
            self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :],
                                                                              dim=1).float()
            self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
            utiliz = self.machines_batch[self.batch_idxes, :, 2]
            cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
            utiliz = torch.minimum(utiliz, cur_time)
            utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
            self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

            # Update other variable according to actions
            self.ope_step_batch[self.batch_idxes, jobs] += 1
            self.mask_job_procing_batch[self.batch_idxes, jobs] = True
            self.mask_ma_procing_batch[self.batch_idxes, mas] = True

            cloned_ope_ma_adj_batch = self.ope_ma_adj_batch.clone()
            cloned_proc_times_batch = self.proc_times_batch.clone()
            cloned_cal_cumul_adj_batch = self.cal_cumul_adj_batch.clone()
            cloned_ope_pre_adj_batch = self.ope_pre_adj_batch.clone()
            cloned_ope_sub_adj_batch = self.ope_sub_adj_batch.clone()
            cloned_opes_appertain_batch = self.opes_appertain_batch.clone()
            cloned_feat_opes_batch = self.feat_opes_batch.clone()

            cloned_ope_step_batch = self.ope_step_batch.clone()
            cloned_end_ope_biases_batch = self.end_ope_biases_batch.clone()
            cloned_nums_ope_batch = self.nums_ope_batch.clone()
            cloned_nums_ope_batch_dynamic = self.nums_ope_batch_dynamic.clone()
            cloned_deadlines_batch = self.deadlines_batch.clone()
            cloned_num_ope_biases_batch = self.num_ope_biases_batch.clone()
            cloned_mask_job_procing_batch = self.mask_job_procing_batch.clone()
            cloned_mask_job_finish_batch = self.mask_job_finish_batch.clone()

            # remove any jobs where all operations have been scheduled
            j = 0
            for i in self.batch_idxes:
                job_idx = int(jobs[j])
                mas_idx = int(mas[j])
                j += 1
                if self.ope_step_batch[i, job_idx] == self.end_ope_biases_batch[i, job_idx] + 1:
                    # if i == 54:
                    # print(f'JOB COMPLETE {job_idx} bi {i}')
                    start_ope = int(self.num_ope_biases_batch[i, job_idx])  # index of starting op
                    end_ope = int(self.end_ope_biases_batch[i, job_idx])  # index of ending
                    num_rmd = int(self.feat_opes_batch.shape[2]) - end_ope - 1  # no. of elems. b/w end and last op
                    self.num_jobs_system[i] -= 1
                    self.num_opes_system[i] -= (end_ope - start_ope + 1)
                    self.tot_scheduled_jobs[i] += 1

                    temp_true_tardiness_batch = self.feat_opes_batch[i, 4, end_ope] - self.feat_opes_batch[
                        i, 6, end_ope]
                    temp_true_tardiness_batch = torch.where(temp_true_tardiness_batch < torch.tensor(0.0),
                                                            torch.tensor(0.0), temp_true_tardiness_batch)

                    self.true_tardiness_batch_cumul[i] += temp_true_tardiness_batch
                    self.true_completion_batch_cumul[i] += self.feat_opes_batch[i, 4, end_ope]
                    self.true_tardiness_batch[i] += temp_true_tardiness_batch

                    if temp_true_tardiness_batch > torch.tensor(0.0):
                        self.tot_scheduled_jobs_late[i] += 1

                    self.ope_ma_adj_batch[i, start_ope:start_ope + num_rmd, :] = cloned_ope_ma_adj_batch[i,
                                                                                 end_ope + 1:, :]
                    self.ope_ma_adj_batch[i, int(self.num_opes_system[i]):, :] = 0

                    self.proc_times_batch[i, start_ope:start_ope + num_rmd, :] = cloned_proc_times_batch[i,
                                                                                 end_ope + 1:, :]
                    self.proc_times_batch[i, int(self.num_opes_system[i]):, :] = 0

                    self.cal_cumul_adj_batch[i, start_ope:start_ope + num_rmd, start_ope:start_ope + num_rmd] = \
                        cloned_cal_cumul_adj_batch[i, end_ope + 1:, end_ope + 1:]
                    self.cal_cumul_adj_batch[i, int(self.num_opes_system[i]):, int(self.num_opes_system[i]):] = 0
                    self.ope_pre_adj_batch[i, start_ope:start_ope + num_rmd, :] = \
                        cloned_ope_pre_adj_batch[i, end_ope + 1:, :]
                    self.ope_pre_adj_batch[i, int(self.num_opes_system[i]):, :] = 0
                    self.ope_sub_adj_batch[i, start_ope:start_ope + num_rmd, :] = \
                        cloned_ope_sub_adj_batch[i, end_ope + 1:, :]
                    self.ope_sub_adj_batch[i, int(self.num_opes_system[i]):, :] = 0

                    self.opes_appertain_batch[i, start_ope:start_ope + num_rmd] = \
                        cloned_opes_appertain_batch[i, end_ope + 1:]
                    self.opes_appertain_batch[i, start_ope:] -= 1
                    self.opes_appertain_batch[i, int(self.num_opes_system[i]):] = 0

                    self.feat_opes_batch[i, :, start_ope:start_ope + num_rmd] = cloned_feat_opes_batch[i, :,
                                                                                end_ope + 1:]
                    self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0

                    for mac in range(ma+1, self.num_mas):
                        if group_actions[mac] is None:
                            continue
                        if (group_actions[mac][0, :][i] > end_ope) and (group_actions[mac][2, :][i] > -1):
                            group_actions[mac][0, :][i] -= (end_ope - start_ope + 1)
                            group_actions[mac][2, :][i] -= 1

                    # update job tensors
                    self.ope_step_batch[i, job_idx:self.max_jobs - 1] = cloned_ope_step_batch[i, job_idx + 1:]
                    self.ope_step_batch[i, job_idx:] -= (end_ope - start_ope + 1)
                    self.ope_step_batch[i, int(self.num_jobs_system[i]):] = -1

                    self.end_ope_biases_batch[i, job_idx:self.max_jobs - 1] = cloned_end_ope_biases_batch[i,
                                                                              job_idx + 1:]
                    self.end_ope_biases_batch[i, job_idx:] -= (end_ope - start_ope + 1)
                    self.end_ope_biases_batch[i, int(self.num_jobs_system[i]):] = -1

                    self.nums_ope_batch[i, job_idx:self.max_jobs - 1] = cloned_nums_ope_batch[i, job_idx + 1:]
                    self.nums_ope_batch[i, self.max_jobs - 1] = 0
                    self.nums_ope_batch_dynamic[i, job_idx:self.max_jobs - 1] = \
                        cloned_nums_ope_batch_dynamic[i, job_idx + 1:]
                    self.nums_ope_batch_dynamic[i, self.max_jobs - 1] = 0
                    self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

                    self.deadlines_batch[i, job_idx:self.max_jobs - 1] = cloned_deadlines_batch[i, job_idx + 1:]
                    self.deadlines_batch[i, self.max_jobs - 1] = 0

                    self.num_ope_biases_batch[i, job_idx:self.max_jobs - 1] = cloned_num_ope_biases_batch[i,
                                                                              job_idx + 1:]
                    self.num_ope_biases_batch[i, job_idx:] -= (end_ope - start_ope + 1)
                    self.num_ope_biases_batch[i, int(self.num_jobs_system[i]):] = -1

                    self.mask_job_procing_batch[i, job_idx:self.max_jobs - 1] = \
                        cloned_mask_job_procing_batch[i, job_idx + 1:]
                    self.mask_job_procing_batch[i, self.max_jobs - 1] = True

                    self.mask_job_finish_batch[i, job_idx:self.max_jobs - 1] = cloned_mask_job_finish_batch[i,
                                                                               job_idx + 1:]
                    self.mask_job_finish_batch[i, self.max_jobs - 1] = True

                    job_ids_adjusted = torch.where(self.machines_batch[i, :, 3] > job_idx,
                                                   self.machines_batch[i, :, 3] - 1,
                                                   self.machines_batch[i, :, 3])
                    self.machines_batch[i, :, 3] = job_ids_adjusted
                    self.machines_batch[i, mas_idx, 3] = -1

            self.max_jobs = int(torch.max(self.num_jobs_system))
            self.num_opes = int(torch.max(self.num_opes_system))
            if (len(self.ope_step_batch[0]) > self.max_jobs):  # resize to max number of jobs in system
                self.ope_step_batch = self.ope_step_batch[:, :self.max_jobs]
                self.end_ope_biases_batch = self.end_ope_biases_batch[:, :self.max_jobs]
                self.nums_ope_batch = self.nums_ope_batch[:, :self.max_jobs]
                self.nums_ope_batch_dynamic = self.nums_ope_batch_dynamic[:, :self.max_jobs]
                self.deadlines_batch = self.deadlines_batch[:, :self.max_jobs]
                self.mask_job_procing_batch = self.mask_job_procing_batch[:, :self.max_jobs]
                self.mask_job_finish_batch = self.mask_job_finish_batch[:, :self.max_jobs]
                self.num_ope_biases_batch = self.num_ope_biases_batch[:, :self.max_jobs]
            if (self.ope_ma_adj_batch.size(dim=1) > self.num_opes):  # resize to max number of ops in system
                self.ope_ma_adj_batch = self.ope_ma_adj_batch[:, :self.num_opes, :]
                self.proc_times_batch = self.proc_times_batch[:, :self.num_opes, :]
                self.cal_cumul_adj_batch = self.cal_cumul_adj_batch[:, :self.num_opes, :self.num_opes]
                self.ope_pre_adj_batch = self.ope_pre_adj_batch[:, :self.num_opes, :self.num_opes]
                self.ope_sub_adj_batch = self.ope_sub_adj_batch[:, :self.num_opes, :self.num_opes]
                self.opes_appertain_batch = self.opes_appertain_batch[:, :self.num_opes]
                self.feat_opes_batch = self.feat_opes_batch[:, :, :self.num_opes]

            self.feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[:, :, :], dim=1).float()
            self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch < 0, self.num_opes - 1,
                                                    self.end_ope_biases_batch)
            self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch >= self.num_opes, self.num_opes - 1,
                                                    self.end_ope_biases_batch)
            self.ope_step_batch = torch.where(self.ope_step_batch < 0, self.num_opes - 1, self.ope_step_batch)
            self.ope_step_batch = torch.where(self.ope_step_batch >= self.num_opes, self.num_opes - 1,
                                              self.ope_step_batch)

            self.opes_appertain_batch = torch.where(self.opes_appertain_batch < 0, 0, self.opes_appertain_batch)

        end_time_batch = (self.feat_opes_batch[:, 5, :] +
                          self.feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch[
                                                                                  :, :])
        self.feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch,
                                                                              self.opes_appertain_batch[
                                                                              :, :])

        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)
        limit_reached = torch.where(self.time >= self.time_limit, True, False)
        continuous_tensor = torch.tensor(self.continuous, dtype=torch.bool)
        self.done_batch = (self.mask_job_finish_batch.all(dim=1) & ~continuous_tensor) | limit_reached
        self.done = self.done_batch.all()

        count = 0

        self.next_time()
        if self.continuous:
            self.add_job()
        flag_trans_2_next_time = self.if_no_eligible()

        while ~((~((flag_trans_2_next_time == 0) & (~self.done_batch))).all()):
            self.next_time_2(flag_trans_2_next_time)
            if self.continuous:
                self.add_job()
            # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
            flag_trans_2_next_time = self.if_no_eligible()
            count += 1
            limit_reached = torch.where(self.time >= self.time_limit, True, False)
            continuous_tensor = torch.tensor(self.continuous, dtype=torch.bool)
            self.done_batch = (self.mask_job_finish_batch.all(dim=1) & ~continuous_tensor) | limit_reached
            self.done = self.done_batch.all()
            if count > 10:
                print('infinite loop')
                flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done_batch)
                print(f'flag {flag_need_trans}')
                print(f'tot jobs {self.tot_jobs_added}')
                #  raise Exception

        self.time_shift_feat_opes_batch()
        end_time_batch = (self.feat_opes_batch[:, 5, :] +
                          self.feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch[
                                                                                  :, :])
        self.feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[
                                                                                              :, :])
        for i in range(self.batch_size):
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0

        mask = self.feat_opes_batch[:, 6, :] != 0
        tardy_ratio = self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]
        tardy_ratio[~mask] = 0
        tardy_ratio = tardy_ratio.gather(1, self.end_ope_biases_batch)
        for k in self.batch_idxes:
            tardy_ratio[k, int(self.num_jobs_system[k]):] = 0.0
        tardy_early = torch.min(torch.zeros_like(tardy_ratio), tardy_ratio)
        tardy_late = torch.max(torch.zeros_like(tardy_ratio), tardy_ratio)

        try:
            tardy_rew_component = torch.sum(tardy_late, dim=1).div(self.num_jobs_system) * -1.0
            tardy_rew_component = torch.where(tardy_rew_component< 0.0, tardy_rew_component, 1.0)
            self.reward_batch = tardy_rew_component
            self.old_reward = tardy_rew_component
        except:
            self.reward_batch = torch.zeros(self.batch_size)
        self.reward_batch = torch.nan_to_num(self.reward_batch) + 0.0 * self.future_reward_batch

        completion_times = self.feat_opes_batch[:, 4, :]
        completion_times[~mask] = 0
        completion_times = completion_times.gather(1, self.end_ope_biases_batch)
        tardy_ratio_clip = torch.max(torch.zeros_like(tardy_ratio), tardy_ratio)
        tardy_ratio_clip_batch = torch.sum(tardy_ratio_clip, dim=1) + self.true_tardiness_batch_cumul
        completion_batch = torch.sum(completion_times, dim=1) + self.true_completion_batch_cumul

        # self.reward_batch = -tardy_ratio_clip_batch + self.tardiness_batch

        self.num_jobs_system_late = torch.count_nonzero(tardy_late)
        self.tardiness_batch = tardy_ratio_clip_batch
        self.completion_batch = completion_batch

        self.batch_idxes = torch.arange(self.batch_size)
        self.feat_opes_batch[:, 7, :] = self.time.view(-1, 1)
        self.feat_mas_batch[:, 3, :] = self.time.view(-1, 1)
        self.prev_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)

        # Update state of the environment
        self.state.update(self.batch_idxes,
                          self.feat_opes_batch,
                          self.feat_mas_batch,
                          self.proc_times_batch,
                          self.ope_ma_adj_batch,
                          self.mask_job_procing_batch,
                          self.mask_job_finish_batch,
                          self.mask_ma_procing_batch,
                          self.ope_step_batch,
                          self.time,
                          self.deadlines_batch,
                          self.cal_cumul_adj_batch,
                          self.ope_pre_adj_batch,
                          self.ope_sub_adj_batch,
                          self.opes_appertain_batch,
                          self.end_ope_biases_batch,
                          self.nums_opes,
                          self.nums_ope_batch,
                          self.nums_ope_batch_dynamic,
                          self.num_ope_biases_batch,
                          self.num_jobs_system,
                          self.min_proc,
                          self.max_proc,
                          self.max_flex,
                          self.min_opes,
                          self.max_opes)

        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        # ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
        #                              self.end_ope_biases_batch, self.ope_step_batch)
        ope_step_batch = self.ope_step_batch
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                           self.proc_times_batch.size(
                                                                                               2)))

        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(
            torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
            dim=[1, 2])

        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    # def ma_make_decision(self):
    #     '''
    #     Check if there are still O-M pairs to be processed
    #     '''
    #     # ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
    #     #                              self.end_ope_biases_batch, self.ope_step_batch)
    #     ope_step_batch = self.ope_step_batch
    #     eligible_proc = self.ope_ma_adj_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
    #                                                                                        self.ope_ma_adj_batch.size(-1)))
    #
    #     ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(eligible_proc)
    #     job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
    #         eligible_proc)
    #     # shape: [len(batch_idxes), num_jobs, num_mas]
    #     eligible = job_eligible & ma_eligible & (eligible_proc == 1)
    #     # num_mas, batch_size
    #     self.mask_idxes = torch.any(eligible, dim=1).transpose(0, 1)
    #     self.mask_finish = ~(self.mask_job_finish_batch.all(dim=1))

    # def mini_update(self, actions):
    #     jobs = actions[2, :]
    #     masky = self.mask_job_procing_batch[torch.arange(0, self.batch_size), jobs]
    #     jobs[masky] = -1
    #     sub_batch_idxes = torch.nonzero(jobs > -1).squeeze(-1)
    #     self.mask_job_procing_batch[sub_batch_idxes, jobs[sub_batch_idxes]] = True
    #     self.state.update(self.batch_idxes,
    #                       self.feat_opes_batch,
    #                       self.feat_mas_batch,
    #                       self.proc_times_batch,
    #                       self.ope_ma_adj_batch,
    #                       self.mask_job_procing_batch,
    #                       self.mask_job_finish_batch,
    #                       self.mask_ma_procing_batch,
    #                       self.ope_step_batch,
    #                       self.time,
    #                       self.deadlines_batch,
    #                       self.cal_cumul_adj_batch,
    #                       self.ope_pre_adj_batch,
    #                       self.ope_sub_adj_batch,
    #                       self.opes_appertain_batch,
    #                       self.end_ope_biases_batch,
    #                       self.nums_opes,
    #                       self.nums_ope_batch,
    #                       self.nums_ope_batch_dynamic,
    #                       self.num_ope_biases_batch,
    #                       self.min_proc,
    #                       self.max_proc,
    #                       self.max_flex,
    #                       self.min_opes,
    #                       self.max_opes)

    def next_time(self):
        '''
        Transit to the next time
        overall transition
        '''
        # need to transit
        arrival_times = [self.arrival_times[i][0] for i in range(self.batch_size)]
        arrival_times = torch.FloatTensor(arrival_times)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # min of arrival times and machine available time
        zero_jobs = torch.zeros_like(a)
        b = torch.where(((a < arrival_times[:, None]) & ((self.num_jobs_system[:, None] > zero_jobs) | ~torch.tensor(self.continuous))), a,
                        arrival_times[:, None])
        # Return the minimum
        c_2 = torch.min(b, dim=1)[0]
        # if minimum is less than current time, move to current time + 1, else current time
        e = torch.where((c_2 <= self.time), self.time+1, c_2)
        # Detect the machines that completed (at above time)
        d = torch.where((a <= e[:, None]) & (self.machines_batch[:, :, 0] == 0), True, False)
        if torch.isinf(e).any():
            print(f'Transitioning to time=inf')
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1


        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        aa[d, 3] = -1  # set index of job for machine as -1
        self.machines_batch = aa.transpose(1, 2)

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False

        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def next_time_2(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        Transition till feasible machine-operation pair
        '''

        arrival_times = [self.arrival_times[i][0] for i in range(self.batch_size)]
        arrival_times = torch.FloatTensor(arrival_times)

        flag_need_trans = ((flag_trans_2_next_time == 0) & (~self.done_batch))
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # mask out available times that are less than the current time
        a = torch.where(a <= self.time[:, None], torch.tensor(float('inf'), dtype=a.dtype), a)
        # min of arrival times and machine available time
        zero_jobs = torch.zeros_like(a)
        b = torch.where(((a < arrival_times[:, None]) & ((self.num_jobs_system[:, None] > zero_jobs) | ~torch.tensor(self.continuous))), a, arrival_times[:, None])
        # Return the minimum
        c_2 = torch.min(b, dim=1)[0]
        # transition batches that require it
        e = torch.where(flag_need_trans, c_2, self.time)
        # check jobs complete at this time step
        d = torch.where((e[:, None] >= a) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True,
                        False)
        if torch.isinf(e).any():
            print(f'Transitioning to time=inf')
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()  # index of completed jobs
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()  # extract completed jobs
        batch_idxes = jobs_index[0]

        aa[d, 3] = -1
        self.machines_batch = aa.transpose(1, 2)  # set index of job for machine as -1

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False

        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def initialise_arrival_times(self, batch_idx):
        tim = 0
        #if (not self.continuous) and (L_jobs is None):
        # self.inital_jobs = self.rngs_init_jobs[batch_idx].integers(1, self.inital_jobs_max)
        # print(f'init_jobs: {self.inital_jobs}')
        for _ in range(self.inital_jobs):
            self.arrival_times[batch_idx].append(int(tim))
        # else:
        #     for _ in range(L_jobs):
        #         self.inital_jobs = L_jobs
        #         self.arrival_times[batch_idx].append(int(tim))

        if not self.continuous:
            for _ in range(3):
                self.arrival_times[batch_idx].append(float('inf'))
        else:
            for _ in range(3):
                inter_arrival_time = self.rngs_job_arr[batch_idx].exponential(1 / self.arrival_rates[batch_idx])
                tim += inter_arrival_time
                self.arrival_times[batch_idx].append(int(tim))
                while self.arrival_times[batch_idx][-1] == self.arrival_times[batch_idx][-2]:
                    tim = self.arrival_times[batch_idx][-1] + self.rngs_job_arr[batch_idx].exponential(
                        1 / self.arrival_rates[batch_idx])
                    self.arrival_times[batch_idx].append(int(tim))

    def next_arrival_times(self, batch_idx):
        if not self.continuous:
            self.arrival_times[batch_idx][0] = float('inf')
        else:
            tim = self.arrival_times[batch_idx][-1] + self.rngs_job_arr[batch_idx].exponential(
                1 / self.arrival_rates[batch_idx])
            self.arrival_times[batch_idx].append(int(tim))
            while self.arrival_times[batch_idx][-1] == self.arrival_times[batch_idx][-2]:
                tim = self.arrival_times[batch_idx][-1] + self.rngs_job_arr[batch_idx].exponential(
                    1 / self.arrival_rates[batch_idx])
                self.arrival_times[batch_idx].append(int(tim))

    def time_shift_feat_opes_batch(self):
        for i in range(self.batch_size):
            j = 0
            for op in self.ope_step_batch[i]:
                if not self.mask_job_finish_batch[i, j]:
                    st_time = self.feat_opes_batch[i, 5, op]
                    end_op = self.end_ope_biases_batch[i, j]
                    mask = st_time < self.time[i]
                    if mask:
                        diff = self.time[i] - st_time
                        sliced = self.feat_opes_batch[i, 5, op:end_op + 1]
                        update = sliced + diff
                        self.feat_opes_batch[i, 5, op:end_op + 1] = torch.where(mask, update, sliced)
                j += 1

    def add_job(self):

        # detect which batches have additions of jobs
        # add jobs to "added jobs" list
        # generate next arrival times
        added_jobs = [[] for _ in range(self.batch_size)]
        tensors = [[] for _ in range(self.num_data)]
        og_ops_added = copy.deepcopy(self.num_opes_system).int()
        og_jobs_added = copy.deepcopy(self.num_jobs_system).int()
        og_max_opes = copy.deepcopy(self.num_opes)
        og_max_jobs = copy.deepcopy(self.max_jobs)
        arrival_times = [[] for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            added_jobs[i].append(self.library[i][0])
            while (self.arrival_times[i][0] <= self.time[i]):  # if job to be added
                job_idx = self.rngs_job_idx[i].integers(1, self.num_jobs)
                added_jobs[i].append(self.library[i][job_idx])  # append jobs to be added in one tim step
                arrival_times[i].append(self.arrival_times[i].pop(0))
                self.tot_jobs_added[i] += 1
                # if i == 54:
                #     print(f'JOB ADDED {job_idx} bi {i}')
            if len(added_jobs[i]) > 1:
                leny = len(added_jobs[i])
                added_num_jobs, num_mas, added_num_opes = nums_detec(added_jobs[i])
                self.num_opes_system[i] += added_num_opes
                self.num_jobs_system[i] += added_num_jobs
                self.tot_ops_added[i] += added_num_opes

                self.next_arrival_times(i)

        if not any(len(added_job) > 1 for added_job in added_jobs):  # if no jobs were added return
            return False
        # Records the maximum number of operations in the parallel instances
        self.num_opes = int(max(max(self.num_opes_system), self.num_opes))
        self.max_jobs = int(max(max(self.num_jobs_system), self.max_jobs))

        pad_size_jobs = self.max_jobs - og_max_jobs
        padding = (0, pad_size_jobs)
        self.ope_step_batch = F.pad(input=self.ope_step_batch, pad=padding, value=-1)
        pad_size = self.max_jobs - og_max_jobs
        padding = (0, pad_size)
        self.mask_job_procing_batch = F.pad(input=self.mask_job_procing_batch, pad=padding, value=True)
        self.mask_job_finish_batch = F.pad(input=self.mask_job_finish_batch, pad=padding, value=True)
        pad_size_opes = self.num_opes - int(og_max_opes)
        padding = (0, pad_size_opes)
        self.feat_opes_batch = F.pad(input=self.feat_opes_batch, pad=padding, value=1)
        for i in range(self.batch_size):
            if self.DDT_a:
                random_number = self.rngs_ddt_select[i].random()
                if random_number < self.tight_percent:
                    load_data = load_fjs_new(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs, og_jobs_added[i],
                                             self.nums_ope_batch[i], self.nums_ope_batch_dynamic[i], og_ops_added[i],
                                             self.deadlines_batch[i], self.opes_appertain_batch[i],
                                             self.num_ope_biases_batch[i], self.proc_times_batch[i],
                                             self.ope_pre_adj_batch[i],
                                             self.cal_cumul_adj_batch[i], arrival_times[i], self.rngs_ddt[i],
                                             DDT_high=self.DDT_low + 0.5, DDT_low=self.DDT_low)
                else:
                    load_data = load_fjs_new(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs, og_jobs_added[i],
                                             self.nums_ope_batch[i], self.nums_ope_batch_dynamic[i], og_ops_added[i],
                                             self.deadlines_batch[i], self.opes_appertain_batch[i],
                                             self.num_ope_biases_batch[i], self.proc_times_batch[i],
                                             self.ope_pre_adj_batch[i],
                                             self.cal_cumul_adj_batch[i], arrival_times[i], self.rngs_ddt[i],
                                             DDT_high=self.DDT_high, DDT_low=self.DDT_low + 0.5)
            else:
                load_data = load_fjs_new(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs, og_jobs_added[i],
                                         self.nums_ope_batch[i], self.nums_ope_batch_dynamic[i], og_ops_added[i],
                                         self.deadlines_batch[i], self.opes_appertain_batch[i],
                                         self.num_ope_biases_batch[i], self.proc_times_batch[i],
                                         self.ope_pre_adj_batch[i],
                                         self.cal_cumul_adj_batch[i], arrival_times[i], self.rngs_ddt[i],
                                         DDT_high=self.DDT_high, DDT_low=self.DDT_low)

            hmm = load_data[5]
            self.ope_step_batch[i:i + 1, int(og_jobs_added[i]):int(self.max_jobs)] = hmm[int(og_jobs_added[i]):int(
                self.max_jobs)]
            self.mask_job_procing_batch[i, int(og_jobs_added[i]):int(self.num_jobs_system[i])] = False
            self.mask_job_finish_batch[i, int(og_jobs_added[i]):int(self.num_jobs_system[i])] = False
            self.feat_opes_batch[i, 0, int(og_ops_added[i]):int(self.num_opes_system[i])] = 0
            for j in range(self.num_data):
                tensors[j].append(load_data[j])

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[8], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        self.nums_ope_batch_dynamic = torch.stack(tensors[7], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)
        # shape: (batch_size, num_jobs), the deadline for each job
        self.deadlines_batch = torch.stack(tensors[9], dim=0).long()

        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch < 0, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch >= self.num_opes, self.num_opes - 1,
                                                self.end_ope_biases_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch < 0, self.num_opes - 1, self.ope_step_batch)
        self.ope_step_batch = torch.where(self.ope_step_batch >= self.num_opes, self.num_opes - 1, self.ope_step_batch)
        self.opes_appertain_batch = torch.where(self.opes_appertain_batch < 0, 0, self.opes_appertain_batch)

        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))

        feat_opes_batch[:, 0, :] = self.feat_opes_batch[:, 0, :]
        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch_dynamic, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        for i in range(self.batch_size):
            feat_opes_batch[:, 5, :og_ops_added[i]] = self.feat_opes_batch[:, 5, :og_ops_added[i]]
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 6, :] = convert_feat_job_2_ope(self.deadlines_batch, self.opes_appertain_batch)
        self.feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch

        for i in self.batch_idxes:
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
            # self.feat_opes_batch[i, 0, int(self.num_opes_system[i]):] = 1
            self.mask_job_procing_batch[i, int(self.num_jobs_system[i]):] = True
            self.mask_job_finish_batch[i, int(self.num_jobs_system[i]):] = True

    def reset(self, reset_rng=True):
        '''
        Reset the environment to its initial state
        '''
        if reset_rng:
            for i in range(self.batch_size):
                self.rngs_job_idx[i].__setstate__(self.old_rngs_job_idx_state[i])
                self.rngs_job_arr[i].__setstate__(self.old_rngs_job_arr_state[i])
                self.rngs_ddt[i].__setstate__(self.old_rngs_ddt_state[i])
                self.rngs_ddt_select[i].__setstate__(self.old_rngs_ddt_select_state[i])
                self.rngs_init_jobs[i].__setstate__(self.old_rngs_init_jobs_state[i])

        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.min_proc = copy.deepcopy(self.old_min_proc)
        self.max_proc = copy.deepcopy(self.old_max_proc)
        self.max_flex = copy.deepcopy(self.old_max_flex)
        self.min_opes = copy.deepcopy(self.old_min_opes)
        self.max_opes = copy.deepcopy(self.old_max_opes)

        self.ope_pre_adj_batch = copy.deepcopy(self.old_ope_pre_adj_batch)
        self.ope_sub_adj_batch = copy.deepcopy(self.old_ope_sub_adj_batch)
        self.opes_appertain_batch = copy.deepcopy(self.old_opes_appertain_batch)

        self.end_ope_biases_batch = copy.deepcopy(self.old_end_ope_biases_batch)
        self.nums_ope_batch = copy.deepcopy(self.old_nums_ope_batch)
        self.nums_ope_batch_dynamic = copy.deepcopy(self.old_nums_ope_batch_dynamic)
        self.deadlines_batch = copy.deepcopy(self.old_deadlines_batch)
        self.num_ope_biases_batch = copy.deepcopy(self.old_num_ope_biases_batch)
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        self.num_opes_system = copy.deepcopy(self.old_num_opes_system)
        self.num_jobs_system = copy.deepcopy(self.old_num_jobs_system)

        self.num_opes = int(max(self.num_opes_system))
        self.max_jobs = int(max(self.num_jobs_system))

        self.tot_jobs_added = copy.deepcopy(self.old_tot_jobs_added)
        self.tot_ops_added = copy.deepcopy(self.old_tot_ops_added)

        self.tot_scheduled_ops = torch.zeros(self.batch_size).int()
        self.tot_scheduled_jobs = torch.zeros(self.batch_size).int()
        self.tot_scheduled_jobs_late = torch.zeros(self.batch_size).int()

        self.arrival_times = copy.deepcopy(self.old_arrival_times)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.periods_counted = 1

        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                 fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)

        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        mask = self.feat_opes_batch[:, 6, :] != 0
        tardy_ratio = self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]
        tardy_ratio[~mask] = 0
        tardy_ratio = tardy_ratio.gather(1, self.end_ope_biases_batch)

        completion_times = self.feat_opes_batch[:, 4, :]
        completion_times[~mask] = 0
        completion_times = completion_times.gather(1, self.end_ope_biases_batch)

        tardy_ratio_clip = torch.max(torch.zeros_like(tardy_ratio), tardy_ratio)
        tardy_ratio_clip_batch = torch.sum(tardy_ratio_clip, dim=1)
        completion_batch = torch.sum(completion_times, dim=1)

        self.tardiness_batch = tardy_ratio_clip_batch
        self.completion_batch = completion_batch

        self.true_tardiness_batch_cumul = torch.zeros(self.batch_size)
        self.true_completion_batch_cumul = torch.zeros(self.batch_size)
        self.true_tardiness_batch = torch.zeros(self.batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        for i in self.batch_idxes:
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
            # self.feat_opes_batch[i, 0, int(self.num_opes_system[i]):] = 1
            self.mask_job_procing_batch[i, int(self.num_jobs_system[i]):] = True
            self.mask_job_finish_batch[i, int(self.num_jobs_system[i]):] = True

        self.prev_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)
        return self.state

    def training_reset(self, reset_rng=True):
        '''
        Reset the environment to its initial state
        '''
        if reset_rng:
            for i in range(self.batch_size):
                self.rngs_job_idx[i].__setstate__(self.old_rngs_job_idx_state[i])
                self.rngs_job_arr[i].__setstate__(self.old_rngs_job_arr_state[i])
                self.rngs_ddt[i].__setstate__(self.old_rngs_ddt_state[i])
                self.rngs_ddt_select[i].__setstate__(self.old_rngs_ddt_select_state[i])
                self.rngs_init_jobs[i].__setstate__(self.old_rngs_init_jobs_state[i])

            self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
            self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
            self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
            self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
            self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
            self.state = copy.deepcopy(self.old_state)

            self.min_proc = copy.deepcopy(self.old_min_proc)
            self.max_proc = copy.deepcopy(self.old_max_proc)
            self.max_flex = copy.deepcopy(self.old_max_flex)
            self.min_opes = copy.deepcopy(self.old_min_opes)
            self.max_opes = copy.deepcopy(self.old_max_opes)

            self.ope_pre_adj_batch = copy.deepcopy(self.old_ope_pre_adj_batch)
            self.ope_sub_adj_batch = copy.deepcopy(self.old_ope_sub_adj_batch)
            self.opes_appertain_batch = copy.deepcopy(self.old_opes_appertain_batch)

            self.end_ope_biases_batch = copy.deepcopy(self.old_end_ope_biases_batch)
            self.nums_ope_batch = copy.deepcopy(self.old_nums_ope_batch)
            self.nums_ope_batch_dynamic = copy.deepcopy(self.old_nums_ope_batch_dynamic)
            self.deadlines_batch = copy.deepcopy(self.old_deadlines_batch)
            self.num_ope_biases_batch = copy.deepcopy(self.old_num_ope_biases_batch)
            self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

            self.num_opes_system = copy.deepcopy(self.old_num_opes_system)
            self.num_jobs_system = copy.deepcopy(self.old_num_jobs_system)

            self.num_opes = int(max(self.num_opes_system))
            self.max_jobs = int(max(self.num_jobs_system))

            self.tot_jobs_added = copy.deepcopy(self.old_tot_jobs_added)
            self.tot_ops_added = copy.deepcopy(self.old_tot_ops_added)

            self.tot_scheduled_ops = torch.zeros(self.batch_size).int()
            self.tot_scheduled_jobs = torch.zeros(self.batch_size).int()
            self.tot_scheduled_jobs_late = torch.zeros(self.batch_size).int()

            self.arrival_times = copy.deepcopy(self.old_arrival_times)

            self.batch_idxes = torch.arange(self.batch_size)
            self.time = torch.zeros(self.batch_size)
            self.periods_counted = 1

            self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
            self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                     fill_value=False)
            self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                    fill_value=False)
            self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                    fill_value=False)

            self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
            self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

            self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
            mask = self.feat_opes_batch[:, 6, :] != 0
            tardy_ratio = self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]
            tardy_ratio[~mask] = 0
            tardy_ratio = tardy_ratio.gather(1, self.end_ope_biases_batch)

            completion_times = self.feat_opes_batch[:, 4, :]
            completion_times[~mask] = 0
            completion_times = completion_times.gather(1, self.end_ope_biases_batch)

            tardy_ratio_clip = torch.max(torch.zeros_like(tardy_ratio), tardy_ratio)
            tardy_ratio_clip_batch = torch.sum(tardy_ratio_clip, dim=1)
            completion_batch = torch.sum(completion_times, dim=1)

            self.tardiness_batch = tardy_ratio_clip_batch
            self.completion_batch = completion_batch

            self.true_tardiness_batch_cumul = torch.zeros(self.batch_size)
            self.true_completion_batch_cumul = torch.zeros(self.batch_size)
            self.true_tardiness_batch = torch.zeros(self.batch_size)
            self.done_batch = self.mask_job_finish_batch.all(dim=1)

            for i in self.batch_idxes:
                self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
                # self.feat_opes_batch[i, 0, int(self.num_opes_system[i]):] = 1
                self.mask_job_procing_batch[i, int(self.num_jobs_system[i]):] = True
                self.mask_job_finish_batch[i, int(self.num_jobs_system[i]):] = True
            self.prev_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)
        else:
            self.tot_scheduled_jobs = torch.zeros(self.batch_size).int()  # Count scheduled jobs
            self.tot_scheduled_jobs_late = torch.zeros(self.batch_size).int()
            self.tot_scheduled_ops = torch.zeros(self.batch_size).int()  # Count scheduled operations

            self.num_opes_system = torch.zeros(self.batch_size)  # num opes in each batch
            self.num_jobs_system = torch.zeros(self.batch_size)  # num of jobs in each batch
            self.tot_jobs_added = torch.zeros(self.batch_size)  # keeps track of tot no. jobs added for each case
            self.arrival_times = [[] for _ in range(self.batch_size)]  # arrival times for upcoming jobs
            self.tot_ops_added = torch.zeros(self.batch_size)  # keeps track of tot no. ops added for each case
            tensors = [[] for _ in range(self.num_data)]
            added_jobs = [[] for _ in range(self.batch_size)]

            for i in range(self.batch_size):
                self.initialise_arrival_times(i)
                added_jobs[i].append(self.library[i][0])
                while (self.arrival_times[i][0] <= 0):
                    job_idx = self.rngs_job_idx[i].integers(1, self.num_jobs)
                    added_jobs[i].append(self.library[i][job_idx])
                    self.arrival_times[i].pop(0)
                    self.tot_jobs_added[i] += 1
                num_jobs, num_mas, num_opes = nums_detec(added_jobs[i])
                self.num_opes_system[i] += num_opes
                self.num_jobs_system[i] += num_jobs

                self.tot_ops_added[i] += num_opes
                self.next_arrival_times(i)

            # Records the maximum number of operations in the parallel instances
            self.num_opes = int(max(self.num_opes_system))
            self.max_jobs = int(max(self.num_jobs_system))

            # load feats
            for i in range(self.batch_size):
                if self.DDT_a:
                    random_number = self.rngs_ddt_select[i].random()
                    if random_number < self.tight_percent:
                        load_data = load_fjs(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs,
                                             self.rngs_ddt[i],
                                             DDT_high=self.DDT_low + 0.5, DDT_low=self.DDT_low)
                    else:
                        load_data = load_fjs(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs,
                                             self.rngs_ddt[i],
                                             DDT_high=self.DDT_high, DDT_low=self.DDT_low + 0.5)
                else:
                    load_data = load_fjs(added_jobs[i], self.num_mas, self.num_opes, self.max_jobs, self.rngs_ddt[i],
                                         DDT_high=self.DDT_high, DDT_low=self.DDT_low)

                for j in range(self.num_data - 1):
                    tensors[j].append(load_data[j])

            # feats
            # shape: (batch_size, num_opes, num_mas)
            self.proc_times_batch = torch.stack(tensors[0], dim=0)
            # shape: (batch_size, num_opes, num_mas)
            self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
            # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
            self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()
            # shape: (batch_size, num_opes, num_opes)
            self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
            # shape: (batch_size, num_opes, num_opes)
            self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
            # shape: (batch_size, num_opes), represents the mapping between operations and jobs
            self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
            # shape: (batch_size, num_jobs), the id of the first operation of each job
            self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
            # shape: (batch_size, num_jobs), the number of operations for each job
            self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
            self.nums_ope_batch_dynamic = torch.stack(tensors[6], dim=0).long()
            # shape: (batch_size, num_jobs), the id of the last operation of each job
            self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
            # shape: (batch_size), the number of operations for each instance
            self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)
            # shape: (batch_size, num_jobs), the deadline for each job
            self.deadlines_batch = torch.stack(tensors[8], dim=0).long()

            # dynamic variable
            self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
            self.time = torch.zeros(self.batch_size)  # Current time of the environment

            # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
            self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

            self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch < 0, self.num_opes - 1,
                                                    self.end_ope_biases_batch)
            self.end_ope_biases_batch = torch.where(self.end_ope_biases_batch >= self.num_opes, self.num_opes - 1,
                                                    self.end_ope_biases_batch)
            self.ope_step_batch = torch.where(self.ope_step_batch < 0, self.num_opes - 1, self.ope_step_batch)
            self.ope_step_batch = torch.where(self.ope_step_batch >= self.num_opes, self.num_opes - 1,
                                              self.ope_step_batch)
            self.opes_appertain_batch = torch.where(self.opes_appertain_batch < 0, 0, self.opes_appertain_batch)
            '''
            features, dynamic
                ope:
                    Status
                    Number of neighboring machines
                    Processing time
                    Number of unscheduled operations in the job
                    Job completion time
                    Start time
                    Job Deadline
                ma:
                    Number of neighboring operations
                    Available time
                    Utilization
            '''
            # Generate raw feature vectors
            feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
            feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas))

            feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
            feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
            feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
            feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                                 self.cal_cumul_adj_batch).squeeze()
            end_time_batch = (feat_opes_batch[:, 5, :] +
                              feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
            feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
            feat_opes_batch[:, 6, :] = convert_feat_job_2_ope(self.deadlines_batch, self.opes_appertain_batch)
            feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
            self.feat_opes_batch = feat_opes_batch
            self.feat_mas_batch = feat_mas_batch

            # Masks of current status, dynamic
            # shape: (batch_size, num_jobs), True for jobs in process
            self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                     fill_value=False)
            # shape: (batch_size, num_jobs), True for completed jobs
            self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.max_jobs), dtype=torch.bool,
                                                    fill_value=False)
            # shape: (batch_size, num_mas), True for machines in process
            self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                    fill_value=False)

            '''
            Partial Schedule (state) of machines, dynamic
                idle
                available_time
                utilization_time
                id_ope
            '''
            self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
            self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))
            self.machines_batch[:, :, 3] = self.machines_batch[:, :, 3] - 1
            self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]

            mask = self.feat_opes_batch[:, 6, :] != 0
            tardy_ratio = self.feat_opes_batch[:, 4, :] - self.feat_opes_batch[:, 6, :]
            tardy_ratio[~mask] = 0
            tardy_ratio = tardy_ratio.gather(1, self.end_ope_biases_batch)

            completion_times = self.feat_opes_batch[:, 4, :]
            completion_times[~mask] = 0
            completion_times = completion_times.gather(1, self.end_ope_biases_batch)

            tardy_ratio_clip = torch.max(torch.zeros_like(tardy_ratio), tardy_ratio)
            tardy_ratio_clip_batch = torch.sum(tardy_ratio_clip, dim=1)
            completion_batch = torch.sum(completion_times, dim=1)
            self.tardiness_batch = tardy_ratio_clip_batch
            self.completion_batch = completion_batch
            self.true_tardiness_batch_cumul = torch.zeros(self.batch_size)
            self.true_completion_batch_cumul = torch.zeros(self.batch_size)
            self.true_tardiness_batch = torch.zeros(self.batch_size)

            self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
            self.old_reward = torch.zeros(self.batch_size)

            for i in self.batch_idxes:
                self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
                self.mask_job_procing_batch[i, int(self.num_jobs_system[i]):] = True
                self.mask_job_finish_batch[i, int(self.num_jobs_system[i]):] = True
            self.prev_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)

            self.state = EnvState(batch_idxes=self.batch_idxes,
                                  feat_opes_batch=self.feat_opes_batch,
                                  feat_mas_batch=self.feat_mas_batch,
                                  proc_times_batch=self.proc_times_batch,
                                  ope_ma_adj_batch=self.ope_ma_adj_batch,
                                  mask_job_procing_batch=self.mask_job_procing_batch,
                                  mask_job_finish_batch=self.mask_job_finish_batch,
                                  mask_ma_procing_batch=self.mask_ma_procing_batch,
                                  ope_step_batch=self.ope_step_batch,
                                  time_batch=self.time,
                                  deadlines_batch=self.deadlines_batch,
                                  cal_cumul_adj_batch=self.cal_cumul_adj_batch,
                                  ope_pre_adj_batch=self.ope_pre_adj_batch,
                                  ope_sub_adj_batch=self.ope_sub_adj_batch,
                                  opes_appertain_batch=self.opes_appertain_batch,
                                  end_ope_biases_batch=self.end_ope_biases_batch,
                                  nums_opes_batch=self.nums_opes,
                                  nums_ope_batch=self.nums_ope_batch,
                                  nums_ope_batch_dynamic=self.nums_ope_batch_dynamic,
                                  num_ope_biases_batch=self.num_ope_biases_batch,
                                  num_jobs_system = self.num_jobs_system,
                                  min_proc=self.min_proc,
                                  max_proc=self.max_proc,
                                  max_flex=self.max_flex,
                                  min_opes=self.min_opes,
                                  max_opes=self.max_opes
                                  )

            # Save initial data for reset - only includes dynamic features
            self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
            self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
            self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
            self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
            self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
            self.old_state = copy.deepcopy(self.state)

            self.old_num_opes_system = copy.deepcopy(self.num_opes_system)
            self.old_num_jobs_system = copy.deepcopy(self.num_jobs_system)

            self.old_tot_jobs_added = copy.deepcopy(self.tot_jobs_added)
            self.old_tot_ops_added = copy.deepcopy(self.tot_ops_added)
            self.old_arrival_times = copy.deepcopy(self.arrival_times)

            self.old_ope_pre_adj_batch = copy.deepcopy(self.ope_pre_adj_batch)
            self.old_ope_sub_adj_batch = copy.deepcopy(self.ope_sub_adj_batch)
            self.old_opes_appertain_batch = copy.deepcopy(self.opes_appertain_batch)

            self.old_ope_step_batch = copy.deepcopy(self.ope_step_batch)
            self.old_end_ope_biases_batch = copy.deepcopy(self.end_ope_biases_batch)
            self.old_nums_ope_batch = copy.deepcopy(self.nums_ope_batch)
            self.old_nums_ope_batch_dynamic = copy.deepcopy(self.nums_ope_batch_dynamic)
            self.old_deadlines_batch = copy.deepcopy(self.deadlines_batch)
            self.old_num_ope_biases_batch = copy.deepcopy(self.num_ope_biases_batch)
            self.old_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)
            self.old_mask_job_finish_batch = copy.deepcopy(self.mask_job_finish_batch)

            # get rng states
            self.old_rngs_job_idx_state = [rng_job_idx_state.__getstate__() for rng_job_idx_state in self.rngs_job_idx]
            self.old_rngs_job_arr_state = [rng_job_arr_state.__getstate__() for rng_job_arr_state in self.rngs_job_arr]
            self.old_rngs_ddt_state = [rng_ddt_state.__getstate__() for rng_ddt_state in self.rngs_ddt]
            self.old_rngs_ddt_select_state = [rng_ddt_state.__getstate__() for rng_ddt_state in self.rngs_ddt_select]
            self.old_rngs_init_jobs_state = [rng_init_jobs_state.__getstate__() for rng_init_jobs_state in self.rngs_init_jobs]

        return self.state

    def half_reset(self):
        '''
        Reset the environment to its half state
        '''
        self.periods_counted += 1  # increment number of periods counted
        self.time_limit = self.time_period * self.periods_counted  # adjust time_limit

        self.true_tardiness_batch = torch.zeros(self.batch_size)  # reset tardiness measurement
        self.tot_scheduled_jobs = torch.zeros(self.batch_size).int()  # reset total number of scheduled jobs
        self.tot_scheduled_jobs_late = torch.zeros(self.batch_size).int()

        limit_reached = torch.where(self.time >= self.time_limit, True, False)
        continuous_tensor = torch.tensor(self.continuous, dtype=torch.bool)
        self.done_batch = (self.mask_job_finish_batch.all(dim=1) & ~continuous_tensor) | limit_reached

        count = 0
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time == 0) & (~self.done_batch))).all()):
            self.next_time_2(flag_trans_2_next_time)
            if self.continuous:
                self.add_job()
            # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
            flag_trans_2_next_time = self.if_no_eligible()
            count += 1
            limit_reached = torch.where(self.time >= self.time_limit, True, False)
            continuous_tensor = torch.tensor(self.continuous, dtype=torch.bool)
            self.done_batch = (self.mask_job_finish_batch.all(dim=1) & ~continuous_tensor) | limit_reached
            self.done = self.done_batch.all()
            if count > 10:
                print('infinite loop')
                flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done_batch)
                print(f'flag {flag_need_trans}')
                print(f'tot jobs {self.tot_jobs_added}')
                #  raise Exception

        self.time_shift_feat_opes_batch()
        end_time_batch = (self.feat_opes_batch[:, 5, :] +
                          self.feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch[
                                                                   :, :])
        self.feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[
                                                                               :, :])
        for i in range(self.batch_size):
            self.feat_opes_batch[i, :, int(self.num_opes_system[i]):] = 0
        self.prev_mask_job_procing_batch = copy.deepcopy(self.mask_job_procing_batch)

        self.batch_idxes = torch.arange(self.batch_size)
        # Update state of the environment
        self.state.update(self.batch_idxes,
                          self.feat_opes_batch,
                          self.feat_mas_batch,
                          self.proc_times_batch,
                          self.ope_ma_adj_batch,
                          self.mask_job_procing_batch,
                          self.mask_job_finish_batch,
                          self.mask_ma_procing_batch,
                          self.ope_step_batch,
                          self.time,
                          self.deadlines_batch,
                          self.cal_cumul_adj_batch,
                          self.ope_pre_adj_batch,
                          self.ope_sub_adj_batch,
                          self.opes_appertain_batch,
                          self.end_ope_biases_batch,
                          self.nums_opes,
                          self.nums_ope_batch,
                          self.nums_ope_batch_dynamic,
                          self.num_ope_biases_batch,
                          self.num_jobs_system,
                          self.min_proc,
                          self.max_proc,
                          self.max_flex,
                          self.min_opes,
                          self.max_opes)

    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def render(self, mode='human'):
        pass

    def close(self):
        pass
