import torch
import numpy as np
def load_fjs(lines, num_mas, num_opes, max_jobs, rng, DDT_high=None, DDT_low=None):
    '''
    Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
    nums_ope = []  # A list of the number of operations for each job
    opes_appertain = np.array([])
    num_ope_biases = []  # The id of the first operation of each job
    deadlines = []  # keeps track of deadline for each loaded job
    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += 1
        # last line
        elif line == "\n":
            break
        # other
        else:
            num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
            num_ope_biases.append(num_ope_bias)
            # Detect information of this job and return the number of operations
            num_ope, max_job_proc_time = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
            nums_ope.append(num_ope)

            DDT = rng.uniform(DDT_low, DDT_high)
            # print(f'DDT {DDT}')
            deadline = DDT * max_job_proc_time
            deadlines.append(int(deadline))

            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope)*(flag-1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes-opes_appertain.size)))
    nums_ope += [0] * (max_jobs - len(nums_ope))
    num_ope_biases += [0] * (max_jobs - len(num_ope_biases))
    deadlines += [0] * (max_jobs - len(deadlines))

    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul, torch.tensor(deadlines).int()


def load_fjs_new(lines, num_mas, num_opes, max_jobs, og_no_jobs, nums_ope, nums_ope_dyn, og_no_opes, deadlines,
                 opes_appertain, num_ope_biases, proc_times, ope_pre_adj, cal_cumul_adj, arrival_time, rng,
                 DDT_high=None, DDT_low=None):
    '''
       Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
    # print(f'og opes {og_no_opes}')
    nums_ope_dyn = nums_ope_dyn.tolist()[:int(og_no_jobs)]  # A list of the number of operations for each job
    nums_ope = nums_ope.tolist()[:int(og_no_jobs)]
    #print(f'nums ope {nums_ope}')
    deadlines = deadlines.tolist()[:int(og_no_jobs)]  # keeps track of deadline for each loaded job
    opes_appertain = opes_appertain.numpy()[:int(og_no_opes)]  # mapping between jobs and operations
    num_ope_biases = num_ope_biases.tolist()[:int(og_no_jobs)]  # The id of the first operation of each job
    init_jobs = int(og_no_jobs) + 1
    init_opes = og_no_opes
    matrix_proc_time[:init_opes, :] = proc_times[:init_opes, :]
    matrix_pre_proc[:init_opes, :init_opes] = ope_pre_adj[:init_opes, :init_opes]
    matrix_cal_cumul[:init_opes, :init_opes] = cal_cumul_adj[:init_opes, :init_opes]
    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += init_jobs
        # last line
        elif line == "\n":
            break
        # other
        else:
            num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
            num_ope_biases.append(num_ope_bias)
            # Detect information of this job and return the number of operations
            num_ope, max_job_proc_time = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc,
                                                    matrix_cal_cumul)
            nums_ope.append(num_ope)
            nums_ope_dyn.append(num_ope)

            DDT = rng.uniform(DDT_low, DDT_high)
            # print(f'DDT {DDT}')
            deadline = arrival_time[flag - init_jobs] + DDT * max_job_proc_time
            deadlines.append(int(deadline))

            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope) * (flag - 1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes - opes_appertain.size)))
    nums_ope += [0] * (max_jobs - len(nums_ope))
    nums_ope_dyn += [0] * (max_jobs - len(nums_ope_dyn))
    num_ope_biases += [-1] * (max_jobs - len(num_ope_biases))
    deadlines += [0] * (max_jobs - len(deadlines))

    #print(f'nums ope end {nums_ope}')
    # print(f'biases {num_ope_biases}')

    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), torch.tensor(nums_ope_dyn).int(), matrix_cal_cumul, torch.tensor(deadlines).int()


def nums_detec(lines):
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i]!="\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = len(lines) - 1  # each line represents a job; minus the first line which has setting details
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes

def lib_detec(lines, num_mas, num_opes, max_jobs):
    '''
    Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    nums_ope = []  # A list of the number of operations for each job
    nums_mas = []  # A list of the number of mas for each operation
    opes_appertain = np.array([])
    num_ope_biases = []  # The id of the first operation of each job
    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += 1
        # last line
        elif line == "\n":
            break
        # other
        else:
            num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
            num_ope_biases.append(num_ope_bias)
            # Detect information of this job and return the number of operations
            num_ope, num_option = lib_edge_detec(line, num_ope_bias, matrix_proc_time)
            nums_mas = nums_mas + num_option
            nums_ope.append(num_ope)

            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope)*(flag-1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes-opes_appertain.size)))
    nums_ope += [0] * (max_jobs - len(nums_ope))
    num_ope_biases += [0] * (max_jobs - len(num_ope_biases))

    min_proc = torch.min(matrix_proc_time)
    max_proc = torch.max(matrix_proc_time)

    max_flex = max(nums_mas)

    min_opes = min(nums_ope)
    max_opes = max(nums_ope)

    return min_proc, max_proc, max_flex, min_opes, max_opes


def num_ma_detec(line):
    line_split = line[0].strip().split()
    num_mas = int(line_split[1])
    return num_mas


def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    '''
    Detect information of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    proc_time_per_ope = []
    max_job_proc_time = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            if proc_time_per_ope:
                max_job_proc_time += max(proc_time_per_ope)
                proc_time_per_ope.clear()
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope-1:
                matrix_pre_proc[idx_ope+num_ope_bias][idx_ope+num_ope_bias+1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope+num_ope_bias-1] = 1
                matrix_cal_cumul[:, idx_ope+num_ope_bias] = matrix_cal_cumul[:, idx_ope+num_ope_bias-1]+vector
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
            proc_time_per_ope.append(x)

    max_job_proc_time += max(proc_time_per_ope)
    proc_time_per_ope.clear()

    return num_ope, max_job_proc_time

def lib_edge_detec(line, num_ope_bias, matrix_proc_time):
    '''
    Detect information of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = []  # Store the number of processable machines for each operation of this job
    mac = 0
    proc_time_per_ope = []
    max_job_proc_time = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            if proc_time_per_ope:
                max_job_proc_time += max(proc_time_per_ope)
                proc_time_per_ope.clear()
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option.append(x)
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
            proc_time_per_ope.append(x)

    max_job_proc_time += max(proc_time_per_ope)
    proc_time_per_ope.clear()

    return num_ope, num_option

def load_for_l(lines):
    """ returns the service rate """

    num_opes_per_job = [int(lines[i].strip().split()[0]) if lines[i] != "\n" else 0 for i in range(1, len(lines))]  # gets first el of each line
    num_opes = sum(num_opes_per_job)
    opes_per_job = num_opes/len(num_opes_per_job)
    line_split = lines[0].strip().split()
    num_mas = int(line_split[1])
    mas_per_ope = float(line_split[2])

    flag = 0
    num_ope_bias = 0
    # operations vs machine processing tim
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    opes_in_job = []

    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += 1
        # last line
        elif line == "\n":
            break
        # other
        else:
            num_ope_bias, num_opes_in_job = edge_detec_l(line, num_ope_bias, matrix_proc_time)
            if not opes_in_job:
                opes_in_job.append(num_opes_in_job)
            else:
                opes_in_job.append(num_opes_in_job + opes_in_job[-1])
            flag += 1

    machines = matrix_proc_time.count_nonzero(dim=1)
    sum_times = matrix_proc_time.sum(dim=1)
    mean_proc_times = sum_times / machines

    proc_of_job = []
    for job, opes in enumerate(opes_in_job):
        if job == 0:
            start = 0
        else:
            start = opes_in_job[job - 1]
        end = opes
        proc_of_job.append(mean_proc_times[start:end].sum())

    proc_per_job = sum(proc_of_job) / len(proc_of_job)

    effective_service_rate = 1.0/proc_per_job

    return effective_service_rate, mas_per_ope, opes_per_job


def edge_detec_l(line, num_ope_bias, matrix_proc_time):
    """
    Detect information of a job
    """
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    mac = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            flag += 1
            num_opes = x
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope + num_ope_bias][mac] = x
            flag += 1
            flag_time = 0

    num_ope_bias += idx_ope + 1
    return num_ope_bias, num_opes

