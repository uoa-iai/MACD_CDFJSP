import math
import os
import random


class CaseGenerator:
    '''
    FJSP instance generator
    '''
    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, ma_util=0.85, fully_flexible=False,
                 nums_ope=None, path='./data/', flag_same_opes=True, flag_doc=False, flag_same_proc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.flag_same_proc = flag_same_proc
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas
        self.ma_util = ma_util

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = num_mas
        self.fully_flexible = fully_flexible
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.proctime_per_ope_max = 20
        self.proctime_dev = 0.2

    def get_case(self, seed, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        '''
        random.seed(seed+idx)
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]

        random_number = random.random()

        # Determine the result based on the weights
        if random_number < 0.0:
            self.nums_option = [int(self.mas_per_ope_max*0.2)+1 for _ in range(self.num_opes)]
            print(f'flex_low {int(self.mas_per_ope_max*0.2)+1}')
        elif random_number < 1.1:
            self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in
                                range(self.num_opes)]
            print(f'flex_med')
        else:
            self.nums_option = [self.mas_per_ope_max for _ in range(self.num_opes)]
            print(f'flex_high')

        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas+1), val))
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)]
        for i in range(len(self.nums_option)):
            low_bound = max(self.proctime_per_ope_min, round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max, round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            if self.flag_same_proc:
                proc_time_ope = [self.proc_times_mean[i] for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i]+self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')
        if self.flag_doc:
            directory = self.path
            os.makedirs(directory, exist_ok=True)
            doc = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx+1), 3)), 'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_jobs

    def get_arrival_rate(self):
        mean_proc_per_ope = (self.proctime_per_ope_min + self.proctime_per_ope_max) / 2
        operations_per_job = (self.opes_per_job_min + self.opes_per_job_max) / 2
        processing_rate = self.num_mas * (1 / mean_proc_per_ope)
        arrival_rate = processing_rate * (self.ma_util / operations_per_job)

        t_avg = mean_proc_per_ope * operations_per_job
        service_rate = 1 / t_avg

        QM = QueueModel(arrival_rate, service_rate, self.num_mas, self.ma_util)
        L = QM.number_in_sys()

        return arrival_rate, L


class QueueModel:
    def __init__(self, arrival_rate, service_rate, num_mas, server_util):
        self.service_rate = service_rate
        self.arrival_rate = arrival_rate
        self.num_mas = num_mas
        self.server_util = server_util

    def number_in_sys(self):
        # p_zero_1 = ((self.num_mas * self.server_util) ** self.num_mas) / (
        #             math.factorial(self.num_mas) * (1 - self.server_util))
        # p_zero_2_1 = [((self.num_mas * self.server_util) ** n) / (math.factorial(n)) for n in
        #               range(0, self.num_mas)]
        # p_zero_2 = sum(p_zero_2_1)
        # p_zero = 1 / (p_zero_1 + p_zero_2)
        #
        # W_q_num = ((self.num_mas * self.server_util) ** self.num_mas) * p_zero
        # W_q_den = math.factorial(self.num_mas) * self.num_mas * self.service_rate * ((1 - self.server_util) ** 2)
        # W_q = W_q_num / W_q_den
        #
        # W = W_q + 1 / self.service_rate
        #
        # L = 1 * self.arrival_rate * W
        L = 10

        return int(L)
