import json
import time

import gym
import torch
from env.case_generator import CaseGenerator
import random

# Generate instances and save to files
def main():
    random.seed(int(time.time()))
    batch_size = 100
    num_jobs_s = [10]
    num_mas_s = [5]

    for num_mas in num_mas_s:
        for num_jobs in num_jobs_s:
            opes_per_job_min = max(1, int(num_mas * 0.4))
            opes_per_job_max = int(num_mas * 1.2)
            with open("../config.json", 'r') as load_f:
                load_dict = json.load(load_f)

            env_paras = load_dict["env_paras"]
            env_paras["batch_size"] = batch_size
            env_paras["num_jobs"] = num_jobs
            env_paras["num_mas"] = num_mas
            env_paras["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max,
                                 path=f'../data_dev/validation/1005_5/',
                                 flag_same_opes=False, flag_doc=True, flag_same_proc=True)
            gym.make('fjsp-v0', case=case, env_paras=env_paras)  # Instances are created when the environment is initialized

if __name__ == "__main__":
    main()
    print("Instances are created and stored in the \"./data\"")