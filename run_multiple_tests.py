import json
import os

def update_test_paras(config_file, updates):
    # Load the existing config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Update the test_paras section with the new values
    for key, value in updates.items():
        if key in config['test_paras']:
            config['test_paras'][key] = value
        else:
            print(f"Warning: {key} not found in test_paras of the config file")

    # Save the updated configuration back to the file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def update_train_paras(config_file, updates):
    # Load the existing config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Update the test_paras section with the new values
    for key, value in updates.items():
        if key in config['train_paras']:
            config['train_paras'][key] = value
        else:
            print(f"Warning: {key} not found in train_paras of the config file")

    # Save the updated configuration back to the file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def update_env_paras(config_file, updates):
    # Load the existing config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Update the test_paras section with the new values
    for key, value in updates.items():
        if key in config['env_paras']:
            config['env_paras'][key] = value
        else:
            print(f"Warning: {key} not found in train_paras of the config file")

    # Save the updated configuration back to the file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)


def run_test_script():
    os.system('python test.py')

if __name__ == "__main__":
    config_file = 'config.json'

    # Example of different scenarios
    scenarios = [
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK1"],
            "ma_util": [0.65, 0.7, 0.75, 0.80, 0.85],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results_the_odd_ones/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK4"],
            "ma_util": [0.55, 0.60, 0.65, 0.7, 0.75],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results_the_odd_ones/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK8"],
            "ma_util": [0.35, 0.40, 0.45, 0.5, 0.55],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results_the_odd_ones/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK10"],
            "ma_util": [0.65, 0.7, 0.75, 0.80, 0.85],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results_the_odd_ones/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK12"],
            "ma_util": [0.45, 0.50, 0.55, 0.6, 0.65],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results_the_odd_ones/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK14"],
            "ma_util": [0.3, 0.35, 0.4, 0.45, 0.5],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results_the_odd_ones/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        }
    ]

    scenarios2 = [
        # {
        #     "rules": [],
        #     "mas_rules": [],
        #     "test_seed": 80,
        #     "model": [0],
        #     "model_location": "./model/",
        #     "data_path": ["MK2", "MK3", "MK5", "MK6", "MK7", "MK9", "MK11", "MK13", "MK15"],
        #     "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
        #     "DDT_low": 0.5,
        #     "DDT_high": 3.0,
        #     "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
        #     "save_path": "./Brandimarte_model_Results_2/",
        #     "flag_filter_machines": True,
        #     "flag_num_to_filter": 5
        # },
        # {
        #     "rules": [],
        #     "mas_rules": [],
        #     "test_seed": 80,
        #     "model": [0],
        #     "model_location": "./model/",
        #     "data_path": ["F2", "F3", "F4", "F5", "F1_5"],
        #     "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
        #     "DDT_low": 0.5,
        #     "DDT_high": 3.0,
        #     "data_path_loc": "/data_test/synthetic/Machine_Selection_Coordination/",
        #     "save_path": "./five_machines_model_results/",
        #     "flag_filter_machines": True,
        #     "flag_num_to_filter": 5
        # },
        # {
        #     "rules": [],
        #     "mas_rules": [],
        #     "test_seed": 80,
        #     "model": [0],
        #     "model_location": "./model/",
        #     "data_path": ["M10P10", "M10P20", "M10P30", "M10P40", "M10P50",
        #                   "M20P10", "M20P20", "M20P30", "M20P40", "M20P50",
        #                   "M30P10", "M30P20", "M30P30", "M30P40", "M30P50"],
        #     "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
        #     "DDT_low": 0.5,
        #     "DDT_high": 3.0,
        #     "data_path_loc": "/data_test/synthetic/Factory_Setup/",
        #     "save_path": "./ave_results_filtered_5/",
        #     "flag_filter_machines": True,
        #     "flag_num_to_filter": 5
        # }
    ]

    hurink_scenarios = [
        {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["HurinkVdata19",
                            "HurinkVdata20",
                            "HurinkVdata21",
                            "HurinkVdata22",
                            "HurinkVdata23",
                            "HurinkVdata24",
                            "HurinkVdata25",
                            "HurinkVdata26",
                            "HurinkVdata27",
                            "HurinkVdata28",
                            "HurinkVdata29",
                            "HurinkVdata30",
                            "HurinkVdata31",
                            "HurinkVdata32",
                            "HurinkVdata33",
                            "HurinkVdata34",
                            "HurinkVdata35",
                            "HurinkVdata36",
                            "HurinkVdata37",
                            "HurinkVdata38",
                            "HurinkVdata39",
                            "HurinkVdata40",
                            "HurinkVdata41",
                            "HurinkVdata42",
                            "HurinkVdata43"],
            "ma_util": [0.70, 0.75, 0.80, 0.85, 0.90],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/2d_Hurink_vdata/",
            "save_path": "./HurinkV_eet/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
    ]

    hurink2_scenarios = [
        {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["HurinkVdata19"],
            "ma_util": [0.70, 0.75, 0.80, 0.85, 0.90],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/2d_Hurink_vdata/",
            "save_path": "./HurinkV_model_testing2/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
    ]


    discounted_testing_scenarios = [
        {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model_095/",
            "data_path": ["M10P10", "M10P20", "M10P30", "M10P40", "M10P50",
                          "M20P10", "M20P20", "M20P30", "M20P40", "M20P50",
                          "M30P10", "M30P20", "M30P30", "M30P40", "M30P50"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/synthetic/Factory_Setup/",
            "save_path": "./095_results_filtered_5/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model_090/",
            "data_path": ["M10P10", "M10P20", "M10P30", "M10P40", "M10P50",
                          "M20P10", "M20P20", "M20P30", "M20P40", "M20P50",
                          "M30P10", "M30P20", "M30P30", "M30P40", "M30P50"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/synthetic/Factory_Setup/",
            "save_path": "./090_results_filtered_5/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model_085/",
            "data_path": ["M10P10", "M10P20", "M10P30", "M10P40", "M10P50",
                          "M20P10", "M20P20", "M20P30", "M20P40", "M20P50",
                          "M30P10", "M30P20", "M30P30", "M30P40", "M30P50"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/synthetic/Factory_Setup/",
            "save_path": "./085_results_filtered_5/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        }
    ]

    discounted_training_scenarios = [
        {
            "discount_factor": 0.95,
            "discounting": True,
        },
        {
            "discount_factor": 0.90,
            "discounting": True,
        },
        {
            "discount_factor": 0.85,
            "discounting": True,
        }
    ]

    rule_testing_scenarios = [
        # {
        #     "rules": [],
        #     "mas_rules": [],
        #     "test_seed": 80,
        #     "model": [0],
        #     "model_location": "./model/",
        #     "data_path": ["F2", "F3", "F4", "F5", "F1_5"],
        #     "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
        #     "DDT_low": 0.5,
        #     "DDT_high": 3.0,
        #     "data_path_loc": "/data_test/synthetic/Machine_Selection_Coordination/",
        #     "save_path": "./Machine_Selection_Coordination/Mixed/",
        #     "flag_filter_machines": True,
        #     "flag_num_to_filter": 5
        # },
        # {
        #     "rules": [],
        #     "mas_rules": [],
        #     "test_seed": 80,
        #     "model": [0],
        #     "model_location": "./model/",
        #     "data_path": ["F2", "F3", "F4", "F5", "F1_5"],
        #     "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
        #     "DDT_low": 0.5,
        #     "DDT_high": 3.0,
        #     "data_path_loc": "/data_test/synthetic/Machine_Selection_Coordination/",
        #     "save_path": "./Machine_Selection_Coordination/EET/",
        #     "flag_filter_machines": True,
        #     "flag_num_to_filter": 5
        # },
        # {
        #     "rules": [],
        #     "mas_rules": [],
        #     "test_seed": 80,
        #     "model": [0],
        #     "model_location": "./model/",
        #     "data_path": ["F2", "F3", "F4", "F5", "F1_5"],
        #     "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
        #     "DDT_low": 0.5,
        #     "DDT_high": 3.0,
        #     "data_path_loc": "/data_test/synthetic/Machine_Selection_Coordination/",
        #     "save_path": "./Machine_Selection_Coordination/SPT_argsortFalse/",
        #     "flag_filter_machines": True,
        #     "flag_num_to_filter": 5
        # }
                {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["F2", "F3", "F4", "F5", "F1_5"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/synthetic/Machine_Selection_Coordination/",
            "save_path": "./Machine_Selection_Coordination/LPT/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        }
        # {
        #     "rules": [],
        #     "mas_rules": [],
        #     "test_seed": 80,
        #     "model": [0],
        #     "model_location": "./model/",
        #     "data_path": ["F2", "F3", "F4", "F5", "F1_5"],
        #     "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
        #     "DDT_low": 0.5,
        #     "DDT_high": 3.0,
        #     "data_path_loc": "/data_test/synthetic/Machine_Selection_Coordination/",
        #     "save_path": "./Machine_Selection_Coordination/LLM/",
        #     "flag_filter_machines": True,
        #     "flag_num_to_filter": 5
        # }
    ]

    rule_training_scenarios = [
        {
            "discounting": False,
        },
        {
            "discounting": False,
        },
        {
            "discounting": False,
        },
        {
            "discounting": False,
        }
    ]

    rule_env_scenarios = [
        # {
        #     "negotiate_rule": "SPT",
        # },
        {
            "negotiate_rule": "LPT",
        }
        # {
        #     "negotiate_rule": "EET",
        # },
        # {
        #     "negotiate_rule": "SPT",
        # },
        # {
        #     "negotiate_rule": "LLM",
        # }
    ]

    # for scenario in hurink_scenarios:
    #     update_test_paras(config_file, scenario)
    #     run_test_script()

    # for i, scenario in enumerate(discounted_testing_scenarios):
    #     update_test_paras(config_file, scenario)
    #     update_train_paras(config_file, discounted_training_scenarios[i])
    #     run_test_script()

    for i, scenario in enumerate(rule_testing_scenarios):
        update_test_paras(config_file, scenario)
        update_train_paras(config_file, rule_training_scenarios[i])
        update_env_paras(config_file, rule_env_scenarios[i])
        run_test_script()

    print("All scenarios processed!")
