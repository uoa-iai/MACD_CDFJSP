import json
import os

def update_test_paras(config_file, updates):
    with open(config_file, 'r') as f:
        config = json.load(f)
    for key, value in updates.items():
        if key in config['test_paras']:
            config['test_paras'][key] = value
        else:
            print(f"Warning: {key} not found in test_paras of the config file")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def update_train_paras(config_file, updates):
    with open(config_file, 'r') as f:
        config = json.load(f)
    for key, value in updates.items():
        if key in config['train_paras']:
            config['train_paras'][key] = value
        else:
            print(f"Warning: {key} not found in train_paras of the config file")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def update_env_paras(config_file, updates):
    with open(config_file, 'r') as f:
        config = json.load(f)
    for key, value in updates.items():
        if key in config['env_paras']:
            config['env_paras'][key] = value
        else:
            print(f"Warning: {key} not found in env_paras of the config file")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def run_test_script():
    os.system('python test.py')

if __name__ == "__main__":
    config_file = 'config.json'

    # -------------------------------------------------------
    # TABLES III & IV: Brandimarte MK1-MK15
    # Note: ma_util varies per instance based on machine count
    # Each instance run separately to allow different util levels
    # -------------------------------------------------------
    brandimarte_scenarios = [
        # MK1  10x6  - standard util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK1"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK2  10x6 - standard util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK2"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK3  15x8 - standard util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK3"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK4  15x8 - lower util (more machines)
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK4"],
            "ma_util": [0.55, 0.60, 0.65, 0.70, 0.75],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK5  15x4 - standard util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK5"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK6  10x15 - standard util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK6"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK7  20x5 - standard util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK7"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK8  20x10 - lower util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK8"],
            "ma_util": [0.35, 0.40, 0.45, 0.50, 0.55],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK9  20x10 - lower util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK9"],
            "ma_util": [0.35, 0.40, 0.45, 0.50, 0.55],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK10  20x15 - lower util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK10"],
            "ma_util": [0.65, 0.70, 0.75, 0.80, 0.85],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK11  30x5 - standard util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK11"],
            "ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK12  30x10 - lower util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK12"],
            "ma_util": [0.45, 0.50, 0.55, 0.60, 0.65],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK13  30x10 - lower util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK13"],
            "ma_util": [0.45, 0.50, 0.55, 0.60, 0.65],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK14  30x15 - lower util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK14"],
            "ma_util": [0.30, 0.35, 0.40, 0.45, 0.50],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        # MK15  30x15 - lower util
        {
            "rules": ["FIFO", "LOR", "LWKR", "EDD", "MST", "CR"],
            "mas_rules": ["LPT", "EET", "SPT", "LLM"],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["MK15"],
            "ma_util": [0.30, 0.35, 0.40, 0.45, 0.50],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/1_Brandimarte/",
            "save_path": "./Brandimarte_Results/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        }
    ]

    # -------------------------------------------------------
    # TABLES VI, VII, VIII: Hurink E, R, V
    # Filenames and util levels match original hurink_scenarios
    # -------------------------------------------------------
    hurink_scenarios = [
        {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["HurinkVdata19", "HurinkVdata20", "HurinkVdata21",
                          "HurinkVdata22", "HurinkVdata23", "HurinkVdata24",
                          "HurinkVdata25", "HurinkVdata26", "HurinkVdata27",
                          "HurinkVdata28", "HurinkVdata29", "HurinkVdata30",
                          "HurinkVdata31", "HurinkVdata32", "HurinkVdata33",
                          "HurinkVdata34", "HurinkVdata35", "HurinkVdata36",
                          "HurinkVdata37", "HurinkVdata38", "HurinkVdata39",
                          "HurinkVdata40", "HurinkVdata41", "HurinkVdata42",
                          "HurinkVdata43"],
            "ma_util": [0.70, 0.75, 0.80, 0.85, 0.90],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/2d_Hurink_vdata/",
            "save_path": "./HurinkV/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["HurinkRdata19", "HurinkRdata20", "HurinkRdata21",
                          "HurinkRdata22", "HurinkRdata23", "HurinkRdata24",
                          "HurinkRdata25", "HurinkRdata26", "HurinkRdata27",
                          "HurinkRdata28", "HurinkRdata29", "HurinkRdata30",
                          "HurinkRdata31", "HurinkRdata32", "HurinkRdata33",
                          "HurinkRdata34", "HurinkRdata35", "HurinkRdata36",
                          "HurinkRdata37", "HurinkRdata38", "HurinkRdata39",
                          "HurinkRdata40", "HurinkRdata41", "HurinkRdata42",
                          "HurinkRdata43"],
            "ma_util": [0.70, 0.75, 0.80, 0.85, 0.90],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/2d_Hurink_rdata/",
            "save_path": "./HurinkR/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "rules": [],
            "mas_rules": [],
            "test_seed": 80,
            "model": [0],
            "model_location": "./model/",
            "data_path": ["HurinkEdata19", "HurinkEdata20", "HurinkEdata21",
                          "HurinkEdata22", "HurinkEdata23", "HurinkEdata24",
                          "HurinkEdata25", "HurinkEdata26", "HurinkEdata27",
                          "HurinkEdata28", "HurinkEdata29", "HurinkEdata30",
                          "HurinkEdata31", "HurinkEdata32", "HurinkEdata33",
                          "HurinkEdata34", "HurinkEdata35", "HurinkEdata36",
                          "HurinkEdata37", "HurinkEdata38", "HurinkEdata39",
                          "HurinkEdata40", "HurinkEdata41", "HurinkEdata42",
                          "HurinkEdata43"],
            "ma_util": [0.70, 0.75, 0.80, 0.85, 0.90],
            "DDT_low": 0.5,
            "DDT_high": 3.0,
            "data_path_loc": "/data_test/FJSP-benchmarks-main/2d_Hurink_edata/",
            "save_path": "./HurinkE/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        }
    ]

    # -------------------------------------------------------
    # TABLE IX: Negotiation strategy empirical analysis
    # SPT, LPT, LLM, EET, Mixed on synthetic flexibility instances
    # Matches original rule_testing_scenarios filenames/paths
    # -------------------------------------------------------
    rule_testing_scenarios = [
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
            "save_path": "./Machine_Selection_Coordination/Mixed/",
            "flag_filter_machines": True,
            "flag_num_to_filter": 5
        },
        {
            "save_path": "./Machine_Selection_Coordination/EET/",
        },
        {
            "save_path": "./Machine_Selection_Coordination/SPT_argsortFalse/",
        },
        {
            "save_path": "./Machine_Selection_Coordination/LPT/",
        },
        {
            "save_path": "./Machine_Selection_Coordination/LLM/",
        }
    ]

    rule_env_scenarios = [
        {"negotiate_rule": "Mixed"},
        {"negotiate_rule": "EET"},
        {"negotiate_rule": "SPT"},
        {"negotiate_rule": "LPT"},
        {"negotiate_rule": "LLM"}
    ]

    rule_training_scenarios = [
        {"discounting": False},
        {"discounting": False},
        {"discounting": False},
        {"discounting": False},
        {"discounting": False}
    ]

    # -------------------------------------------------------
    # RUN LOOPS
    # -------------------------------------------------------

    # Tables III & IV - Brandimarte
    for scenario in brandimarte_scenarios:
        update_test_paras(config_file, scenario)
        run_test_script()

    # Tables VI, VII, VIII - Hurink E, R, V
    for scenario in hurink_scenarios:
        update_test_paras(config_file, scenario)
        run_test_script()

    # Table IX - Negotiation strategy comparison
    for i, scenario in enumerate(rule_testing_scenarios):
        update_test_paras(config_file, scenario)
        update_train_paras(config_file, rule_training_scenarios[i])
        update_env_paras(config_file, rule_env_scenarios[i])
        run_test_script()

    print("All scenarios completed!")