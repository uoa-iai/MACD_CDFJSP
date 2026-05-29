# MACD: Multi-Agent Continuous Decision-Making for the Continuous Dynamic Flexible Job Shop Scheduling Problem

Official implementation of the paper:

> **Multi-Agent Continuous Decision-Making for the Continuous Dynamic Flexible Job Shop Scheduling Problem**  
> Dazzle Johnson, Gang Chen, Yuqian Lu  
> *IEEE Transactions on Automation Science and Engineering (T-ASE)*, 2026  
> DOI: [to be added on publication]

---

## Overview

MACD is a multi-agent reinforcement learning system for the **Continuous Dynamic Flexible Job Shop Scheduling Problem (C-DFJSP)** -- a scheduling formulation where job orders arrive continuously from a dynamic product library, machines operate asynchronously, and decisions must be made in real time without knowledge of future arrivals.

MACD integrates:
- **Graph Neural Networks (GNNs)** for factory-invariant feature extraction via a heterogeneous GAT + MLP architecture
- **Proximal Policy Optimisation (PPO)** for end-to-end multi-agent policy learning
- A **hybrid SPT/LLM negotiation strategy** (`negotiate_rule: "Mixed"` in config) to resolve rare inter-agent conflicts

A single policy trained on small static instances (5 machines, 10 products, 20 jobs) generalises directly to unseen factories with up to 15 machines, 30 products, and continuous job arrivals -- **no retraining required**.

---

## Repository Structure

```
MACD/
├── env/                  # C-DFJSP environment
├── graph/                # GNN architecture (GATedge + MLPsim)
├── model/                # Trained model weights (.pt)
├── utils/                # Helper functions
├── data_test/            # Test instances (Brandimarte & Hurink, C-DFJSP format)
│   ├── Brandimarte/      # MK01–MK15
│   └── Hurink/           # E, R, V subsets (5, 10, 15 machines)
├── PPO_model.py          # Core MACD algorithm: HGNNScheduler + PPO
├── mlp.py                # MLP layers (actor, critic)
├── train.py              # Training script
├── test.py               # Testing / evaluation script
├── validate.py           # Validation (called automatically during training)
├── run_multiple_tests.py       # Batch testing across configurations
├── run_multiple_trainings.py   # Batch training runs
├── config.json           # All hyperparameters and run settings
└── requirements.txt      # Python dependencies
```

---

## Installation

Python >= 3.8 recommended. CPU-only installation is sufficient to reproduce all results.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

Key dependencies (pinned versions from requirements.txt):

| Package | Version |
|---------|---------|
| torch | 2.3.1+cpu |
| gym | 0.19.0 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| networkx | 3.3 |
| wandb | 0.17.5 (optional) |
| openpyxl | 3.1.5 |

> **Note:** The requirements.txt pins CPU-only PyTorch (`torch==2.3.1+cpu`). If you have a GPU, replace with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

> **Note on wandb:** Experiment logging via Weights & Biases is enabled by default (`use_wandB: true` in config). To disable, set `"use_wandB": false` in `config.json`, or run `wandb disabled` before training.

---

## Quick Start

### Training

Training is conducted on small static FJSP instances (5 machines, 10 products, 20 jobs per instance).

```bash
python train.py
```

Key training parameters in `config.json` under `env_paras` and `train_paras`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_jobs` | 10 | Jobs per static training instance |
| `num_mas` | 5 | Number of machines |
| `ma_util` | 0.9 | Machine utilisation level |
| `DDT_low` / `DDT_high` | 1.5 / 5.0 | Due date tightness range |
| `K_epochs` | 5 | PPO update epochs |
| `minibatch_size` | 1024 | PPO minibatch size |
| `negotiate_rule` | `"SPT"` | Negotiation rule during training |
| `max_iterations` | 10000 | Total training iterations |

The best-performing model on the validation set is saved to `./model/`.

### Testing

Evaluate the trained policy on C-DFJSP instances derived from the Brandimarte or Hurink benchmarks:

```bash
python test.py
```

Key test parameters in `config.json` under `test_paras`:

| Parameter | Description |
|-----------|-------------|
| `data_path` | List of instance folder names under `data_path_loc` |
| `data_path_loc` | Path to test instance folder |
| `ma_util` | List of utilisation levels to test (U1–U5) |
| `num_periods` | Number of evaluation periods (default: 100) |
| `flag_filter_machines` | Enable machine pre-filtering before GAT (default: `true`) |
| `flag_num_to_filter` | Max machines retained per operation (default: 5) |
| `negotiate_rule` | Negotiation rule: `"SPT"`, `"LLM"`, `"EET"`, `"LPT"` or `"Mixed"` (hybrid SPT/LLM used in paper) |

---

## Reproducing Paper Results

### Brandimarte Dataset (MK01–MK15)

Set the following in `config.json` under `test_paras`:

```json
"data_path": ["MK01", "MK02", "MK03", "MK04", "MK05",
              "MK06", "MK07", "MK08", "MK09", "MK10",
              "MK11", "MK12", "MK13", "MK14", "MK15"],
"data_path_loc": "./data_test/Brandimarte/",
"ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
"negotiate_rule": "Mixed",
"num_periods": 100
```

Then run:
```bash
python test.py
```

### Hurink Dataset (E, R, V)

```json
"data_path": ["E", "R", "V"],
"data_path_loc": "./data_test/Hurink/",
"ma_util": [0.75, 0.80, 0.85, 0.90, 0.95],
"negotiate_rule": "Mixed",
"num_periods": 100
```

### Negotiation Strategy Empirical Study

To reproduce the negotiation strategy comparison (Table IX in the paper), run tests individually with `"negotiate_rule"` set to `"SPT"`, `"LLM"`, `"EET"`, and `"Mixed"` respectively using the synthetic test instances under `data_test/synthetic/`.

### Batch testing

To run across all configurations at once:

```bash
python run_multiple_tests.py
```

Results are saved to the path specified by `save_path` in `config.json`.

---

## Test Instances

The `data_test/` folder contains the C-DFJSP formulations of the standard FJSP benchmarks used in the paper:

- **Brandimarte (MK01–MK15):** Mixed flexibility, 4–15 machines, 10–30 products
- **Hurink E/R/V:** Low / medium / high flexibility subsets, 5–15 machines (40 instances per subset)

These are derived from the original benchmark files:
- Brandimarte (1993): *Routing and Scheduling in a Flexible Job Shop by Tabu Search*, Annals of Operations Research, 41, 157–183
- Hurink et al. (1994): *Tabu Search for the Job-Shop Scheduling Problem with Multi-Purpose Machines*, Operations-Research-Spektrum, 15, 205–215

Each instance defines the product library used to continuously generate job orders during simulation. Job arrivals follow a Poisson process with rate controlled by `ma_util`. Due dates are assigned using `DDT ~ U(0.5, 3.0)`.

---

## Pretrained Model

The pretrained MACD model is provided in `model/`. This model was trained under static conditions with 5 machines, 10 products, and 20 jobs, and can be applied directly to any Brandimarte or Hurink factory configuration without retraining.

To use the pretrained model, ensure `"load_model": true` in `config.json` and that the model file is present at `model_location` (default: `"./model/"`).

---

## Citation

If you use this code or the C-DFJSP formulation, please cite:

```bibtex
@article{johnson2026macd,
  author  = {Johnson, Dazzle and Chen, Gang and Lu, Yuqian},
  title   = {Multi-Agent Continuous Decision-Making for the Continuous
             Dynamic Flexible Job Shop Scheduling Problem},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2026},
  note    = {DOI to be added on publication}
}
```

The C-DFJSP problem formulation was first introduced in:

```bibtex
@inproceedings{johnson2024macsched,
  author    = {Johnson, Dazzle and Chen, Gang and Lu, Yuqian},
  title     = {Multi-Agent Scheduler for the Continuous Dynamic Flexible
               Job Shop Scheduling Problem},
  booktitle = {2024 IEEE 20th International Conference on Automation
               Science and Engineering (CASE)},
  pages     = {2924--2930},
  year      = {2024},
  doi       = {10.1109/CASE59546.2024.10711538}
}
```

---

## Acknowledgements

The base GNN-RL scheduling framework builds on the architecture introduced by:

> Song et al. (2023), *Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning*, IEEE Transactions on Industrial Informatics, 19(2), 1600–1610. DOI: 10.1109/TII.2022.3189725

The C-DFJSP environment, multi-agent framework, negotiation strategy, and factory-invariant feature design are original contributions of this work.

---

## Contact

Corresponding author: Yuqian Lu -- yuqian.lu@auckland.ac.nz  
Industrial AI Research Group, University of Auckland, New Zealand
