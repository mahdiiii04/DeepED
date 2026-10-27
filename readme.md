# Tracking Non-Stationary Equilibria in Deep Multi-Agent Reinforcement Learning via Evolutionary Dynamics

[![Paper](https://img.shields.io/badge/paper-link--pending-b31b1b.svg)](#)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](#installation)

Official implementation of **Deep Evolutionary Dynamics (DeepED)** and the experiments from the paper:

> **Tracking Non-Stationary Equilibria in Deep Multi-Agent Reinforcement Learning via Evolutionary Dynamics**
> [Author Name, Author Name] — [Venue, Year]
> **Paper:** [Add paper link here] · **Code:** this repository

<!-- TODO: Replace this with 2-3 sentences from your abstract — the problem DeepED solves,
     the core idea (tracking non-stationary equilibria via evolutionary dynamics in deep MARL),
     and the headline empirical result. This is usually the first thing readers look for. -->

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Environments](#environments)
- [Running an Experiment](#running-an-experiment)
- [GPU Usage](#gpu-usage)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Generating Plots](#generating-plots)
- [Outputs](#outputs)
- [Algorithms](#algorithms)
- [Shared Utilities](#shared-utilities)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Before You Publish](#before-you-publish)

---

## Overview

This repository provides:

- A PyTorch / TorchRL implementation of **DeepED**
- Baseline MARL algorithms: **MADDPG**, **MAPPO**, **QMIX**, and others under `algos/`
- Custom **matrix-game** and **gridworld** environments, plus a **VMAS** integration via TorchRL
- Hydra-based experiment configs, reproduction scripts, and plotting utilities used to generate the paper's results

---

## Repository Structure

```text
.
├── algos/
│   ├── deeped/
│   │   ├── algorithm.py       # DeepED implementation
│   │   ├── deeped_loss.py     # DeepED-specific loss function
│   │   └── __init__.py
│   ├── maddpg/                # MADDPG implementation
│   ├── mappo/                 # MAPPO implementation
│   ├── qmix/                  # QMIX implementation
│   └── __init__.py
│
├── configs/
│   ├── ablation/               # Ablation-study configurations
│   ├── gridworld/               # Gridworld configurations
│   ├── matrix_games/            # Matrix-game configurations
│   └── vmas/                    # VMAS configurations
│
├── envs/
│   ├── gridworld/                # Gridworld environments
│   ├── matrix_games/             # Matrix-game environments
│   └── __init__.py               # Environment factory and dispatch logic
│
├── outputs/                      # Generated experiment outputs
│
├── reproduction/
│   ├── ablation/                 # Ablation experiments
│   ├── gridworld/                # Gridworld experiments
│   ├── matrix_games/             # Matrix-game experiments
│   ├── vmas/                     # VMAS experiments
│   └── plotting.sh               # Plotting commands
│
├── scripts/
│   ├── train.py                  # Training entry point
│   └── plot.py                   # Plotting entry point
│
├── shared/
│   ├── avg_policy.py             # Average-policy extraction utilities
│   ├── db.py                     # SQLite results database utilities
│   ├── done_transformation.py    # TorchRL done-signal transformation
│   ├── evaluate_policy.py        # Policy evaluation utilities
│   └── __init__.py               # Package initialization
│
├── environment.yml               # Conda environment specification
└── requirements.txt              # Python package requirements
```

The `vmas` environments are provided through TorchRL's VMAS integration and are not implemented locally in the `envs/` directory.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

Replace `USERNAME/REPOSITORY` with the actual GitHub repository path.

### 2. Create the Conda environment

```bash
conda env create -f environment.yml -n deeped
```

### 3. Activate the environment

```bash
conda activate deeped
```

### 4. Install the Python dependencies

```bash
python -m pip install -r requirements.txt
```

> **Note:** If `environment.yml` already pins all required packages, step 4 may be redundant. Consider trimming `requirements.txt` to only the packages *not* already captured by the conda environment, to avoid version conflicts for anyone reproducing your setup.

### 5. Verify the installation

```bash
python -c "import torch, torchrl, tensordict, vmas; print('Core imports successful')"
```

The repository was tested with the following core package versions:

```text
PyTorch:    2.11.0
TorchRL:    0.11.1
TensorDict: 0.11.0
VMAS:       1.5.2
```

The exported environment currently uses the CPU version of PyTorch. GPU users may need to install a CUDA-compatible PyTorch build appropriate for their system.

---

## Quick Start

Once installed, run a short training job to confirm everything works end-to-end:

```bash
python ./scripts/train.py \
  --config-path ./configs/matrix_games/biased_rps \
  --config-name deeped
```

This trains DeepED on the biased Rock-Paper-Scissors matrix game with default settings. Results and logs are written to `outputs/`.

---

## Configuration

Experiment configurations are stored as Hydra YAML files in the `configs/` directory, organized by environment family, scenario, and algorithm.

A configuration defines, among other settings:

- Random seed
- Experiment name
- Environment type and scenario
- Number of environments
- Maximum episode length
- Model architecture
- Data collection parameters
- Buffer settings
- Actor and critic optimization parameters
- Loss-function parameters
- Evaluation frequency
- Hydra output settings

For example:

```yaml
seed: 0
experiment_name: BiasedRPS
env_type: matrix_games
algo_name: deep_ed
sub_exp: ${env.scenario_name}/${algo_name}

env:
  scenario_name: biased_rps
  num_envs: null
  max_steps: 100
  device: null

model:
  shared_params: false
  centralized_critic: false
  depth: 2
  num_cells: 64

collector:
  frames_per_batch: 51200
  n_iters: 100
  total_frames: null

buffer:
  memory_size: null

train:
  num_epochs: 10
  minibatch_size: 4096
  actor_lr: 1e-4
  critic_lr: 1e-3
  max_grad_norm: 40.0
  device: null
  tau: 0.005

loss:
  entropy_eps: 0.01
  alpha: 0.1
  gamma: 0.9

eval:
  frequency: 10
```

The available configuration fields may differ depending on the environment and algorithm.

---

## Environments

The repository supports three environment types:

| `env_type`     | Implementation                                    |
|----------------|----------------------------------------------------|
| `matrix_games` | `envs.matrix_games.envs.MatrixGameFactory`          |
| `gridworld`    | `envs.gridworld.envs.GridWorldFactory`              |
| `vmas`         | `torchrl.envs.libs.vmas.VmasEnv`                    |

The environment and scenario are selected through the configuration:

```yaml
env_type: matrix_games

env:
  scenario_name: biased_rps
  num_envs: null
  max_steps: 100
  device: null
```

The shared environment factory in `envs/__init__.py` creates the appropriate environment based on `env_type`.

### Scenarios used in the paper

Based on `reproduction/plotting.sh`, the following scenarios are covered:

**`matrix_games`**
- Biased Rock-Paper-Scissors (`biased_rps`)
- Battle of the Sexes (`bos`)

**`gridworld`**
- Role-shifting environments
- Non-stationary role-navigation environments
- Non-stationary cooperative navigation

**`vmas`**
- VMAS Simple Spread
- VMAS Balance

> Config directory names above (e.g. `biased_rps`, `bos`) are best guesses based on convention — double-check these against `configs/matrix_games/` and adjust if they differ. Consider adding the remaining scenarios' exact config-folder names too, so readers don't need to browse `configs/` to find them.

---

## Running an Experiment

Training is launched through `scripts/train.py` and a Hydra configuration.

For example, to run DeepED on the biased Rock-Paper-Scissors matrix game:

```bash
python ./scripts/train.py \
  --config-path ./configs/matrix_games/biased_rps \
  --config-name deeped
```

To run an experiment over multiple random seeds:

```bash
python ./scripts/train.py \
  --config-path ./configs/matrix_games/biased_rps \
  --config-name deeped \
  --multirun seed=0,1,2,3,4,5,6,7,8,9
```

The exact configuration paths and names depend on the experiment being run.

---

## GPU Usage

Restrict training to a specific GPU with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py \
  --config-path ./configs/matrix_games/biased_rps \
  --config-name deeped \
  --multirun seed=0,1,2,3,4,5,6,7,8,9
```

Change the index (e.g. to `1`) to use a different GPU.

<details>
<summary>Windows PowerShell</summary>

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python .\scripts\train.py `
  --config-path .\configs\matrix_games\biased_rps `
  --config-name deeped `
  --multirun seed=0,1,2,3,4,5,6,7,8,9
```

</details>

For CPU execution, omit `CUDA_VISIBLE_DEVICES` and set the relevant device fields in the configuration to `cpu`, if required.

---

## Reproducing the Experiments

The `reproduction/` directory contains shell scripts for reproducing the experiments reported in the paper, using the provided configurations:

```text
reproduction/
├── ablation/
├── gridworld/
├── matrix_games/
├── vmas/
└── plotting.sh
```

For example, the matrix-game script for biased Rock-Paper-Scissors runs MAPPO, IPPO, DeepED, and QMIX over ten random seeds:

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py \
  --config-path ../configs/matrix_games/biased_rps \
  --config-name mappo \
  --multirun seed=0,1,2,3,4,5,6,7,8,9

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py \
  --config-path ../configs/matrix_games/biased_rps \
  --config-name ippo \
  --multirun seed=0,1,2,3,4,5,6,7,8,9

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py \
  --config-path ../configs/matrix_games/biased_rps \
  --config-name deeped \
  --multirun seed=0,1,2,3,4,5,6,7,8,9

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py \
  --config-path ../configs/matrix_games/biased_rps \
  --config-name qmix \
  --multirun seed=0,1,2,3,4,5,6,7,8,9
```

Run a reproduction script with:

```bash
bash reproduction/matrix_games/biased_rps.sh
```

The reproduction scripts are intended to be run from the repository root using Bash, Git Bash, WSL, or a Linux environment. Training should be completed before running the plotting script.

---

## Generating Plots

The plotting script reads experiment results from the SQLite databases generated during training and saves the resulting figures to the corresponding `plots/` directories.

Run:

```bash
bash reproduction/plotting.sh
```

The plotting commands follow this pattern:

```bash
python ./scripts/plot.py \
  --algos deep_ed mappo ippo qmix \
  --scenario biased_rps \
  --db ./outputs/BiasedRPS/results.db \
  --out_dir ./outputs/BiasedRPS/plots \
  --no_title
```

The required result database must already exist before plotting:

```text
outputs/
└── BiasedRPS/
    ├── results.db
    └── plots/
```

> ⚠️ Make sure the output directory names in `reproduction/plotting.sh` exactly match the names generated by the corresponding training configurations — see [Before You Publish](#before-you-publish) for a specific naming issue to resolve.

---

## Outputs

Training and evaluation outputs are stored in the `outputs/` directory. The top-level experiment directory is determined by the experiment configuration, e.g.:

```yaml
experiment_name: BiasedRPS
```

produces:

```text
outputs/
└── BiasedRPS/
    ├── plots/
    ├── tb_logs/
    │   ├── deep_ed/
    │   ├── ippo/
    │   ├── maddpg/
    │   ├── mappo/
    │   ├── qmix/
    │   └── vdn/
    └── results.db
```

- `plots/` — generated experiment figures
- `tb_logs/` — TensorBoard logs organized by algorithm
- `results.db` — SQLite database containing runs, training metrics, and policy snapshots

To inspect TensorBoard logs:

```bash
tensorboard --logdir outputs/BiasedRPS/tb_logs
```

---

## Algorithms

The `algos/` directory contains the implementations of DeepED and the baseline algorithms used in the experiments. Each algorithm is organized as a separate Python package and generally contains:

- `algorithm.py` — main algorithm implementation
- `__init__.py` — package initialization
- Additional algorithm-specific modules

The DeepED package also contains a dedicated loss module:

```text
algos/deeped/
├── algorithm.py
├── deeped_loss.py   # DeepED-specific loss function
└── __init__.py
```

---

## Shared Utilities

The `shared/` directory contains reusable components used throughout training, evaluation, and plotting:

| File | Purpose |
|---|---|
| `avg_policy.py` | Extracts average policies from rollout trajectories for actor and Q-network policies |
| `db.py` | Defines the SQLite schema and stores training runs, metrics, and policy snapshots |
| `done_transformation.py` | TorchRL transformation adapting done signals to the environments' reward structure |
| `evaluate_policy.py` | Evaluates policies over multiple episodes and computes mean episode rewards |
| `__init__.py` | Package initialization |

---

## Reproducibility

For the most faithful reproduction:

1. Create the Conda environment.
2. Install the requirements.
3. Use the provided configuration files.
4. Run the corresponding reproduction script.
5. Use the same random seeds.
6. Generate plots only after training has completed.
7. Ensure the same hardware and device settings are used where possible.

---

## Citation

If you use this repository or DeepED in your research, please cite:

```bibtex
@article{YOUR_CITATION_KEY,
  title   = {Tracking Non-Stationary Equilibria in Deep Multi-Agent Reinforcement Learning via Evolutionary Dynamics},
  author  = {Author Name and Author Name},
  journal = {Venue},
  year    = {YEAR},
  doi     = {DOI}
}
```

Replace the placeholder citation with the final citation information for the paper.

---

## License

This project is released under the [LICENSE NAME] License — see [`LICENSE`](./LICENSE) for details.

---

## Before You Publish

- [ ] Replace `https://github.com/USERNAME/REPOSITORY.git` with the real clone URL
- [ ] Add the paper link (arXiv / venue page) in the header and Citation section
- [ ] Fill in author names/affiliations and the badges at the top
- [ ] Write the 2–3 sentence Overview summary (marked with a TODO above)
- [ ] Fill in the BibTeX citation (authors, journal/venue, year, DOI)
- [ ] Add the actual license text and confirm it matches the `LICENSE` file
- [ ] Double-check all command paths match your final repository layout
- [ ] **Resolve the ablation output-directory name** — confirm whether it's `AblationRPS` or `AblationBRPS`, and make sure `reproduction/plotting.sh`, the corresponding training config's `experiment_name`, and the `outputs/` directory all use the same name consistently