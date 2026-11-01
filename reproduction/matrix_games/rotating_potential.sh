#!/bin/bash
# Main comparison, 10 seeds, same seeds (=> same phase schedule) for every algorithm.
# Delete outputs/RotatingPotential first if it contains test runs (results.db appends).

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/rotating_potential --config-name mappo --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/rotating_potential --config-name ippo --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/rotating_potential --config-name deeped --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/rotating_potential --config-name qmix --multirun seed=0,1,2,3,4,5,6,7,8,9

# Controls (uncomment; each needs its own experiment_name so DBs do not mix):
# pure team game (no dummy terms):
# CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/rotating_potential --config-name deeped experiment_name=RPotTeam env.kwargs.dummy_scale=0.0 --multirun seed=0,1,2,3,4,5,6,7,8,9
# cyclic phase order:
# CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/rotating_potential --config-name deeped experiment_name=RPotCyclic env.kwargs.phase_order=cyclic --multirun seed=0,1,2,3,4,5,6,7,8,9
