#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/gridworld/new_shifting_role --config-name mappo --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/gridworld/new_shifting_role --config-name ippo --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/gridworld/new_shifting_role --config-name deeped --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/gridworld/new_shifting_role --config-name qmix --multirun seed=0,1,2,3,4,5,6,7,8,9
