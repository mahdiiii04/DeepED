#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/biased_rps --config-name mappo --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/biased_rps --config-name ippo --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/biased_rps --config-name deeped --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/matrix_games/biased_rps --config-name qmix --multirun seed=0,1,2,3,4,5,6,7,8,9
