#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/vmas/simple_spread --config-name mappo --multirun seed=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/vmas/simple_spread --config-name deeped --multirun seed=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/vmas/simple_spread --config-name ippo --multirun seed=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/vmas/simple_spread --config-name qmix --multirun seed=0,1,2,3,4,5
