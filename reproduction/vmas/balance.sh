#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/vmas/balance --config-name mappo --multirun seed=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/vmas/balance --config-name deeped --multirun seed=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/vmas/balance --config-name ippo --multirun seed=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/vmas/balance --config-name qmix --multirun seed=0,1,2,3,4,5
