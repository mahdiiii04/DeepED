#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/ablation/biased_rps --config-name bnn --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/ablation/biased_rps --config-name replicator --multirun seed=0,1,2,3,4,5,6,7,8,9
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/ablation/biased_rps --config-name smith --multirun seed=0,1,2,3,4,5,6,7,8,9
