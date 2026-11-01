#!/bin/bash
# Target-construction ablation on the rotating potential game (all share experiment_name AblationRPot).

for cfg in smith bnn replicator softmax safe; do
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/ablation/rotating_potential --config-name $cfg --multirun seed=0,1,2,3,4,5,6,7,8,9
done
