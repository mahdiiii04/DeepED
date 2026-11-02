#!/bin/bash
for cfg in smith bnn replicator softmax safe; do
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/ablation/team_goal_shift_shaped --config-name $cfg --multirun seed=0,1,2,3,4,5,6,7,8,9
done
