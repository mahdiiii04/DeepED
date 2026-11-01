#!/bin/bash
# Run AFTER the main comparison works; fix/tune the target variants on matrix games first.
for cfg in smith bnn replicator softmax safe; do
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/ablation/team_goal_shift --config-name $cfg --multirun seed=0,1,2,3,4,5,6,7,8,9
done
