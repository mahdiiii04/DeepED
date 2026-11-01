#!/bin/bash
# Markov potential game (team reward, moving goal). Smoke test first (see APPLY_GRIDWORLD.md).
for algo in mappo ippo deeped qmix; do
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/gridworld/team_goal_shift --config-name $algo --multirun seed=0,1,2,3,4,5,6,7,8,9
done
