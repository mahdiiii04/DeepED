#!/bin/bash
# Team-reward game WITH a per-agent goal bonus (agent_goal_reward=0.1): the old convention is exploitable,
# NashConv registers each switch. Pair with team_goal_shift.sh (pure coordination, bonus 0).
for algo in mappo ippo deeped qmix; do
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/gridworld/team_goal_shift_shaped --config-name $algo --multirun seed=0,1,2,3,4,5,6,7,8,9
done
