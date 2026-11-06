#!/bin/bash
# Method comparison on the ring potential game (cheap). Delete outputs/RingMain first if it exists.
for algo in mappo ippo deeped qmix; do
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/ring --config-name $algo --multirun seed=0,1,2,3,4,5,6,7,8,9
done
