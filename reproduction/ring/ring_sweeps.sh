#!/bin/bash
# Theory-validation sweeps (DeepED only, 5 seeds each). One experiment_name per setting.
# Predictions of Theorem 1 being tested:
#   phase_length (change rate)   -> floor falls as phases get longer
#   alpha                        -> eps_a/alpha and alpha terms
#   q_noise (critic error eps_c) -> floor rises with eps_c
#   actor_lr (actor lag, eps_a)  -> floor rises when the actor lags its target
#   payoff_scale (change size)   -> floor scales with v_t (note: also rescales the Q-range B)
CFG=../configs/ring
SEEDS=0,1,2,3,4
run() { # name value override
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path $CFG --config-name deeped \
    experiment_name=RingSweep_$1_$2 $3 --multirun seed=$SEEDS
}
for v in 500 1000 2000 4000 8000;   do run phase $v   env.kwargs.phase_length=$v; done
for v in 0.025 0.05 0.1 0.2 0.4;    do run alpha $v   loss.alpha=$v; done
for v in 0.0 0.05 0.1 0.2 0.4;      do run qnoise $v  loss.q_noise=$v; done
for v in 3e-5 1e-4 3e-4 1e-3;       do run actorlr $v train.actor_lr=$v; done
for v in 0.5 1.0 2.0;               do run scale $v   env.kwargs.payoff_scale=$v; done
