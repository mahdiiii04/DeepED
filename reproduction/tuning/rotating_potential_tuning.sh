#!/bin/bash
# Equal small tuning budget for every method, on SEPARATE seeds (100-102), then freeze the
# best setting per method for the final seeds 0-9. One experiment_name per setting because
# the results DB key (env, scenario, algo, seed) does not include hyperparameters.
CFG=../configs/matrix_games/rotating_potential
SEEDS=100,101,102

for algo in mappo ippo; do
  for ent in 0.01 0.05 0.1; do
    for lr in 1e-4 5e-4; do
      CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path $CFG --config-name $algo \
        experiment_name=RPotTune_${algo}_ent${ent}_lr${lr} loss.entropy_eps=$ent train.lr=$lr \
        --multirun seed=$SEEDS
    done
  done
done

# optional: advantage normalisation for the PPO baselines
for algo in mappo ippo; do
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path $CFG --config-name $algo \
    experiment_name=RPotTune_${algo}_advnorm loss.normalize_advantage=true --multirun seed=$SEEDS
done

for alpha in 0.05 0.1 0.2; do
  for alr in 1e-4 5e-4; do
    CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path $CFG --config-name deeped \
      experiment_name=RPotTune_deeped_a${alpha}_lr${alr} loss.alpha=$alpha train.actor_lr=$alr \
      --multirun seed=$SEEDS
  done
done

for lr in 5e-4 1e-3; do
  CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path $CFG --config-name qmix \
    experiment_name=RPotTune_qmix_lr${lr} train.lr=$lr --multirun seed=$SEEDS
done

# Score every setting with:  python ./scripts/analyze_rotating.py --db outputs/<name>/results.db --algos <algo>
