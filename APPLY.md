# DeepED rotating-potential patch: how to apply

Copy the files in this tree over the same paths in your repo (they are drop-in replacements),
then make the three manual edits below. Not run end-to-end: torch/torchrl were not available
where this was prepared, so run the smoke test first.

## Files
Modified : envs/__init__.py, envs/matrix_games/envs.py, shared/avg_policy.py, scripts/train.py,
           scripts/plot.py, algos/__init__.py, algos/deeped/deeped_loss.py, algos/deeped/algorithm.py,
           configs/matrix_games/rotating_potential/{deeped,mappo,ippo,qmix}.yaml
New      : configs/ablation/rotating_potential/{smith,bnn,replicator,softmax,safe}.yaml,
           scripts/analyze_rotating.py,
           reproduction/matrix_games/rotating_potential.sh, reproduction/ablation/rotating_potential.sh,
           reproduction/tuning/rotating_potential_tuning.sh, reproduction/plotting_rotating_potential.sh

## Manual edits
1. algos/mappo/algorithm.py, in ClipPPOLoss(...): replace
       normalize_advantage=False,
   with
       normalize_advantage=cfg.loss.get("normalize_advantage", False),
2. Append the lines of reproduction/plotting_rotating_potential.sh to reproduction/plotting.sh.
3. .gitignore:  multirun/   outputs/*/tb_logs/   __pycache__/

## Before the real run
- Delete test results (results.db appends metrics for repeated seeds):
    rm -rf outputs/RotatingPotential            # PowerShell: Remove-Item -Recurse -Force outputs\RotatingPotential
- Smoke test (CPU is fine):
    python ./scripts/train.py --config-path ../configs/matrix_games/rotating_potential --config-name deeped collector.n_iters=20 env.kwargs.phase_length=500 experiment_name=Smoke
  Check in the DB: nash/nash_conv rises at the SAME global_step where env/phase changes,
  and diag/clamp_frac and diag/step_cond_max are logged for deep_ed.
- Tune on seeds 100-102 (reproduction/tuning/...), freeze settings, then run seeds 0-9.

## Behaviour changes to be aware of
- avg-policy extraction no longer advances the env clock, so phase_length now counts only collector
  steps (100 per iteration). phase_length=1500 gives a phase every 15 iterations.
- n_iters is 150 (about 9 phase changes); phase order is random with phase_seed=${seed}.
- gamma is 0.9 for all four algorithms; eval reward is disabled for this scenario.
