# Ring potential game (theory-validation environment): how to apply

Copy over the same paths (supersedes earlier versions of envs/__init__.py, shared/grid_policy.py,
algos/deeped/deeped_loss.py, algos/deeped/algorithm.py, scripts/train.py):
  NEW : envs/ring/{__init__,ring_oracle,ring_game}.py, shared/ring_diag.py,
        configs/ring/{deeped,mappo,ippo,qmix}.yaml,
        scripts/{ideal_dynamics_check,analyze_sweep,check_bound}.py,
        reproduction/ring/{ring_main,ring_sweeps}.sh
  MOD : envs/__init__.py (ring branch), shared/grid_policy.py, algos/deeped/deeped_loss.py (q_noise, target_policy),
        algos/deeped/algorithm.py (q_noise, state_diagnostics), scripts/train.py (ring logging)

Step 0 (no GPU, no torch): python scripts/ideal_dynamics_check.py check
Step 1 smoke test:
  python ./scripts/train.py --config-path ../configs/ring --config-name deeped collector.n_iters=30 env.kwargs.phase_length=500 experiment_name=SmokeRing
  Expect in the DB: nash/nash_conv, nash/value_gap, diag/eps_c, diag/eps_a, diag/critic_range, diag/lambda, env/v_t (=1.0
  after the first switch), env/V_T growing. A converged policy hit by a switch should show NashConv ~3.3 and value gap ~3.3.
Step 2: reproduction/ring/ring_main.sh (method comparison) and ring_sweeps.sh (theory sweeps), then
  python scripts/analyze_sweep.py --glob "outputs/RingSweep_alpha_*/results.db" --param loss.alpha --name alpha --out_dir outputs/ring_plots
  python scripts/check_bound.py --db outputs/RingMain/results.db --algo deep_ed
