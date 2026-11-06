# Markov potential game (TeamGoalShift): how to apply

Copy over the same paths (these files supersede earlier versions of envs/__init__.py and
scripts/train.py / scripts/plot.py in the previous patch):
  NEW: envs/gridworld/team_nash.py, envs/gridworld/team_goal_shift.py, shared/grid_policy.py,
       configs/gridworld/team_goal_shift/*.yaml, configs/ablation/team_goal_shift/*.yaml,
       reproduction/gridworld/team_goal_shift.sh, reproduction/ablation/team_goal_shift.sh,
       reproduction/plotting_team_goal_shift.sh
  MOD: envs/gridworld/__init__.py (registers the env), envs/__init__.py, scripts/train.py, scripts/plot.py

Smoke test (CPU ok; checks wiring, exact NashConv, phase switching):
  python ./scripts/train.py --config-path ../configs/gridworld/team_goal_shift --config-name deeped \
      collector.n_iters=20 env.kwargs.phase_length=500 experiment_name=SmokeGrid
Then in the DB check: nash/nash_conv, nash/potential, nash/value_gap exist; env/phase changes every
5 iterations; the optimal value (nash/optimal_value) changes when the goal moves.
