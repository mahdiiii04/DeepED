#!/bin/bash
python ./scripts/plot.py --algos deep_ed mappo ippo qmix --scenario team_goal_shift --db ./outputs/TeamGoalShift/results.db --out_dir ./outputs/TeamGoalShift/plots --no_title
python ./scripts/analyze_rotating.py --db ./outputs/TeamGoalShift/results.db --scenario team_goal_shift --algos deep_ed mappo ippo qmix --csv ./outputs/TeamGoalShift/analysis.csv
# same analysis on the value gap (distance to the optimal team value), not only NashConv:
python ./scripts/analyze_rotating.py --db ./outputs/TeamGoalShift/results.db --scenario team_goal_shift --algos deep_ed mappo ippo qmix --nash_key nash/value_gap --eps 1.0 --csv ./outputs/TeamGoalShift/analysis_value_gap.csv
