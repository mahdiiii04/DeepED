#!/bin/bash
python ./scripts/plot.py --algos deep_ed mappo ippo qmix --scenario team_goal_shift --db ./outputs/TeamGoalShiftShaped/results.db --out_dir ./outputs/TeamGoalShiftShaped/plots --no_title
python ./scripts/analyze_rotating.py --db ./outputs/TeamGoalShiftShaped/results.db --scenario team_goal_shift --algos deep_ed mappo ippo qmix --eps 0.5 --csv ./outputs/TeamGoalShiftShaped/analysis.csv
python ./scripts/analyze_rotating.py --db ./outputs/TeamGoalShiftShaped/results.db --scenario team_goal_shift --algos deep_ed mappo ippo qmix --nash_key nash/value_gap --eps 1.0 --csv ./outputs/TeamGoalShiftShaped/analysis_value_gap.csv
