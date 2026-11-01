#!/bin/bash
# Append these lines to reproduction/plotting.sh

python ./scripts/plot.py --algos deep_ed mappo ippo qmix --scenario rotating_potential --db ./outputs/RotatingPotential/results.db --out_dir ./outputs/RotatingPotential/plots --no_title

python ./scripts/plot.py --algos deep_ed deep_ed_bnn deep_ed_replicator deep_ed_softmax deep_ed_safe --scenario rotating_potential --db ./outputs/AblationRPot/results.db --out_dir ./outputs/AblationRPot/plots --no_title

python ./scripts/analyze_rotating.py --db ./outputs/RotatingPotential/results.db --algos deep_ed mappo ippo qmix --csv ./outputs/RotatingPotential/analysis.csv
python ./scripts/analyze_rotating.py --db ./outputs/AblationRPot/results.db --algos deep_ed deep_ed_bnn deep_ed_replicator deep_ed_softmax deep_ed_safe --csv ./outputs/AblationRPot/analysis.csv
