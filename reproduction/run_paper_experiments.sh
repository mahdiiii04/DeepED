#!/bin/bash
# Every run that feeds the paper, in order. Delete the matching outputs/<Experiment> folders first
# (results.db appends metrics for repeated seeds). Run from the repository root.
set -e

# 0. training-free theory check (no GPU)
python scripts/ideal_dynamics_check.py check
python scripts/ideal_dynamics_check.py sweep --param L     --values 5 10 20 40 80          --out paper_out/ideal_phase.pdf
python scripts/ideal_dynamics_check.py sweep --param alpha --values 0.02 0.05 0.1 0.2 0.4  --out paper_out/ideal_alpha.pdf

# 1. ring game (theory validation)
bash reproduction/ring/ring_main.sh
bash reproduction/ring/ring_sweeps.sh

# 2. rotating potential game + target ablation
bash reproduction/matrix_games/rotating_potential.sh
bash reproduction/ablation/rotating_potential.sh

# 3. battle of the sexes (stationary)
bash reproduction/matrix_games/battle_of_sexes.sh

# 4. potential GridWorld, both settings, + raw-step ablation on the shaped setting
bash reproduction/gridworld/team_goal_shift.sh
bash reproduction/gridworld/team_goal_shift_shaped.sh
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py --config-path ../configs/ablation/team_goal_shift_shaped --config-name safe --multirun seed=0,1,2,3,4,5,6,7,8,9

# 5. Biased RPS (outside the theory, empirical)
bash reproduction/matrix_games/biased_rps.sh

# 6. tables, figures, mechanism check
python scripts/make_paper_report.py --config paper/paper_report.yaml --out paper_out --plots
python scripts/premass_check.py --db outputs/AblationRPot/results.db --algos deep_ed deep_ed_safe deep_ed_replicator deep_ed_softmax deep_ed_bnn > paper_out/mechanism_rotpot.txt
