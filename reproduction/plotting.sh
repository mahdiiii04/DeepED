#!/bin/bash

python ./scripts/plot.py --algos deep_ed mappo ippo qmix --scenario biased_rps --db .\outputs\BiasedRPS\results.db --out_dir .\outputs\BiasedRPS\plots --no_title
python ./scripts/plot.py --algos deep_ed mappo ippo qmix --scenario battle_of_sexes --db .\outputs\BattleOfSexes\results.db --out_dir .\outputs\BattleOfSexes\plots --no_title

python .\scripts\plot.py --algos deep_ed mappo ippo qmix --scenario role_nav --db .\outputs\RoleShifting\results.db --out_dir .\outputs\RoleShifting\plots --no_title
python .\scripts\plot.py --algos deep_ed mappo ippo qmix --scenario ns_role_nav --db .\outputs\NSRoleShifting\results.db --out_dir .\outputs\NSRoleShifting\plots --no_title
python .\scripts\plot.py --algos deep_ed mappo ippo qmix --scenario ns_cooperative_nav --db .\outputs\NSCooperation\results.db --out_dir .\outputs\NSCooperation\plots --no_title

python .\scripts\plot.py --algos deep_ed mappo ippo qmix --scenario vmas --db .\outputs\SimpleSpread\results.db --out_dir .\outputs\SimpleSpread\plots --no_title 
python .\scripts\plot.py --algos deep_ed mappo ippo qmix --scenario vmas --db .\outputs\Balance\results.db --out_dir .\outputs\Balance\plots --no_title 

python .\scripts\plot.py --algos deep_ed deep_ed_bnn deep_ed_replicator --scenario biased_rps --db .\outputs\AblationBRPS\results.db --out_dir .\outputs\AblationBRPS\plots --no_title  