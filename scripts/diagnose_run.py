"""
Print key metrics at fixed iteration offsets after every phase switch, for ONE run.
Use it to see whether a NashConv excursion coincides with a high clamp fraction, entropy collapse,
a critic blow-up, or simply with slow recovery.

    python scripts/diagnose_run.py --db outputs/TeamGoalShift/results.db --algo deep_ed --seed 0 \
        --scenario team_goal_shift --offsets 0 1 2 5 10 20 30 39
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_METRICS = [
    "nash/nash_conv", "nash/value_gap", "reward/mean_episode_reward", "all_on_goal_rate",
    "loss/entropy", "loss/critic", "grad/norm", "diag/clamp_frac", "diag/step_cond_max",
]


def load(conn, run_id, metric):
    rows = conn.execute(
        "SELECT global_step, value FROM metrics WHERE run_id=? AND metric=? ORDER BY global_step",
        (run_id, metric)).fetchall()
    return {int(s): v for s, v in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--scenario", default="team_goal_shift")
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offsets", type=int, nargs="*", default=[0, 1, 2, 5, 10, 20, 30, 39])
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--phase_key", default="env/phase")
    args = ap.parse_args()

    if not Path(args.db).exists():
        raise FileNotFoundError(args.db)
    conn = sqlite3.connect(args.db)
    row = conn.execute(
        "SELECT run_id FROM runs WHERE scenario=? AND algo_name=? AND seed=?",
        (args.scenario, args.algo, args.seed)).fetchone()
    if row is None:
        raise SystemExit("run not found")
    run_id = row[0]

    phase = load(conn, run_id, args.phase_key)
    steps = sorted(phase)
    pv = [phase[s] for s in steps]
    starts = [0] + [i for i in range(1, len(steps)) if pv[i] != pv[i - 1]]
    ends = starts[1:] + [len(steps)]
    data = {m: load(conn, run_id, m) for m in args.metrics}
    avail = [m for m in args.metrics if data[m]]
    missing = [m for m in args.metrics if not data[m]]
    if missing:
        print("not logged:", ", ".join(missing))

    head = f"{'phase#':>6s} {'iter':>5s} {'off':>4s} " + " ".join(f"{m.split('/')[-1][:14]:>14s}" for m in avail)
    print(head); print("-" * len(head))
    for k, (a, b) in enumerate(zip(starts, ends)):
        for off in args.offsets:
            i = a + off
            if i >= b:
                continue
            vals = " ".join(f"{data[m].get(steps[i], float('nan')):14.4f}" for m in avail)
            print(f"{k:6d} {i:5d} {off:4d} {vals}")
        print()
    conn.close()


if __name__ == "__main__":
    main()
