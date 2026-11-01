"""
Per-phase and per-run statistics for phase-switching matrix games, read from results.db.

For every run:
  * phases are the maximal stretches with constant `env/phase`; the logged iteration at which
    the phase value changes belongs to the NEW phase (that is where NashConv spikes);
  * recovery time = env steps from the start of a phase until NashConv <= eps
    (censored at the phase length if it never gets there);
  * time-averaged NashConv, fraction of iterations above eps, peak NashConv.

Aggregation is over seeds, with a percentile bootstrap 95% CI of the mean.
The first phase (initial learning) is skipped by default.

    python scripts/analyze_rotating.py --db outputs/RotatingPotential/results.db \
        --algos deep_ed mappo ippo qmix --eps 0.1 --csv outputs/RotatingPotential/analysis.csv
"""
import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np


def load_metric(conn, run_id, metric):
    rows = conn.execute(
        "SELECT global_step, value FROM metrics WHERE run_id=? AND metric=? ORDER BY global_step",
        (run_id, metric),
    ).fetchall()
    if not rows:
        return np.array([]), np.array([])
    s, v = zip(*rows)
    return np.array(s, dtype=float), np.array(v, dtype=float)


def analyze_run(conn, run_id, eps, skip_first, nash_key, phase_key):
    s_n, nash = load_metric(conn, run_id, nash_key)
    s_p, phase = load_metric(conn, run_id, phase_key)
    if len(s_n) == 0 or len(s_p) == 0:
        return None
    n = min(len(s_n), len(s_p))
    steps, nash, phase = s_n[:n], nash[:n], phase[:n]

    change = [i for i in range(1, n) if phase[i] != phase[i - 1]]
    bounds = [0] + change + [n]
    segs = list(zip(bounds[:-1], bounds[1:]))
    if len(segs) < 2:
        return None

    lens = np.array([b - a for a, b in segs])
    full = np.median(lens[:-1]) if len(lens) > 1 else lens[0]
    if lens[-1] < 0.9 * full:          # drop a truncated final phase
        segs = segs[:-1]
    if skip_first:
        segs = segs[1:]
    if not segs:
        return None

    rec, cens, peak, mean_n = [], [], [], []
    for a, b in segs:
        seg = nash[a:b]
        hit = np.nonzero(seg <= eps)[0]
        phase_len = steps[b - 1] - steps[a] + (steps[1] - steps[0])
        if len(hit):
            rec.append(steps[a + hit[0]] - steps[a]); cens.append(0)
        else:
            rec.append(phase_len); cens.append(1)
        peak.append(seg.max()); mean_n.append(seg.mean())

    start = segs[0][0]
    tail = nash[start:segs[-1][1]]
    return dict(
        mean_nash=float(tail.mean()),
        frac_above_eps=float((tail > eps).mean()),
        mean_peak=float(np.mean(peak)),
        mean_recovery_steps=float(np.mean(rec)),
        censored_frac=float(np.mean(cens)),
        n_phases=len(segs),
    )


def boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--scenario", default="rotating_potential")
    p.add_argument("--algos", nargs="*", default=None)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--skip_first", type=int, default=1, choices=[0, 1])
    p.add_argument("--nash_key", default="nash/nash_conv")
    p.add_argument("--phase_key", default="env/phase")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    if not Path(args.db).exists():
        raise FileNotFoundError(args.db)
    conn = sqlite3.connect(args.db)
    algos = args.algos or [r[0] for r in conn.execute(
        "SELECT DISTINCT algo_name FROM runs WHERE scenario=?", (args.scenario,))]

    keys = ["mean_nash", "frac_above_eps", "mean_peak", "mean_recovery_steps", "censored_frac"]
    per_run_rows, summary = [], {}
    for algo in algos:
        runs = conn.execute(
            "SELECT run_id, seed FROM runs WHERE scenario=? AND algo_name=? ORDER BY seed",
            (args.scenario, algo)).fetchall()
        res = []
        for run_id, seed in runs:
            r = analyze_run(conn, run_id, args.eps, bool(args.skip_first), args.nash_key, args.phase_key)
            if r is None:
                print(f"[warn] {algo} seed {seed}: no usable phases")
                continue
            res.append(r)
            per_run_rows.append(dict(algo=algo, seed=seed, **r))
        summary[algo] = res

    print(f"\neps={args.eps}  skip_first={bool(args.skip_first)}  (mean [95% bootstrap CI] over seeds)\n")
    head = f"{'algo':22s} {'n':>3s} " + " ".join(f"{k:>30s}" for k in keys)
    print(head); print("-" * len(head))
    for algo, res in summary.items():
        if not res:
            continue
        cells = []
        for k in keys:
            vals = [r[k] for r in res]
            lo, hi = boot_ci(vals)
            cells.append(f"{np.mean(vals):9.3f} [{lo:7.3f},{hi:7.3f}]".rjust(30))
        print(f"{algo:22s} {len(res):3d} " + " ".join(cells))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["algo", "seed", *keys, "n_phases"])
            w.writeheader()
            for r in per_run_rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"\nper-run results -> {args.csv}")
    conn.close()


if __name__ == "__main__":
    main()
