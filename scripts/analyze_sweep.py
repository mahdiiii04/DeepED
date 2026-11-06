"""
Aggregate a one-parameter sweep stored as several results.db files (one per setting).

    python scripts/analyze_sweep.py --glob "outputs/RingSweep_alpha_*/results.db" \
        --param loss.alpha --name alpha --out_dir outputs/ring_plots [--x_transform inv]

For every run: skip the first phase (initial learning), then average NashConv^2, NashConv and the value
gap; also averages the measured theory quantities when logged (diag/eps_a, diag/eps_c, diag/critic_range,
diag/lambda) and the variation budget rate V_T/T (env/V_T). Aggregates over seeds (mean, sd) per setting.
"""
import argparse
import csv
import glob
import sqlite3
from pathlib import Path

import numpy as np
import yaml


def dig(cfg, dotted):
    for k in dotted.split("."):
        cfg = cfg[k]
    return cfg


def series(conn, run_id, metric):
    rows = conn.execute("SELECT global_step, value FROM metrics WHERE run_id=? AND metric=? ORDER BY global_step",
                        (run_id, metric)).fetchall()
    return np.array([r[1] for r in rows], dtype=float)


def run_stats(conn, run_id):
    nc, ph = series(conn, run_id, "nash/nash_conv"), series(conn, run_id, "env/phase")
    if len(nc) == 0:
        return None
    n = len(nc)
    start = next((i for i in range(1, min(n, len(ph))) if ph[i] != ph[i - 1]), int(0.2 * n)) if len(ph) else int(0.2 * n)
    sl = slice(start, n)
    out = dict(n2=float(np.mean(nc[sl] ** 2)), n1=float(np.mean(nc[sl])))
    vg = series(conn, run_id, "nash/value_gap")
    if len(vg): out["value_gap"] = float(np.mean(vg[sl]))
    for m in ["diag/eps_a", "diag/eps_c", "diag/eps_c_avg", "diag/critic_range", "diag/lambda", "diag/clamp_frac"]:
        s = series(conn, run_id, m)
        if len(s): out[m] = float(np.mean(s[sl]))
    vt = series(conn, run_id, "env/V_T")
    if len(vt): out["V_T_per_iter"] = float(vt[-1] / len(vt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True); ap.add_argument("--param", required=True)
    ap.add_argument("--name", default="sweep"); ap.add_argument("--out_dir", default="outputs/sweeps")
    ap.add_argument("--scenario", default=None); ap.add_argument("--algo", default="deep_ed")
    ap.add_argument("--x_transform", default="none", choices=["none", "inv"])
    args = ap.parse_args()

    groups = {}
    for db in sorted(glob.glob(args.glob)):
        conn = sqlite3.connect(db)
        for run_id, algo, seed, cfg_json in conn.execute("SELECT run_id, algo_name, seed, cfg_json FROM runs"):
            if args.algo and algo != args.algo:
                continue
            x = float(dig(yaml.safe_load(cfg_json), args.param))
            st = run_stats(conn, run_id)
            if st: groups.setdefault(x, []).append(st)
        conn.close()
    if not groups:
        raise SystemExit("no runs found")

    keys = sorted({k for g in groups.values() for st in g for k in st})
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    xs = sorted(groups)
    print(f"{args.param:>12s} {'n':>3s} " + " ".join(f"{k:>18s}" for k in keys))
    rows = []
    for x in xs:
        g = groups[x]
        cells = {k: (np.mean([s[k] for s in g if k in s]), np.std([s[k] for s in g if k in s])) for k in keys}
        print(f"{x:12g} {len(g):3d} " + " ".join(f"{cells[k][0]:11.4f}±{cells[k][1]:6.3f}" for k in keys))
        rows.append(dict(x=x, n=len(g), **{f"{k}_mean": cells[k][0] for k in keys}, **{f"{k}_sd": cells[k][1] for k in keys}))
    with open(Path(args.out_dir) / f"{args.name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    X = [1.0 / x for x in xs] if args.x_transform == "inv" else xs
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(X, [r["n2_mean"] for r in rows], yerr=[r["n2_sd"] for r in rows], marker="o", lw=2, capsize=3)
    ax.set_xlabel(("1 / " if args.x_transform == "inv" else "") + args.param)
    ax.set_ylabel("time-averaged NashConv$^2$"); ax.grid(alpha=.3)
    if min(xs) > 0 and max(xs) / min(xs) > 20: ax.set_xscale("log")
    fig.tight_layout(); out = Path(args.out_dir) / f"{args.name}.pdf"; fig.savefig(out); print("saved", out)


if __name__ == "__main__":
    main()
