"""
Compare the measured time-average of NashConv^2 with the right-hand side of Theorem 1, using quantities
MEASURED during training on the ring game (eps_a, eps_c, B, Lambda, V_T).

    python scripts/check_bound.py --db outputs/RingMain/results.db --algo deep_ed

Caveats (state them in the paper): (i) the theorem's step t is one target-then-actor update, here one
logged iteration contains many gradient steps, so T is only an approximate mapping; (ii) eps_a is a proxy
(distance of the actor to its own Smith target after the iteration); (iii) with q_noise > 0 the injected noise
is added to eps_c through diag/noise_sup. The bound is expected to hold with a large margin.
"""
import argparse
import sqlite3

import numpy as np
import yaml


def series(conn, run_id, metric):
    rows = conn.execute("SELECT value FROM metrics WHERE run_id=? AND metric=? ORDER BY global_step", (run_id, metric)).fetchall()
    return np.array([r[0] for r in rows], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--algo", default="deep_ed")
    ap.add_argument("--stat", default="max", choices=["max", "p95", "mean"])
    a = ap.parse_args()
    agg = {"max": np.max, "p95": lambda x: np.percentile(x, 95), "mean": np.mean}[a.stat]
    conn = sqlite3.connect(a.db)
    print(f"{'seed':>4s} {'LHS mean N^2':>13s} {'RHS bound':>12s} {'RHS/LHS':>9s}   shares of RHS: init / change / actor / critic / simult   (Lambda, eps_a, eps_c, B)")
    for run_id, seed, cfg_json in conn.execute("SELECT run_id, seed, cfg_json FROM runs WHERE algo_name=? ORDER BY seed", (a.algo,)):
        cfg = yaml.safe_load(cfg_json)
        al, g = float(cfg["loss"]["alpha"]), float(cfg["loss"]["gamma"])
        kw = cfg["env"]["kwargs"]; Rmax = float(kw.get("payoff_scale", 1.0)) * float(kw.get("v", 1.0))
        N, M = 2, int(kw.get("n_actions", 3)); CR = max(1.0, Rmax)
        nc = series(conn, run_id, "nash/nash_conv"); T = len(nc)
        need = ["diag/eps_a", "diag/eps_c", "diag/critic_range", "diag/lambda", "env/V_T"]
        data = {m: series(conn, run_id, m) for m in need}
        if T == 0 or any(len(v) == 0 for v in data.values()):
            print(f"{seed:4d}  missing metrics"); continue
        eps_a, B, Lam, VT = agg(data["diag/eps_a"]), agg(data["diag/critic_range"]), agg(data["diag/lambda"]), data["env/V_T"][-1]
        eps_c = agg(data["diag/eps_c"]) + (agg(series(conn, run_id, "diag/noise_sup")) if len(series(conn, run_id, "diag/noise_sup")) else 0.0)
        kappa = Rmax * (1 + g) / (1 - g) ** 3
        terms = dict(init=4 * N * Rmax / ((1 - g) * T), change=2 * N * CR * VT / ((1 - g) ** 2 * T),
                     actor=N * Rmax * eps_a / (1 - g) ** 2, critic=4 * al * N * M * Rmax * eps_c / (1 - g) ** 2,
                     simult=2 * kappa * N * (N - 1) * (M - 1) ** 2 * B ** 2 * al ** 2)
        rhs = Lam * N / (al * (1 - g)) * sum(terms.values()); lhs = float(np.mean(nc ** 2)); tot = sum(terms.values())
        print(f"{seed:4d} {lhs:13.4f} {rhs:12.3e} {rhs / max(lhs, 1e-12):9.1e}   "
              + " / ".join(f"{terms[k] / tot:5.2f}" for k in terms) + f"   ({Lam:.2f}, {eps_a:.3f}, {eps_c:.3f}, {B:.2f})")
    conn.close()


if __name__ == "__main__":
    main()
