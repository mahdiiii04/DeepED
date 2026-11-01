"""
Mechanism check for phase-switching matrix games.

For every phase change, look at the average policy logged just BEFORE the switch and ask:
how much probability did the agents give to the action that is about to become the
equilibrium action (the new `env/phase`)? Then ask whether the run recovered
(NashConv <= eps at some point before the next switch).

Hypothesis (multiplicative updates cannot revive an extinct action; Smith can):
    P(recover | action extinct) is high for Smith, low for replicator / exp-weights.

    python scripts/premass_check.py --db outputs/AblationRPot/results.db \
        --algos deep_ed deep_ed_replicator deep_ed_softmax deep_ed_bnn --extinct 0.02 --eps 0.1
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


def load_metric(conn, run_id, metric):
    rows = conn.execute(
        "SELECT global_step, value FROM metrics WHERE run_id=? AND metric=? ORDER BY global_step",
        (run_id, metric)).fetchall()
    if not rows:
        return np.array([]), np.array([])
    s, v = zip(*rows)
    return np.array(s, dtype=float), np.array(v, dtype=float)


def load_policy(conn, run_id):
    rows = conn.execute(
        "SELECT global_step, agent_idx, probs_json FROM policy_snapshots WHERE run_id=? "
        "ORDER BY global_step, agent_idx", (run_id,)).fetchall()
    if not rows:
        return np.array([]), None
    steps = sorted({r[0] for r in rows})
    n_ag = len({r[1] for r in rows})
    n_ac = len(json.loads(rows[0][2]))
    P = np.zeros((len(steps), n_ag, n_ac))
    idx = {s: i for i, s in enumerate(steps)}
    for st, ag, js in rows:
        P[idx[st], ag] = json.loads(js)
    return np.array(steps, dtype=float), P


def events_for_run(conn, run_id, eps):
    s_n, nash = load_metric(conn, run_id, "nash/nash_conv")
    s_p, phase = load_metric(conn, run_id, "env/phase")
    s_q, P = load_policy(conn, run_id)
    if len(s_n) == 0 or len(s_p) == 0 or P is None:
        return []
    n = min(len(s_n), len(s_p), len(s_q))
    nash, phase, P = nash[:n], phase[:n], P[:n]
    change = [i for i in range(1, n) if phase[i] != phase[i - 1]]
    ends = change[1:] + [n]
    out = []
    for i, e in zip(change, ends):
        new_a = int(phase[i])
        p_pre = P[i - 1, :, new_a]               # each agent's prob on the NEW equilibrium action, just before the switch
        out.append(dict(
            p_pre_min=float(p_pre.min()),
            p_pre_mean=float(p_pre.mean()),
            p_end_min=float(P[e - 1, :, new_a].min()),
            recovered=bool(nash[i:e].min() <= eps),
            nash_at_switch=float(nash[i]),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--scenario", default="rotating_potential")
    ap.add_argument("--algos", nargs="*", default=None)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--extinct", type=float, default=0.02,
                    help="pre-switch probability below this counts as 'extinct'")
    args = ap.parse_args()

    if not Path(args.db).exists():
        raise FileNotFoundError(args.db)
    conn = sqlite3.connect(args.db)
    algos = args.algos or [r[0] for r in conn.execute(
        "SELECT DISTINCT algo_name FROM runs WHERE scenario=?", (args.scenario,))]

    print(f"eps={args.eps}  extinct if pre-switch prob (min over agents) < {args.extinct}\n")
    head = (f"{'algo':20s} {'events':>6s} {'median p_pre':>13s} {'frac extinct':>13s} "
            f"{'recov|extinct':>14s} {'recov|alive':>12s} {'median p_end':>13s}")
    print(head); print("-" * len(head))
    for algo in algos:
        ev = []
        for (rid,) in conn.execute(
                "SELECT run_id FROM runs WHERE scenario=? AND algo_name=?", (args.scenario, algo)):
            ev += events_for_run(conn, rid, args.eps)
        if not ev:
            print(f"{algo:20s}  no data"); continue
        pre = np.array([e["p_pre_min"] for e in ev])
        end = np.array([e["p_end_min"] for e in ev])
        rec = np.array([e["recovered"] for e in ev])
        ext = pre < args.extinct

        def rate(m):
            return f"{rec[m].mean():.2f} (n={int(m.sum())})" if m.any() else "   n/a"

        print(f"{algo:20s} {len(ev):6d} {np.median(pre):13.3f} {ext.mean():13.2f} "
              f"{rate(ext):>14s} {rate(~ext):>12s} {np.median(end):13.3f}")
    conn.close()


if __name__ == "__main__":
    main()
