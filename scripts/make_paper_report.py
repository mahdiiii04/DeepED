"""
One command that turns the result databases into every table the paper needs.

    python scripts/make_paper_report.py --config paper/paper_report.yaml --out paper_out [--plots]

For each experiment in the config it writes (csv + markdown + LaTeX booktabs):
  summary_<view>   per-algorithm table: mean metric [95% CI over seeds], time above eps, recovery and settle
                   times (pooled over seed x phase, median [IQR]) with the share that never recovered/settled,
                   relapses per phase, number of seeds
  paired_<view>    paired differences versus a reference algorithm on the same seeds (same phase schedule):
                   mean difference [95% CI], wins, exact sign-test p-value
  per_seed_<view>  the raw per-seed numbers (appendix / reproducibility)
  diagnostics      theory-code link: clamp share, step-size condition, critic/actor error proxies, QMIX epsilon
and, for the sweeps and bound checks listed in the config, runs analyze_sweep.py / check_bound.py.
Everything is collected in <out>/REPORT.md. Confidence intervals are printed only for >= min_seeds seeds
(otherwise the [min, max] range is shown and flagged), so 2-seed pilot runs cannot masquerade as results.
"""
import argparse
import csv
import json
import math
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

DEFAULT_NAMES = {
    "deep_ed": "DeepED (clamp)", "deep_ed_safe": "DeepED (raw step)", "deep_ed_bnn": "BNN target",
    "deep_ed_replicator": "Replicator target", "deep_ed_softmax": "Exp.-weights target",
    "mappo": "MAPPO", "ippo": "IPPO", "qmix": "QMIX",
}


# ----------------------------------------------------------------------------- data access
def series(conn, run_id, metric):
    rows = conn.execute("SELECT global_step, value FROM metrics WHERE run_id=? AND metric=? ORDER BY global_step",
                        (run_id, metric)).fetchall()
    if not rows:
        return np.array([]), np.array([])
    s, v = zip(*rows)
    return np.array(s, dtype=float), np.array(v, dtype=float)


def view_series(conn, run_id, view):
    """(steps, values) of the view's quantity; norm_gap = 1 - num/den (normalised distance to the optimum)."""
    if "num_key" in view:
        s1, a = series(conn, run_id, view["num_key"]); s2, b = series(conn, run_id, view["den_key"])
        n = min(len(a), len(b))
        return s1[:n], 1.0 - a[:n] / np.maximum(b[:n], 1e-12)
    return series(conn, run_id, view["key"])


# ----------------------------------------------------------------------------- per-run statistics
def phase_stats(steps, vals, phase, eps, skip_first):
    n = min(len(steps), len(phase), len(vals))
    steps, vals, phase = steps[:n], vals[:n], phase[:n]
    change = [i for i in range(1, n) if phase[i] != phase[i - 1]]
    segs = list(zip([0] + change, change + [n]))
    if len(segs) < 2:
        return None
    lens = np.array([b - a for a, b in segs])
    if len(lens) > 1 and lens[-1] < 0.9 * np.median(lens[:-1]):
        segs = segs[:-1]
    if skip_first:
        segs = segs[1:]
    if not segs:
        return None
    dt = steps[1] - steps[0]
    rec, rec_c, settle, settle_c, relapses, peaks = [], [], [], [], [], []
    for a, b in segs:
        seg = vals[a:b]
        L = steps[b - 1] - steps[a] + dt
        hit = np.nonzero(seg <= eps)[0]
        rec.append(steps[a + hit[0]] - steps[a] if len(hit) else L); rec_c.append(0 if len(hit) else 1)
        above = seg > eps
        idx = np.nonzero(above)[0]
        if len(idx) == 0:
            settle.append(0.0); settle_c.append(0)
        elif idx[-1] == len(seg) - 1:
            settle.append(L); settle_c.append(1)
        else:
            settle.append(steps[a + idx[-1] + 1] - steps[a]); settle_c.append(0)
        relapses.append(int(np.sum(above[hit[0]:][1:] & ~above[hit[0]:][:-1])) if len(hit) else 0)
        peaks.append(seg.max())
    tail = vals[segs[0][0]:segs[-1][1]]
    return dict(mean=float(tail.mean()), mean_sq=float((tail ** 2).mean()), frac_above=float((tail > eps).mean()),
                peak=float(np.mean(peaks)), rec_list=rec, rec_c_list=rec_c, settle_list=settle, settle_c_list=settle_c,
                relapses=float(np.mean(relapses)), rec_mean=float(np.mean(rec)), settle_mean=float(np.mean(settle)),
                n_phases=len(segs))


def stationary_stats(steps, vals, eps, window):
    n = len(vals)
    if n == 0:
        return None
    hit = np.nonzero(vals <= eps)[0]
    k = max(1, int(round(window * n)))
    return dict(final=float(vals[-k:].mean()), auc=float(vals.mean()),
                first_hit=float(steps[hit[0]]) if len(hit) else float("nan"))


def reward_stats(vals, window):
    k = max(1, int(round(window * len(vals))))
    return dict(final=float(vals[-k:].mean()), auc=float(vals.mean()))


# ----------------------------------------------------------------------------- aggregation helpers
def boot_ci(x, n_boot, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt_ci(x, n_boot, min_seeds, nd=2):
    x = np.asarray([v for v in x if not np.isnan(v)], dtype=float)
    if len(x) == 0:
        return "n/a"
    if len(x) >= min_seeds:
        lo, hi = boot_ci(x, n_boot)
        return f"{x.mean():.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    return f"{x.mean():.{nd}f} (range {x.min():.{nd}f} to {x.max():.{nd}f}; n={len(x)})*"


def fmt_med(x, scale=1.0, nd=2):
    x = np.asarray(x, dtype=float) / scale
    if len(x) == 0:
        return "n/a"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{nd}f} [{q1:.{nd}f}, {q3:.{nd}f}]"


def sign_test_p(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = min(sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def write_table(base, header, rows, title, bold=None, notes=None):
    base.parent.mkdir(parents=True, exist_ok=True)
    with open(str(base) + ".csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    md = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if notes:
        md += ["", *notes]
    Path(str(base) + ".md").write_text("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{l" + "r" * (len(header) - 1) + "}", "\\toprule",
           " & ".join(h.replace("%", "\\%").replace("_", "\\_") for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append(" & ".join(str(c).replace("%", "\\%").replace("_", "\\_").replace("*", "$^*$") for c in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path(str(base) + ".tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


# ----------------------------------------------------------------------------- one experiment
def run_experiment(exp, cfg, out_dir, report, manifest):
    d = cfg["defaults"]; names = {**DEFAULT_NAMES, **cfg.get("names", {}), **exp.get("names", {})}
    db = Path(exp["db"])
    if not db.exists():
        print(f"[skip] {exp['name']}: database not found ({db})"); return
    conn = sqlite3.connect(db)
    algos = exp.get("algos") or [r[0] for r in conn.execute("SELECT DISTINCT algo_name FROM runs WHERE scenario=?", (exp["scenario"],))]
    ref = exp.get("ref", algos[0])
    min_seeds, n_boot = d["min_seeds"], d["n_boot"]
    runs = {a: dict(conn.execute("SELECT seed, run_id FROM runs WHERE scenario=? AND algo_name=? ORDER BY seed", (exp["scenario"], a)).fetchall()) for a in algos}
    runs = {a: r for a, r in runs.items() if r}
    manifest[exp["name"]] = dict(db=str(db), seeds={a: sorted(r) for a, r in runs.items()},
                                 db_mtime=datetime.fromtimestamp(db.stat().st_mtime).isoformat())
    report.append(f"\n## {exp.get('title', exp['name'])}\n")
    ed = out_dir / exp["name"]

    for view in exp.get("views", []):
        kind = view.get("kind", "phase"); label = view["label"]
        eps = view.get("eps", d["eps"]); lower = view.get("lower_is_better", True)
        per = {a: {} for a in runs}
        for a, rr in runs.items():
            for seed, rid in rr.items():
                steps, vals = view_series(conn, rid, view)
                if len(vals) == 0:
                    continue
                if kind == "phase":
                    _, phase = series(conn, rid, exp.get("phase_key", "env/phase"))
                    st = phase_stats(steps, vals, phase, eps, d["skip_first"])
                elif kind == "stationary":
                    st = stationary_stats(steps, vals, eps, d["final_window"])
                else:
                    st = reward_stats(vals, d["final_window"])
                if st: per[a][seed] = st
        per = {a: v for a, v in per.items() if v}
        if not per:
            report.append(f"_{label}: no data_\n"); continue
        sname = f"{label.lower().replace(' ', '_').replace('/', '_')}"

        # ---- per-seed csv
        keys = [k for k in next(iter(next(iter(per.values())).values())) if not k.endswith("_list")]
        ed.mkdir(parents=True, exist_ok=True)
        with open(ed / f"per_seed_{sname}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["algo", "seed", *keys])
            for a, v in per.items():
                for seed, st in sorted(v.items()): w.writerow([names.get(a, a), seed, *[st[k] for k in keys]])

        # ---- summary table
        if kind == "phase":
            header = ["Method", "n", f"Mean {label}", "Time above eps (%)", "Recovery (M steps)", "Never recovered (%)",
                      "Settle (M steps)", "Never settled (%)", "Relapses/phase"]
            rows = []
            for a, v in per.items():
                S = list(v.values())
                rec = [x for s in S for x in s["rec_list"]]; rc = [x for s in S for x in s["rec_c_list"]]
                se = [x for s in S for x in s["settle_list"]]; sc = [x for s in S for x in s["settle_c_list"]]
                rows.append([names.get(a, a), len(S), fmt_ci([s["mean"] for s in S], n_boot, min_seeds),
                             f"{100 * np.mean([s['frac_above'] for s in S]):.1f}", fmt_med(rec, 1e6),
                             f"{100 * np.mean(rc):.1f}", fmt_med(se, 1e6), f"{100 * np.mean(sc):.1f}",
                             f"{np.mean([s['relapses'] for s in S]):.2f}"])
            notes = [f"eps = {eps} ({label}); first phase excluded; recovery/settle pooled over seed x phase, median [IQR]; "
                     f"'never' = share of phases in which the threshold was not reached / not held until the phase end; "
                     f"* = fewer than {min_seeds} seeds, range instead of CI."]
        elif kind == "stationary":
            header = ["Method", "n", f"Final {label}", f"Mean {label} over training", f"Steps to {label} <= {eps} (M)"]
            rows = [[names.get(a, a), len(v), fmt_ci([s["final"] for s in v.values()], n_boot, min_seeds, 3),
                     fmt_ci([s["auc"] for s in v.values()], n_boot, min_seeds, 3),
                     fmt_med([s["first_hit"] for s in v.values() if not np.isnan(s["first_hit"])], 1e6)] for a, v in per.items()]
            notes = [f"final = mean of the last {int(100 * d['final_window'])}% of iterations."]
        else:
            header = ["Method", "n", f"Final {label}", f"Mean {label} over training"]
            rows = [[names.get(a, a), len(v), fmt_ci([s["final"] for s in v.values()], n_boot, min_seeds),
                     fmt_ci([s["auc"] for s in v.values()], n_boot, min_seeds)] for a, v in per.items()]
            notes = [f"final = mean of the last {int(100 * d['final_window'])}% of iterations; higher is better."]
        report.append(write_table(ed / f"summary_{sname}", header, rows, f"{exp.get('title', exp['name'])} - {label}", notes=notes) + "\n")

        # ---- paired table vs reference
        if ref in per and len(per) > 1:
            cols = ([("mean", f"Diff. mean {label}", 3), ("frac_above", "Diff. time above eps (pp)", 3), ("settle_mean", "Diff. settle (M steps)", 3)]
                    if kind == "phase" else [("final", f"Diff. final {label}", 3), ("auc", f"Diff. mean {label}", 3)])
            header = ["Method vs " + names.get(ref, ref), "n pairs"] + [c[1] for c in cols] + [f"{names.get(ref, ref)} better (wins/n)", "sign-test p"]
            rows = []
            for a, v in per.items():
                if a == ref: continue
                seeds = sorted(set(v) & set(per[ref]))
                if not seeds: continue
                cells = []
                for key, _, nd in cols:
                    scale = 1e6 if key == "settle_mean" else (0.01 if key == "frac_above" else 1.0)
                    diff = [(v[s][key] - per[ref][s][key]) / scale for s in seeds]      # algo - ref
                    cells.append(fmt_ci(diff, n_boot, min_seeds, nd))
                main = [v[s][cols[0][0]] - per[ref][s][cols[0][0]] for s in seeds]
                wins = sum(1 for x in main if (x > 0 if lower else x < 0))
                rows.append([names.get(a, a), len(seeds), *cells, f"{wins}/{len(seeds)}", f"{sign_test_p(main):.3f}"])
            if rows:
                report.append(write_table(ed / f"paired_{sname}", header, rows, f"Paired vs {names.get(ref, ref)} - {label}",
                    notes=["difference = method - reference on the same seeds (same phase schedule); "
                           + ("lower is better; " if lower else "higher is better; ") + "p: exact two-sided sign test."]) + "\n")

    # ---- diagnostics (theory-code link)
    drows = []
    for a, rr in runs.items():
        agg = {}
        for seed, rid in rr.items():
            for m in ["diag/clamp_frac", "diag/step_cond_max", "diag/eps_c", "diag/eps_a", "diag/critic_range", "diag/lambda", "diag/epsilon"]:
                _, v = series(conn, rid, m)
                if len(v): agg.setdefault(m, []).append(v)
        if not agg: continue
        def share(m, thr): return np.mean([np.mean(x > thr) for x in agg[m]]) if m in agg else float("nan")
        def mean_of(m, f): return np.mean([f(x) for x in agg[m]]) if m in agg else float("nan")
        def f(x, nd): return "-" if (x is None or np.isnan(x)) else f"{x:.{nd}f}"
        drows.append([names.get(a, a), len(rr), f(100 * share('diag/clamp_frac', 0), 1), f(mean_of('diag/clamp_frac', np.mean), 3),
                      f(100 * share('diag/step_cond_max', 1), 1), f(mean_of('diag/step_cond_max', np.max), 2),
                      f(mean_of('diag/eps_c', np.mean), 3), f(mean_of('diag/eps_a', np.mean), 3),
                      f(mean_of('diag/critic_range', np.max), 2), f(mean_of('diag/lambda', np.max), 2),
                      f(mean_of('diag/epsilon', lambda x: x[-1]), 3)])
    if drows:
        report.append(write_table(ed / "diagnostics",
            ["Method", "n", "Iters clamp>0 (%)", "Mean clamp frac", "Iters cond>1 (%)", "Max cond", "eps_c", "eps_a", "B (max)", "Lambda (max)", "final epsilon"],
            drows, "Theory-code diagnostics",
            notes=["clamp/cond: how often the update differs from the raw Smith update (condition alpha(M-1)B<=1); "
                   "eps_c, eps_a, B, Lambda: measured assumption quantities (ring game); epsilon: QMIX exploration rate at the end."]) + "\n")
    conn.close()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", default="paper_out")
    ap.add_argument("--plots", action="store_true", help="also call scripts/plot.py for every experiment")
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiment names")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["defaults"] = {"eps": 0.5, "skip_first": True, "min_seeds": 5, "n_boot": 5000, "final_window": 0.1, **cfg.get("defaults", {})}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report, manifest = [f"# Paper report - generated {datetime.now():%Y-%m-%d %H:%M}\n"], {}
    here = Path(__file__).resolve().parent

    for exp in cfg.get("experiments", []):
        if a.only and exp["name"] not in a.only: continue
        run_experiment(exp, cfg, out, report, manifest)
        if a.plots and Path(exp["db"]).exists():
            subprocess.run([sys.executable, str(here / "plot.py"), "--scenario", exp["scenario"], "--db", exp["db"],
                            "--out_dir", str(out / "figs" / exp["name"]), "--no_title", "--algos", *exp.get("algos", [])], check=False)

    for sw in cfg.get("sweeps", []):
        if a.only and sw["name"] not in a.only: continue
        cmd = [sys.executable, str(here / "analyze_sweep.py"), "--glob", sw["glob"], "--param", sw["param"], "--name", sw["name"],
               "--out_dir", str(out / "sweeps"), "--x_transform", sw.get("x_transform", "none")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        report.append(f"\n## Sweep: {sw['name']} ({sw['param']})\n\n```\n{r.stdout.strip() or r.stderr.strip()}\n```\n")
    for bd in cfg.get("bound", []):
        if a.only and bd["name"] not in a.only: continue
        r = subprocess.run([sys.executable, str(here / "check_bound.py"), "--db", bd["db"], "--algo", bd.get("algo", "deep_ed")], capture_output=True, text=True)
        txt = r.stdout.strip() or r.stderr.strip()
        (out / f"bound_{bd['name']}.txt").write_text(txt + "\n")
        report.append(f"\n## Theorem-1 bound check: {bd['name']}\n\n```\n{txt}\n```\n")

    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    (out / "manifest.json").write_text(json.dumps(dict(config=cfg, runs=manifest, generated=datetime.now().isoformat()), indent=2, default=str))
    print(f"wrote {out/'REPORT.md'}  (+ per-experiment tables, manifest.json)")


if __name__ == "__main__":
    main()
