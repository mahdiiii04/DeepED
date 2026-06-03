import argparse
import os
import sqlite3
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
})

PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky-blue
    "#E69F00",  # amber
]

GAME_ACTION_LABELS: dict[str, list[str]] = {
    "biased_rps":          ["Rock", "Paper", "Scissors"],
    "stag_hunt":           ["Stag", "Hare"],
    "battle_of_sexes":     ["Football", "Opera"],
}

#───── DB queries ───────────────────────────────────
def _connect(db_path: str) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}"
        )
    return sqlite3.connect(db_path)

def load_run_ids(
    conn: sqlite3.Connection,
    scenario: str,
    algo: str | None = None,
    seeds: list[int] | None = None
) -> dict[int, int]:
    
    conditions = ["scenario=?"]
    params: list = [scenario]

    if algo:
        conditions.append("algo_name=?")
        params.append(algo)

    if seeds:
        placeholders = ",".join("?" * len(seeds))
        conditions.append(f"seed IN ({placeholders})")
        params.extend(seeds)

    where = " AND ".join(conditions)
    rows = conn.execute(
        f"SELECT seed, run_id FROM runs WHERE {where}",
        params,
    ).fetchall()

    if not rows:
        available = [
            r[0]
            for r in conn.execute("SELECT DISTINCT scenario FROM runs").fetchall()
        ]
        raise ValueError(
            f"No runs found for scenario='{scenario}' algo='{algo}' seeds={seeds}.\n"
            f"Available scenarios: {available}"
        )
    return {seed: run_id for seed, run_id in rows}

def load_metric(
    conn: sqlite3.Connection,
    run_id: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for a single metric / run."""
    rows = conn.execute(
        "SELECT global_step, value FROM metrics "
        "WHERE run_id=? AND metric=? ORDER BY global_step",
        (run_id, metric),
    ).fetchall()
    if not rows:
        return np.array([]), np.array([])
    steps, values = zip(*rows)
    return np.array(steps, dtype=float), np.array(values, dtype=float)

def load_policy_snapshots(
    conn: sqlite3.Connection,
    run_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (steps, probs) where probs has shape (T, n_agents, n_actions).
    """
    rows = conn.execute(
        """
        SELECT global_step, agent_idx, probs_json
        FROM   policy_snapshots
        WHERE  run_id=?
        ORDER  BY global_step, agent_idx
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return np.array([]), np.array([])

    steps_set = sorted({r[0] for r in rows})
    agent_idxs = sorted({r[1] for r in rows})
    n_agents = len(agent_idxs)
    n_actions = len(json.loads(rows[0][2]))

    probs = np.zeros((len(steps_set), n_agents, n_actions))
    step_to_idx = {s: i for i, s in enumerate(steps_set)}

    for step, ag, probs_json in rows:
        probs[step_to_idx[step], ag] = json.loads(probs_json)

    return np.array(steps_set, dtype=float), probs

#───── interpolation ────────────────────────────────────────

def interpolate_to_grid(
    steps: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    if len(steps) == 0:
        return np.full(len(grid), np.nan)
    return np.interp(grid, steps, values)

def build_common_grid(
    conn: sqlite3.Connection,
    run_ids: dict[int, int],
    metric: str,
    n_points: int = 200,
) -> np.ndarray:
    all_steps = []
    for run_id in run_ids.values():
        steps, _ = load_metric(conn, run_id, metric)
        if len(steps):
            all_steps.append(steps)
    if not all_steps:
        return np.array([])
    combined = np.concatenate(all_steps)
    return np.linspace(combined.min(), combined.max(), n_points)


#───── plotting helpers ──────────────────────────────────────────────────────────

def shade(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    label: str,
    alpha: float = 0.18,
    lw: float = 2.0,
):
    ax.plot(x, mean, color=color, linewidth=lw, label=label)
    ax.fill_between(
        x, mean - std, mean + std, color=color, alpha=alpha, linewidth=0
    )

def fmt_steps(x: float, _) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return str(int(x))

def savefig(fig: plt.Figure, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    print(f"  Saved → {path}")
    plt.close(fig)

#───── plots ─────────────────────────────────────────────

def plot_scalar_metric(
    conn: sqlite3.Connection,
    run_ids: dict[int, int],
    metric_key: str,
    scenario: str,
    out_dir: str,
    filename: str,
    ylabel: str,
    title: str,
    hline: float | None = None,
    hline_label: str = "",
    n_points: int = 200,
):
    grid = build_common_grid(conn, run_ids, metric_key, n_points)
    if len(grid) == 0:
        print(f"  ⚠  No data for '{metric_key}' — skipping.")
        return

    interp = np.stack([
        interpolate_to_grid(*load_metric(conn, rid, metric_key), grid)
        for rid in run_ids.values()
    ])
    mean, std = interp.mean(0), interp.std(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    shade(ax, grid, mean, std, PALETTE[0], label=f"MAPPO (n={len(run_ids)} seeds)")

    if hline is not None:
        ax.axhline(hline, ls="--", color="gray", lw=1, alpha=0.6, label=hline_label)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_steps))
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir, filename)

def plot_policy_probs(
    conn: sqlite3.Connection,
    run_ids: dict[int, int],
    scenario: str,
    out_dir: str,
    n_points: int = 200,
):
    # Collect per-seed policy trajectories
    all_steps_list, all_probs_list = [], []
    for run_id in run_ids.values():
        steps, probs = load_policy_snapshots(conn, run_id)
        if len(steps) == 0:
            continue
        all_steps_list.append(steps)
        all_probs_list.append(probs)

    if not all_probs_list:
        print("  ⚠  No policy snapshots found — skipping policy_probs plot.")
        return

    n_agents = all_probs_list[0].shape[1]
    n_actions = all_probs_list[0].shape[2]

    # Build shared grid from all steps
    all_steps_flat = np.concatenate(all_steps_list)
    grid = np.linspace(all_steps_flat.min(), all_steps_flat.max(), n_points)

    # Interpolate every seed onto the grid  →  (n_seeds, T, n_agents, n_actions)
    interp_all = np.stack([
        np.stack([
            np.stack([
                interpolate_to_grid(steps, probs[:, ag, ac], grid)
                for ac in range(n_actions)
            ], axis=-1)
            for ag in range(n_agents)
        ], axis=-2)
        for steps, probs in zip(all_steps_list, all_probs_list)
    ])  # (seeds, T, agents, actions)

    mean_pol = interp_all.mean(0)  # (T, agents, actions)
    std_pol = interp_all.std(0)

    action_labels = GAME_ACTION_LABELS.get(
        scenario, [f"Action {a}" for a in range(n_actions)]
    )

    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 4), sharey=True)
    if n_agents == 1:
        axes = [axes]

    for ag, ax in enumerate(axes):
        for ac in range(n_actions):
            shade(
                ax,
                grid,
                mean_pol[:, ag, ac],
                std_pol[:, ag, ac],
                color=PALETTE[ac % len(PALETTE)],
                label=action_labels[ac] if ac < len(action_labels) else f"A{ac}",
            )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Action Probability" if ag == 0 else "")
        ax.set_title(f"Agent {ag}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_steps))
        ax.legend(loc="upper right")

    fig.suptitle(
        f"Policy Probabilities — {scenario.replace('_', ' ').title()}"
        f"  (n={len(run_ids)} seeds)",
        y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "policy_probs.pdf")


#───── cli ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot multi-seed MARL results (reads from results.db)."
    )
    p.add_argument("--scenario", required=True, help="e.g. biased_rps")
    p.add_argument("--algo", default=None, help="Filter by algorithm (e.g. mappo)")
    p.add_argument(
        "--db", default="outputs/matrix_games/results.db",
        help="Path to SQLite DB",
    )
    p.add_argument("--out_dir", default="outputs/matrix_games/plots",
                   help="Root output folder")
    p.add_argument(
        "--seeds", nargs="*", type=int, default=None,
        help="Subset of seeds to include (default: all found)",
    )
    p.add_argument(
        "--n_points", type=int, default=200,
        help="Points on the shared interpolation grid",
    )
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = os.path.join(args.out_dir, args.scenario)
    scenario = args.scenario

    print(f"\n[plot] scenario : {scenario}")
    print(f"[plot] algo     : {args.algo or 'all'}")
    print(f"[plot] db       : {args.db}")
    print(f"[plot] out_dir  : {out_dir}")
    print(f"[plot] seeds    : {'all' if args.seeds is None else args.seeds}\n")

    conn = _connect(args.db)
    run_ids = load_run_ids(conn, scenario, algo=args.algo, seeds=args.seeds)

    print(f"  → {len(run_ids)} run(s) found: seeds {sorted(run_ids.keys())}\n")
    print("[plot] Generating figures …")

    title_prefix = scenario.replace("_", " ").title()

    plot_scalar_metric(
        conn,
        run_ids,
        metric_key="nash/nash_conv",
        scenario=scenario,
        out_dir=out_dir,
        filename="nash_conv.pdf",
        ylabel="Nash Convergence",
        title=f"Nash Convergence — {title_prefix}",
        hline=0.0,
        hline_label="Nash equilibrium",
        n_points=args.n_points,
    )

    plot_scalar_metric(
        conn,
        run_ids,
        metric_key="reward/mean_episode_reward",
        scenario=scenario,
        out_dir=out_dir,
        filename="reward.pdf",
        ylabel="Mean Episode Reward",
        title=f"Episode Reward — {title_prefix}",
        n_points=args.n_points,
    )

    plot_policy_probs(conn, run_ids, scenario, out_dir, n_points=args.n_points)

    conn.close()
    print("\n[plot] Done.")


if __name__ == "__main__":
    main()
