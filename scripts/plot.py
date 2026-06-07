import argparse
import os
import sqlite3
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

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
    "biased_rps":      ["Rock", "Paper", "Scissors"],
    "stag_hunt":       ["Stag", "Hare"],
    "battle_of_sexes": ["Football", "Opera"],
}

# Known Nash equilibria per scenario: list of (agent0_probs, agent1_probs)
# Each is a probability vector over actions.
NASH_EQUILIBRIA: dict[str, list[tuple[list[float], list[float]]]] = {
    # biased_rps intentionally omitted: NE = (v/(v+2), 1/(v+2), 1/(v+2)) depends
    # on the bias parameter v which changes across phases, so no fixed point exists.
    "stag_hunt":       [([1.0, 0.0], [1.0, 0.0]),
                        ([0.0, 1.0], [0.0, 1.0])],
    "battle_of_sexes": [
        ([1.0, 0.0], [1.0, 0.0]),          # pure NE: both Football
        ([0.0, 1.0], [0.0, 1.0]),          # pure NE: both Opera
        ([2/3, 1/3], [1/3, 2/3]),          # mixed NE (standard 2-1/1-2 payoffs)
    ],
}

#───── DB queries ───────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return sqlite3.connect(db_path)


def discover_algos(conn: sqlite3.Connection, scenario: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT algo_name FROM runs WHERE scenario=?", (scenario,)
    ).fetchall()
    return [r[0] for r in rows]


def load_runs_by_algo(
    conn: sqlite3.Connection,
    scenario: str,
    algos: list[str],
    seeds: list[int] | None = None,
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for algo in algos:
        conditions = ["scenario=?", "algo_name=?"]
        params: list = [scenario, algo]
        if seeds:
            placeholders = ",".join("?" * len(seeds))
            conditions.append(f"seed IN ({placeholders})")
            params.extend(seeds)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT seed, run_id FROM runs WHERE {where}", params
        ).fetchall()
        if rows:
            result[algo] = {seed: run_id for seed, run_id in rows}
        else:
            print(f"  ⚠  No runs found for algo='{algo}' — skipping.")
    if not result:
        available = [r[0] for r in conn.execute(
            "SELECT DISTINCT scenario FROM runs").fetchall()]
        raise ValueError(
            f"No runs for scenario='{scenario}' algos={algos} seeds={seeds}.\n"
            f"Available scenarios: {available}"
        )
    return result


def load_metric(
    conn: sqlite3.Connection, run_id: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
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
    conn: sqlite3.Connection, run_id: int
) -> tuple[np.ndarray, np.ndarray]:
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

def load_avg_policy_snapshots(
    conn: sqlite3.Connection, run_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as load_policy_snapshots but returns the cumulative time-average
    policy pi_bar(t) = (1/t) * sum_{s<=t} pi(s).

    This matches what nash_conv is computed against (avg_pi), so the
    simplex trajectories will be consistent with the Nash conv plot.
    """
    steps, probs = load_policy_snapshots(conn, run_id)
    if len(steps) == 0:
        return steps, probs
    # probs: (T, n_agents, n_actions)
    cum = np.cumsum(probs, axis=0)
    counts = np.arange(1, len(steps) + 1)[:, None, None]
    avg_probs = cum / counts
    return steps, avg_probs


#───── interpolation ────────────────────────────────────────────────────────────

def interpolate_to_grid(
    steps: np.ndarray, values: np.ndarray, grid: np.ndarray
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

#───── plotting helpers ─────────────────────────────────────────────────────────

def shade(ax, x, mean, std, color, label, alpha=0.18, lw=2.0):
    ax.plot(x, mean, color=color, linewidth=lw, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha, linewidth=0)


def draw_phase_lines(ax, conn, run_ids):
    change_steps: set[float] = set()
    for run_id in run_ids.values():
        steps, values = load_metric(conn, run_id, "env/phase")
        if len(steps) == 0:
            continue
        prev = values[0]
        for s, v in zip(steps[1:], values[1:]):
            if v != prev:
                change_steps.add(float(s))
                prev = v
    for s in sorted(change_steps):
        ax.axvline(s, linestyle="--", color="#E63946", alpha=0.55, linewidth=1)


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

#───── simplex geometry ─────────────────────────────────────────────────────────

# Equilateral triangle vertices for 3-action simplex (in 2-D Cartesian)
_TRI_VERTICES = np.array([
    [0.0,        0.0       ],   # action 0  (bottom-left)
    [1.0,        0.0       ],   # action 1  (bottom-right)
    [0.5,  np.sqrt(3) / 2 ],   # action 2  (top)
])

def barycentric_to_cartesian(probs: np.ndarray) -> np.ndarray:
    """
    probs : (..., 3)  – rows of probability vectors
    returns (..., 2)  – Cartesian coordinates inside the equilateral triangle
    """
    return probs @ _TRI_VERTICES


def draw_simplex_triangle(ax: plt.Axes, action_labels: list[str]):
    """Draw the equilateral triangle border and label its vertices."""
    tri = np.vstack([_TRI_VERTICES, _TRI_VERTICES[0]])  # close the loop
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.5, zorder=2)

    offsets = [(-0.07, -0.06), (0.03, -0.06), (0.0, 0.04)]
    ha       = ["right",       "left",         "center"   ]
    for i, (label, (dx, dy), h) in enumerate(
            zip(action_labels, offsets, ha)):
        x, y = _TRI_VERTICES[i]
        ax.text(x + dx, y + dy, label,
                ha=h, va="center", fontsize=11, fontweight="bold")

    ax.set_aspect("equal")
    ax.axis("off")


def draw_simplex_segment(ax: plt.Axes, action_labels: list[str]):
    """Draw the [0,1] line segment for a 2-action simplex."""
    ax.axhline(0, color="black", lw=1.0, zorder=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"P({action_labels[0]})", fontsize=11)
    ax.set_ylabel(f"P({action_labels[0]})", fontsize=11)  # will be overridden per agent
    ax.set_aspect("equal")


def _add_colorbar(fig, ax, cmap, label="Time →"):
    """Attach a slim colorbar showing trajectory time."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Start", "End"])
    cb.set_label(label, fontsize=9)

#───── core simplex trajectory logic ────────────────────────────────────────────

def _collect_trajectories(
    conn: sqlite3.Connection,
    run_ids: dict[int, int],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns (grid, interp_all) where
      grid       : (T,)
      interp_all : (n_seeds, T, n_agents, n_actions)
    or None if no data.
    """
    steps_list, probs_list = [], []
    for run_id in run_ids.values():
        steps, probs = load_policy_snapshots(conn, run_id)
        if len(steps):
            steps_list.append(steps)
            probs_list.append(probs)
    if not probs_list:
        return None

    n_agents  = probs_list[0].shape[1]
    n_actions = probs_list[0].shape[2]
    flat      = np.concatenate(steps_list)
    grid      = np.linspace(flat.min(), flat.max(), n_points)

    interp_all = np.stack([
        np.stack([
            np.stack([
                interpolate_to_grid(steps, probs[:, ag, ac], grid)
                for ac in range(n_actions)
            ], axis=-1)                       # (T, n_actions)
            for ag in range(n_agents)
        ], axis=-2)                           # (T, n_agents, n_actions)
        for steps, probs in zip(steps_list, probs_list)
    ])                                        # (seeds, T, n_agents, n_actions)

    return grid, interp_all


def _draw_trajectory_2d(
    ax: plt.Axes,
    xy_seeds: np.ndarray,    # (n_seeds, T, 2)  Cartesian coords
    color: str,
    cmap_name: str = "Blues",
    seed_alpha: float = 0.18,
    mean_lw: float = 2.5,
    n_arrows: int = 6,
):
    """
    Draw individual seed trails (faint) + mean trajectory (bold, coloured by time).
    Arrows indicate direction of travel along the mean.
    """
    T = xy_seeds.shape[1]
    cmap = matplotlib.colormaps[cmap_name]

    # ── faint individual seeds ──────────────────────────────────────────────
    for xy in xy_seeds:
        ax.plot(xy[:, 0], xy[:, 1],
                color=color, alpha=seed_alpha, lw=0.8, zorder=3)

    # ── mean trajectory coloured by time ───────────────────────────────────
    mean_xy = xy_seeds.mean(axis=0)  # (T, 2)
    t_norm  = np.linspace(0, 1, T)

    for i in range(T - 1):
        c = cmap(0.35 + 0.65 * t_norm[i])   # avoid very pale start
        ax.plot(mean_xy[i:i+2, 0], mean_xy[i:i+2, 1],
                color=c, lw=mean_lw, zorder=4, solid_capstyle="round")

    # ── direction arrows along mean ─────────────────────────────────────────
    arrow_indices = np.linspace(T // 10, T - T // 10, n_arrows, dtype=int)
    for idx in arrow_indices:
        if idx + 1 >= T:
            continue
        dx = mean_xy[idx + 1, 0] - mean_xy[idx, 0]
        dy = mean_xy[idx + 1, 1] - mean_xy[idx, 1]
        if np.hypot(dx, dy) < 1e-6:
            continue
        c = cmap(0.35 + 0.65 * t_norm[idx])
        ax.annotate(
            "", xy=(mean_xy[idx, 0] + dx, mean_xy[idx, 1] + dy),
            xytext=(mean_xy[idx, 0], mean_xy[idx, 1]),
            arrowprops=dict(arrowstyle="-|>", color=c,
                            lw=1.6, mutation_scale=12),
            zorder=5,
        )

    # ── start / end markers ─────────────────────────────────────────────────
    ax.scatter(*mean_xy[0],  s=70,  color=cmap(0.4),  marker="o",
               zorder=6, edgecolors="white", linewidths=0.8)
    ax.scatter(*mean_xy[-1], s=120, color=cmap(0.95), marker="*",
               zorder=6, edgecolors="white", linewidths=0.8)


def _mark_nash_2action(ax: plt.Axes, nash_list, agent_idx: int):
    """Mark Nash equilibria as grey crosses on a 2-action (x-y) plot."""
    for ne in nash_list:
        x = ne[0][0]   # agent-0 prob of action-0
        y = ne[1][0]   # agent-1 prob of action-0
        pt = (x, y) if agent_idx == 0 else (x, y)   # same coords; kept for clarity
        ax.scatter(*pt, s=120, marker="x", color="gray",
                   linewidths=2, zorder=7, label="Nash eq.")


def _mark_nash_3action(ax: plt.Axes, nash_list, agent_idx: int):
    """Mark Nash equilibria as grey crosses on the 3-action simplex."""
    for ne in nash_list:
        probs = np.array(ne[agent_idx], dtype=float)
        xy = barycentric_to_cartesian(probs)
        ax.scatter(*xy, s=120, marker="x", color="gray",
                   linewidths=2, zorder=7, label="Nash eq.")

#───── per-algo simplex figure ──────────────────────────────────────────────────

# One colour-map per algo so multiple algos on the same axes stay distinct.
_ALGO_CMAPS = ["Blues", "Oranges", "Greens", "RdPu", "YlOrBr", "PuBu"]


def plot_simplex_trajectories(
    conn: sqlite3.Connection,
    runs_by_algo: dict[str, dict[int, int]],
    scenario: str,
    out_dir: str,
    n_points: int = 200,
):
    """
    For each algorithm: save a PDF with one subplot per agent showing the
    policy trajectory on the probability simplex.
    - 2 actions → 2-D unit-square phase portrait (x = p_A0 agent-0,
                                                   y = p_A0 agent-1)
    - 3 actions → equilateral-triangle simplex
    """
    # ── detect shape from first available snapshot ───────────────────────
    n_agents = n_actions = None
    for run_ids in runs_by_algo.values():
        for run_id in run_ids.values():
            steps, probs = load_policy_snapshots(conn, run_id)
            if len(steps):
                n_agents, n_actions = probs.shape[1], probs.shape[2]
                break
        if n_agents is not None:
            break

    if n_agents is None:
        print("  ⚠  No policy snapshots — skipping simplex plot.")
        return

    if n_actions not in (2, 3):
        print(f"  ⚠  Simplex plot supports 2 or 3 actions only "
              f"(found {n_actions}) — skipping.")
        return

    action_labels = GAME_ACTION_LABELS.get(
        scenario, [f"A{i}" for i in range(n_actions)]
    )
    nash_list = NASH_EQUILIBRIA.get(scenario, [])
    title_prefix = scenario.replace("_", " ").title()

    # ── one file per algo ────────────────────────────────────────────────
    for algo_idx, (algo, run_ids) in enumerate(runs_by_algo.items()):
        cmap_name = _ALGO_CMAPS[algo_idx % len(_ALGO_CMAPS)]
        color     = PALETTE[algo_idx % len(PALETTE)]

        result = _collect_trajectories(conn, run_ids, n_points)
        if result is None:
            print(f"  ⚠  No snapshots for algo='{algo}' — skipping simplex.")
            continue
        _, interp_all = result   # (seeds, T, n_agents, n_actions)

        fig, axes = plt.subplots(1, n_agents,
                                 figsize=(4.5 * n_agents, 4.2),
                                 squeeze=False)
        axes = axes[0]

        for ag, ax in enumerate(axes):
            if n_actions == 3:
                # ── triangle simplex ─────────────────────────────────────
                draw_simplex_triangle(ax, action_labels)

                # (seeds, T, 3) → (seeds, T, 2)
                xy_seeds = barycentric_to_cartesian(interp_all[:, :, ag, :])
                _draw_trajectory_2d(ax, xy_seeds, color, cmap_name)
                if nash_list:
                    _mark_nash_3action(ax, nash_list, ag)

                _add_colorbar(fig, ax, matplotlib.colormaps[cmap_name])
                ax.set_title(f"Agent {ag}", pad=8)

            else:
                # ── 2-action: joint phase-portrait in [0,1]² ────────────
                # x-axis = agent-0 prob(action-0)
                # y-axis = agent-1 prob(action-0)
                # Both panels are identical for 2-action games so we only
                # draw on the first subplot and hide the second.
                if ag == 0:
                    p_ag0 = interp_all[:, :, 0, 0]  # (seeds, T)
                    p_ag1 = interp_all[:, :, 1, 0]  # (seeds, T)
                    xy_seeds = np.stack([p_ag0, p_ag1], axis=-1)  # (seeds, T, 2)

                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_aspect("equal")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_xlabel(f"Agent 0  P({action_labels[0]})", fontsize=11)
                    ax.set_ylabel(f"Agent 1  P({action_labels[0]})", fontsize=11)
                    ax.set_title("Joint Policy Trajectory", pad=8)

                    _draw_trajectory_2d(ax, xy_seeds, color, cmap_name)
                    if nash_list:
                        _mark_nash_2action(ax, nash_list, ag)

                    _add_colorbar(fig, ax, matplotlib.colormaps[cmap_name])
                else:
                    # Hide the redundant second panel for 2-action games
                    ax.set_visible(False)

        # ── Nash legend entry (deduplicated) ────────────────────────────
        if nash_list:
            nash_handle = Line2D([0], [0], marker="x", color="gray",
                                 linewidth=0, markersize=8, markeredgewidth=2,
                                 label="Nash eq.")
            for ax in axes:
                ax.legend(handles=[nash_handle], loc="upper right",
                          framealpha=0.7, fontsize=9)

        n_seeds = len(run_ids)
        fig.suptitle(
            f"Simplex Trajectories — {title_prefix}   "
            f"{algo}  (n={n_seeds} seeds)",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        savefig(fig, out_dir, f"simplex_{algo}.pdf")

#───── scalar + policy-prob plots (unchanged API) ───────────────────────────────

def plot_scalar_metric(
    conn, runs_by_algo, metric_key, scenario,
    out_dir, filename, ylabel, title,
    hline=None, hline_label="", n_points=200,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    any_data = False
    all_run_ids: dict[int, int] = {}
    for run_ids in runs_by_algo.values():
        all_run_ids.update(run_ids)

    for idx, (algo, run_ids) in enumerate(runs_by_algo.items()):
        color = PALETTE[idx % len(PALETTE)]
        grid  = build_common_grid(conn, run_ids, metric_key, n_points)
        if len(grid) == 0:
            print(f"  ⚠  No data for '{metric_key}' / algo='{algo}' — skipping.")
            continue
        interp = np.stack([
            interpolate_to_grid(*load_metric(conn, rid, metric_key), grid)
            for rid in run_ids.values()
        ])
        mean, std = interp.mean(0), interp.std(0)
        shade(ax, grid, mean, std, color, label=f"{algo} (n={len(run_ids)} seeds)")
        any_data = True

    if not any_data:
        print(f"  ⚠  No data for '{metric_key}' — skipping.")
        plt.close(fig)
        return

    if hline is not None:
        ax.axhline(hline, ls="--", color="gray", lw=1, alpha=0.6, label=hline_label)

    draw_phase_lines(ax, conn, all_run_ids)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_steps))
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir, filename)


def plot_policy_probs(
    conn, runs_by_algo, scenario, out_dir, n_points=200,
):
    LINESTYLES = ["-", "--", "-.", ":"]
    n_agents = n_actions = None
    for run_ids in runs_by_algo.values():
        for run_id in run_ids.values():
            steps, probs = load_policy_snapshots(conn, run_id)
            if len(steps):
                n_agents, n_actions = probs.shape[1], probs.shape[2]
                break
        if n_agents is not None:
            break

    if n_agents is None:
        print("  ⚠  No policy snapshots — skipping policy_probs plot.")
        return

    action_labels = GAME_ACTION_LABELS.get(
        scenario, [f"Action {a}" for a in range(n_actions)]
    )
    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 4), sharey=True)
    if n_agents == 1:
        axes = [axes]

    all_run_ids: dict[int, int] = {}
    for run_ids in runs_by_algo.values():
        all_run_ids.update(run_ids)

    for algo_idx, (algo, run_ids) in enumerate(runs_by_algo.items()):
        ls = LINESTYLES[algo_idx % len(LINESTYLES)]
        steps_list, probs_list = [], []
        for run_id in run_ids.values():
            steps, probs = load_policy_snapshots(conn, run_id)
            if len(steps):
                steps_list.append(steps)
                probs_list.append(probs)
        if not probs_list:
            continue

        flat = np.concatenate(steps_list)
        grid = np.linspace(flat.min(), flat.max(), n_points)
        interp_all = np.stack([
            np.stack([
                np.stack([
                    interpolate_to_grid(steps, probs[:, ag, ac], grid)
                    for ac in range(n_actions)
                ], axis=-1)
                for ag in range(n_agents)
            ], axis=-2)
            for steps, probs in zip(steps_list, probs_list)
        ])
        mean_pol = interp_all.mean(0)
        std_pol  = interp_all.std(0)

        for ag, ax in enumerate(axes):
            for ac in range(n_actions):
                color = PALETTE[ac % len(PALETTE)]
                action_name = action_labels[ac] if ac < len(action_labels) else f"A{ac}"
                label = f"{algo} – {action_name}" if len(runs_by_algo) > 1 else action_name
                ax.plot(grid, mean_pol[:, ag, ac],
                        color=color, linestyle=ls, linewidth=2.0, label=label)
                ax.fill_between(
                    grid,
                    mean_pol[:, ag, ac] - std_pol[:, ag, ac],
                    mean_pol[:, ag, ac] + std_pol[:, ag, ac],
                    color=color, alpha=0.18, linewidth=0,
                )

    for ag, ax in enumerate(axes):
        draw_phase_lines(ax, conn, all_run_ids)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Action Probability" if ag == 0 else "")
        ax.set_title(f"Agent {ag}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_steps))
        ax.legend(loc="upper right")

    n_seeds_str = ", ".join(f"{a}: {len(r)}" for a, r in runs_by_algo.items())
    fig.suptitle(
        f"Policy Probabilities — {scenario.replace('_', ' ').title()}"
        f"  ({n_seeds_str} seeds)",
        y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "policy_probs.pdf")

#───── cli ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot multi-seed / multi-algo MARL results."
    )
    p.add_argument("--scenario", required=True)
    p.add_argument("--algos", nargs="*", default=None)
    p.add_argument("--algo",  default=None, help=argparse.SUPPRESS)
    p.add_argument("--db",    default="outputs/matrix_games/results.db")
    p.add_argument("--out_dir", default="outputs/matrix_games/plots")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--n_points", type=int, default=200)
    return p.parse_args()


def main():
    args    = parse_args()
    scenario = args.scenario
    out_dir  = os.path.join(args.out_dir, scenario)
    conn     = _connect(args.db)

    if args.algos:
        algos = args.algos
    elif args.algo:
        algos = [args.algo]
    else:
        algos = discover_algos(conn, scenario)
        if not algos:
            raise ValueError(f"No runs for scenario='{scenario}' in {args.db}")

    print(f"\n[plot] scenario : {scenario}")
    print(f"[plot] algos    : {algos}")
    print(f"[plot] db       : {args.db}")
    print(f"[plot] out_dir  : {out_dir}")
    print(f"[plot] seeds    : {'all' if args.seeds is None else args.seeds}\n")

    runs_by_algo = load_runs_by_algo(conn, scenario, algos, seeds=args.seeds)
    for algo, run_ids in runs_by_algo.items():
        print(f"  → {algo}: {len(run_ids)} run(s), seeds {sorted(run_ids.keys())}")
    print()
    print("[plot] Generating figures …")

    title_prefix = scenario.replace("_", " ").title()

    plot_scalar_metric(
        conn, runs_by_algo,
        metric_key="nash/nash_conv",
        scenario=scenario, out_dir=out_dir,
        filename="nash_conv.pdf",
        ylabel="Nash Convergence",
        title=f"Nash Convergence — {title_prefix}",
        hline=0.0, hline_label="Nash equilibrium",
        n_points=args.n_points,
    )
    plot_scalar_metric(
        conn, runs_by_algo,
        metric_key="reward/mean_episode_reward",
        scenario=scenario, out_dir=out_dir,
        filename="reward.pdf",
        ylabel="Mean Episode Reward",
        title=f"Episode Reward — {title_prefix}",
        n_points=args.n_points,
    )
    if scenario == "biased_rps":
        plot_simplex_trajectories(conn, runs_by_algo, scenario, out_dir,
                                   n_points=args.n_points)

    conn.close()
    print("\n[plot] Done.")


if __name__ == "__main__":
    main()