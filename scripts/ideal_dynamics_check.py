"""
Training-free validation of the theory on the ring potential game (pure numpy).

Simulates the IDEALISED DeepED loop used in the analysis: tabular policies, exact (optionally noisy)
critic, raw simultaneous Smith target, actor that reaches the target (or lags it), exogenous payoff
rotation every L steps. Because every quantity is exact, we can check

  * each lemma used in the proof of Theorem 1, step by step (any violation = an error in the proof):
      (a) potential shift under an environment change        <= 2 N C_R v_t / (1-g)^2
      (b) actor tracking error                               <= N Rmax eps_a / (1-g)^2
      (c) simultaneous-update cross term                     <= Xi_alpha
      (d) single-agent Smith improvement vs best-response gap  >= a(1-g)Phi^2/Lambda - 4 a M Rmax eps_c/(1-g)^2
  * the final bound of Theorem 1 against the measured (1/T) sum N_t^2;
  * the qualitative predictions: floor vs change rate L, step size alpha, critic error sigma, actor lag.

    python scripts/ideal_dynamics_check.py check
    python scripts/ideal_dynamics_check.py sweep --param alpha --values 0.02 0.05 0.1 0.2 0.4 --out sweep_alpha.pdf
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for name, rel in [("envs", "envs"), ("envs.gridworld", "envs/gridworld"), ("envs.ring", "envs/ring")]:
    if name not in sys.modules:                 # avoid importing torch-heavy package __init__ files
        m = types.ModuleType(name); m.__path__ = [str(ROOT / rel)]; sys.modules[name] = m
from envs.ring import ring_oracle as ro          # noqa: E402
from envs.gridworld import team_nash as tn       # noqa: E402


def smith_target(pi, Qh, alpha):
    """pi [S,2,M], Qh [S,2,M] -> raw simultaneous Smith target (no clamp) and the max critic range."""
    S, N, M = pi.shape
    tgt = np.empty_like(pi)
    for s in range(S):
        for i in range(N):
            q, p = Qh[s, i], pi[s, i]
            Rm = np.clip(q[:, None] - q[None, :], 0, None)       # Rm[i,j] = [q_i - q_j]_+
            tgt[s, i] = p + alpha * (Rm @ p - p * Rm.sum(0))
    return tgt, float((Qh.max(2) - Qh.min(2)).max())


def simulate(T=300, alpha=0.1, L=20, sigma=0.0, kappa=1.0, scale=1.0, V=1.0, b=0.5, gamma=0.7,
             S=4, M=3, seed=0, order="random", check=True, persistent=False):
    rng = np.random.default_rng(seed)
    N = 2
    Rmax, CR = scale * V, max(1.0, scale * V)
    thetas = [0]
    for t in range(1, T + 1):
        cur = thetas[-1]
        nxt = cur
        if t % L == 0:
            nxt = (cur + 1) % M if order == "cyclic" else int(rng.choice([x for x in range(M) if x != cur]))
        thetas.append(nxt)
    games = [ro.build_tables(S, M, th, V, b, scale) for th in thetas]          # G_0 .. G_T
    pi = np.full((S, N, M), 1.0 / M)
    fixed_noise = rng.uniform(-sigma, sigma, size=(S, N, M)) if (persistent and sigma > 0) else None
    log = dict(N=[], viol=dict(a=0, b=0, c=0, d=0, valid=0), eps_a=[], eps_c=[], B=[], Lam=[], v=[], steps=0)
    ns, Rw = games[0]
    log["N"].append(tn.team_nash_stats(ns, Rw, pi, gamma)["nash_conv"])
    slack = dict(a=np.inf, b=np.inf, c=np.inf, d=np.inf)
    for t in range(1, T + 1):
        ns, Rw = games[t - 1]
        Q0, Q1 = ro.exact_q(ns, Rw, pi, gamma)
        Q = np.stack([Q0, Q1], 1)
        if fixed_noise is not None:
            noise = fixed_noise                      # persistent (biased) critic error, same every step
        else:
            noise = rng.uniform(-sigma, sigma, size=Q.shape) if sigma > 0 else np.zeros_like(Q)
        Qh = Q + noise
        tgt, Bh = smith_target(pi, Qh, alpha)
        if alpha * (M - 1) * Bh > 1 + 1e-12 or tgt.min() < -1e-12:
            log["viol"]["valid"] += 1                                            # A6 violated
            tgt = np.clip(tgt, 0, None); tgt /= tgt.sum(-1, keepdims=True)
        new = pi + kappa * (tgt - pi)
        eps_a = float(np.abs(tgt - new).sum(-1).max()); eps_c = float(np.abs(noise).max())
        zeta = 2 * alpha * (M - 1) * Bh
        ns1, R1 = games[t]
        v_t = ro.change_size(Rw, R1) if t % L == 0 else 0.0

        if check:
            V_old = tn.team_nash_stats(ns, Rw, pi, gamma)["value"]
            Lam = ro.concentrability(ns, Rw, pi, tgt, gamma)
            # (d) per-agent single-agent improvement vs best-response gap, with the per-agent Lambda
            I_sum = 0.0
            for i in (0, 1):
                dev = pi.copy(); dev[:, i, :] = tgt[:, i, :]
                I_i = tn.policy_value(ns, Rw, dev, gamma).mean() - V_old
                I_sum += I_i
                st = tn.team_nash_stats(ns, Rw, pi, gamma)
                Phi_i = st["gap_agent0"] if i == 0 else st["gap_agent1"]
                alt = pi.copy(); alt[:, i, :] = tgt[:, i, :]
                br = pi.copy(); br[:, i, :] = ro.best_response_policy(ns, Rw, pi, gamma, i)
                lam_i = float((ro.state_distribution(ns, br, gamma) / ro.state_distribution(ns, alt, gamma)).max())
                rhs = alpha * (1 - gamma) * Phi_i ** 2 / lam_i - 4 * alpha * M * Rmax * eps_c / (1 - gamma) ** 2
                slack["d"] = min(slack["d"], I_i - rhs)
                if I_i < rhs - 1e-9: log["viol"]["d"] += 1
            # (c) cross term: potential (= team value) of the joint target vs the sum of unilateral gains
            Xi = 2 * ro_kappa(Rmax, gamma) * N * (N - 1) * (M - 1) ** 2 * Bh ** 2 * alpha ** 2
            gain = tn.policy_value(ns, Rw, tgt, gamma).mean() - V_old
            slack["c"] = min(slack["c"], gain - (I_sum - Xi))
            if gain < I_sum - Xi - 1e-9: log["viol"]["c"] += 1
            # (b) actor tracking
            lhs_b = tn.policy_value(ns, Rw, new, gamma).mean()
            rhs_b = tn.policy_value(ns, Rw, tgt, gamma).mean() - N * Rmax * eps_a / (1 - gamma) ** 2
            slack["b"] = min(slack["b"], lhs_b - rhs_b)
            if lhs_b < rhs_b - 1e-9: log["viol"]["b"] += 1
            # (a) environment shift of the potential (team value)
            lhs_a = tn.policy_value(ns1, R1, new, gamma).mean()
            rhs_a = tn.policy_value(ns, Rw, new, gamma).mean() - 2 * N * CR * v_t / (1 - gamma) ** 2
            slack["a"] = min(slack["a"], lhs_a - rhs_a)
            if lhs_a < rhs_a - 1e-9: log["viol"]["a"] += 1
            log["Lam"].append(Lam)
        else:
            log["Lam"].append(ro.concentrability(ns, Rw, pi, tgt, gamma))
        log["eps_a"].append(eps_a); log["eps_c"].append(eps_c); log["B"].append(Bh); log["v"].append(v_t)
        pi = new
        log["N"].append(tn.team_nash_stats(ns1, R1, pi, gamma)["nash_conv"])
        log["steps"] += 1
    log["slack"] = slack
    log["params"] = dict(alpha=alpha, gamma=gamma, N=N, M=M, Rmax=Rmax, CR=CR, T=T)
    return log


def ro_kappa(Rmax, gamma):
    return Rmax * (1 + gamma) / (1 - gamma) ** 3


def bound_rhs(log):
    p = log["params"]; a, g, N, M, Rmax, CR, T = p["alpha"], p["gamma"], p["N"], p["M"], p["Rmax"], p["CR"], p["T"]
    Lam, eps_a, eps_c, B = max(log["Lam"]), max(log["eps_a"]), max(log["eps_c"]), max(log["B"])
    VT = sum(log["v"])
    Xi = 2 * ro_kappa(Rmax, g) * N * (N - 1) * (M - 1) ** 2 * B ** 2 * a ** 2
    terms = dict(init=4 * N * Rmax / ((1 - g) * T), change=2 * N * CR * VT / ((1 - g) ** 2 * T),
                 actor=N * Rmax * eps_a / (1 - g) ** 2, critic=4 * a * N * M * Rmax * eps_c / (1 - g) ** 2, simult=Xi)
    return Lam * N / (a * (1 - g)) * sum(terms.values()), terms, dict(Lam=Lam, eps_a=eps_a, eps_c=eps_c, B=B, V_T=VT)


def cmd_check(args):
    print("Per-step lemma checks (violations / steps) and minimum slack; slack >= 0 everywhere means the lemma held.\n")
    print(f"{'setting':44s} {'A6 viol':>8s} {'(a)':>5s} {'(b)':>5s} {'(c)':>5s} {'(d)':>5s} {'min slack (a,b,c,d)':>34s}  bound ratio RHS/LHS")
    rng = np.random.default_rng(0)
    tot = dict(a=0, b=0, c=0, d=0)
    for k in range(args.n):
        alpha = float(rng.choice([0.02, 0.05, 0.1, 0.2, 0.4])); L = int(rng.choice([5, 10, 20, 40]))
        sigma = float(rng.choice([0.0, 0.05, 0.2])); kappa = float(rng.choice([1.0, 0.7, 0.4]))
        scale = float(rng.choice([0.5, 1.0]))
        lg = simulate(T=args.T, alpha=alpha, L=L, sigma=sigma, kappa=kappa, scale=scale, seed=k)
        rhs, _, _ = bound_rhs(lg); lhs = float(np.mean(np.array(lg["N"][:-1]) ** 2))
        v = lg["viol"]; sl = lg["slack"]
        for key in tot: tot[key] += v[key]
        tag = f"a={alpha} L={L} sig={sigma} kap={kappa} sc={scale}"
        print(f"{tag:44s} {v['valid']:8d} {v['a']:5d} {v['b']:5d} {v['c']:5d} {v['d']:5d} "
              f"{sl['a']:8.1e} {sl['b']:8.1e} {sl['c']:8.1e} {sl['d']:8.1e}  {rhs / max(lhs, 1e-12):10.1e}")
    print("\ntotal lemma violations:", tot)


def cmd_sweep(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    base = dict(T=args.T, alpha=0.1, L=20, sigma=0.0, kappa=1.0, scale=1.0)
    key = {"alpha": "alpha", "L": "L", "sigma": "sigma", "kappa": "kappa", "scale": "scale"}[args.param]
    xs, means, errs, rhss = [], [], [], []
    print(f"{args.param:>8s}  mean N^2 (±sd over seeds)   mean N   Theorem-1 RHS")
    for val in args.values:
        kw = dict(base); kw[key] = type(base[key])(val); kw["persistent"] = args.persistent; vals, ns_, rh = [], [], []
        for sd in range(args.seeds):
            lg = simulate(seed=sd, check=False, **kw)
            N_ = np.array(lg["N"][args.warmup:]); vals.append(float(np.mean(N_ ** 2))); ns_.append(float(np.mean(N_)))
            rh.append(bound_rhs(lg)[0])
        xs.append(val); means.append(np.mean(vals)); errs.append(np.std(vals)); rhss.append(np.mean(rh))
        print(f"{val:8g}  {np.mean(vals):10.4f} ± {np.std(vals):7.4f}   {np.mean(ns_):7.4f}   {np.mean(rh):10.3e}")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(xs, means, yerr=errs, marker="o", lw=2, capsize=3, label="measured  mean $N_t^2$")
    ax.set_xlabel(args.param); ax.set_ylabel("time-averaged NashConv$^2$"); ax.grid(alpha=.3)
    if args.param in ("alpha", "L"): ax.set_xscale("log")
    ax.legend(); fig.tight_layout(); fig.savefig(args.out); print("saved", args.out)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--n", type=int, default=12); c.add_argument("--T", type=int, default=120)
    s = sub.add_parser("sweep"); s.add_argument("--param", required=True, choices=["alpha", "L", "sigma", "kappa", "scale"])
    s.add_argument("--values", type=float, nargs="+", required=True); s.add_argument("--T", type=int, default=400)
    s.add_argument("--seeds", type=int, default=5); s.add_argument("--warmup", type=int, default=60)
    s.add_argument("--out", default="sweep.pdf")
    s.add_argument("--persistent", action="store_true", help="critic error is a fixed bias, not fresh noise each step")
    a = ap.parse_args()
    cmd_check(a) if a.cmd == "check" else cmd_sweep(a)


if __name__ == "__main__":
    main()
