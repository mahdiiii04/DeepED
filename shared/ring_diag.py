import numpy as np

from envs.ring import ring_oracle as ro


def ring_diagnostics(algo, env, gamma: float) -> dict:
    """
    Measured quantities of the theory on the ring game, against EXACT oracles:
      diag/eps_c       sup |critic Q_i - true Q_i^pi| for the live policy pi (assumption A3)
      diag/eps_c_avg   same against the EMA (average) policy the critic is actually trained for
      diag/eps_a       sup_{s,i} || Smith target(pi) - pi ||_1: how far the actor is from its own target (proxy of A4)
      diag/critic_range B: largest action gap of the critic in a state
      diag/true_q_range same for the true Q
      diag/lambda      concentrability coefficient (A5), exact
    """
    d = algo.state_diagnostics(env.state_observations())
    q = d["q"].double().cpu().numpy()
    pi = d["pi"].double().cpu().numpy()
    pi_avg = d["pi_avg"].double().cpu().numpy()
    tgt = d["target"].double().cpu().numpy()
    ns, R = env.exact_tables()

    Q0, Q1 = ro.exact_q(ns, R, pi, gamma)
    A0, A1 = ro.exact_q(ns, R, pi_avg, gamma)
    return {
        "diag/eps_c": float(max(np.abs(q[:, 0] - Q0).max(), np.abs(q[:, 1] - Q1).max())),
        "diag/eps_c_avg": float(max(np.abs(q[:, 0] - A0).max(), np.abs(q[:, 1] - A1).max())),
        "diag/eps_a": float(np.abs(tgt - pi).sum(-1).max()),
        "diag/critic_range": float((q.max(-1) - q.min(-1)).max()),
        "diag/true_q_range": float(max(ro.q_range(Q0), ro.q_range(Q1))),
        "diag/lambda": float(ro.concentrability(ns, R, pi, tgt, gamma)),
    }
