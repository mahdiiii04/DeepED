"""
Exact oracles for the stateful ring potential game (RingGameEnv). Pure numpy (testable without torch).

Game: S states on a ring, 2 agents, M actions each, SHARED reward (=> exact Markov potential game).
In state s with phase offset theta the target action is tau = (theta + s) % M and the stage reward is
    scale * ( V * [a0 = a1 = tau] + b * [exactly one agent plays tau] ).
The ring advances (s -> s+1) only when BOTH agents play the target; otherwise the state stays.
This makes (tau, tau) the unique equilibrium of every state (see tests), for any 0 < b < V.
Everything here is exact: policy value, best responses, NashConv, true Q-values (critic error),
and the concentrability coefficient of the theory.
"""
import numpy as np

from envs.gridworld import team_nash as tn   # generic [S,A,A] solver: policy_value, best_response_value, ...


def build_tables(S, M, theta, V=1.0, b=0.25, scale=1.0):
    """next_state [S,M,M] (int64) and reward [S,M,M] (float64) for phase offset theta."""
    next_state = np.zeros((S, M, M), dtype=np.int64)
    reward = np.zeros((S, M, M))
    for s in range(S):
        tau = (theta + s) % M
        for a0 in range(M):
            for a1 in range(M):
                on0, on1 = a0 == tau, a1 == tau
                reward[s, a0, a1] = scale * (V * (on0 and on1) + b * (on0 != on1))
                next_state[s, a0, a1] = (s + 1) % S if (on0 and on1) else s
    return next_state, reward


def change_size(reward_old, reward_new):
    """v_t of the theory: sup |r_new - r_old| (transitions do not change)."""
    return float(np.abs(reward_new - reward_old).max())


def exact_q(next_state, reward, probs, gamma):
    """True marginal action values Q_i^pi(s, a_i) of both agents: arrays [S, M] each."""
    V = tn.policy_value(next_state, reward, probs, gamma)
    G = reward + gamma * V[next_state]                       # [S, a0, a1]
    Q0 = (G * probs[:, 1, None, :]).sum(2)
    Q1 = (G * probs[:, 0, :, None]).sum(1)
    return Q0, Q1


def best_response_policy(next_state, reward, probs, gamma, agent, V0=None):
    """Deterministic best response of `agent` to the others (probs [S,2,M]); returns probs-like [S,M]."""
    if V0 is None:
        V0 = tn.policy_value(next_state, reward, probs, gamma)
    Vb = tn.best_response_value(next_state, reward, probs, gamma, agent, V0)
    G = reward + gamma * Vb[next_state]
    opp = probs[:, 1, :] if agent == 0 else probs[:, 0, :]
    Q = (G * opp[:, None, :]).sum(2) if agent == 0 else (G * opp[:, :, None]).sum(1)
    br = np.zeros_like(Q)
    br[np.arange(Q.shape[0]), Q.argmax(1)] = 1.0
    return br


def state_distribution(next_state, probs, gamma):
    """Discounted state distribution d^pi from the uniform start distribution."""
    S, M, _ = next_state.shape
    joint = probs[:, 0, :, None] * probs[:, 1, None, :]
    P = np.zeros((S, S))
    np.add.at(P, (np.repeat(np.arange(S), M * M), next_state.reshape(-1)), joint.reshape(-1))
    rho = np.full(S, 1.0 / S)
    return (1 - gamma) * rho @ np.linalg.inv(np.eye(S) - gamma * P)


def concentrability(next_state, reward, probs, target, gamma):
    """
    Lambda = max_i sup_s d*(s) / d'(s), with d' the state distribution under (target_i, pi_{-i})
    and d* under (best response_i, pi_{-i}).  target: [S, 2, M] Smith targets of both agents.
    """
    lam = 0.0
    for i in (0, 1):
        alt = probs.copy(); alt[:, i, :] = target[:, i, :]
        br = probs.copy(); br[:, i, :] = best_response_policy(next_state, reward, probs, gamma, i)
        d_alt = state_distribution(next_state, alt, gamma)
        d_br = state_distribution(next_state, br, gamma)
        lam = max(lam, float((d_br / d_alt).max()))
    return lam


def q_range(Q):
    """B: largest gap between two actions' values in the same state (Q: [S, M])."""
    return float((Q.max(1) - Q.min(1)).max())
