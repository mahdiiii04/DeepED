"""
Exact tables and exact NashConv for the 2-agent team-reward grid game (TeamGoalShiftEnv).

Setting (matches the theory exactly): finite state space (joint positions), deterministic
transitions, identical-interest reward, discount gamma, stationary policies that see the joint
position, uniform start distribution rho. Because the reward is shared, V_1 = V_2 = V is an exact
potential, and a unilateral best response is a single-agent MDP solved by value iteration.

Pure numpy on purpose: it can be unit-tested without torch.
"""
import numpy as np

DR = np.array([0, -1, 1, 0, 0])   # stay / up / down / left / right  (same order as GridWorldEnv)
DC = np.array([0, 0, 0, -1, 1])


def state_positions(K: int) -> np.ndarray:
    """[S, 2, 2] (row, col) of both agents for every joint state; s = idx0 * K^2 + idx1, idx = r*K + c."""
    idx = np.arange(K * K)
    cell = np.stack([idx // K, idx % K], -1)
    p0 = np.broadcast_to(cell[:, None, :], (K * K, K * K, 2))
    p1 = np.broadcast_to(cell[None, :, :], (K * K, K * K, 2))
    return np.stack([p0, p1], axis=2).reshape(K ** 4, 2, 2)


def team_reward(prev, new, goal, step_penalty, shaping_coef, goal_reward):
    """Shared reward. prev/new: [..., 2, 2]; goal: [2]. Same formula as TeamGoalShiftEnv._compute_rewards."""
    d_prev = np.abs(prev - goal).sum(-1).sum(-1)
    d_new = np.abs(new - goal).sum(-1).sum(-1)
    on = (np.abs(new - goal).sum(-1) == 0).all(-1)
    return step_penalty + shaping_coef * (d_prev - d_new) + goal_reward * on


def build_tables(K, goal, step_penalty=-0.01, shaping_coef=0.1, goal_reward=1.0):
    """next_state [S,A,A] (int64) and reward [S,A,A] (float64) for joint action (a0, a1)."""
    pos = state_positions(K)
    S, A = pos.shape[0], len(DR)
    prev = np.broadcast_to(pos[:, None, None, :, :], (S, A, A, 2, 2))
    new = prev.copy()
    new[:, :, :, 0, 0] += DR[None, :, None]
    new[:, :, :, 0, 1] += DC[None, :, None]
    new[:, :, :, 1, 0] += DR[None, None, :]
    new[:, :, :, 1, 1] += DC[None, None, :]
    new = np.clip(new, 0, K - 1)
    reward = team_reward(prev, new, np.asarray(goal), step_penalty, shaping_coef, goal_reward)
    i0 = new[..., 0, 0] * K + new[..., 0, 1]
    i1 = new[..., 1, 0] * K + new[..., 1, 1]
    return (i0 * (K * K) + i1).astype(np.int64), reward.astype(np.float64)


def policy_value(next_state, reward, probs, gamma):
    S, A, _ = reward.shape
    p0, p1 = probs[:, 0, :], probs[:, 1, :]
    joint = p0[:, :, None] * p1[:, None, :]
    r_pi = (joint * reward).sum((1, 2))
    P = np.zeros((S, S))
    np.add.at(P, (np.repeat(np.arange(S), A * A), next_state.reshape(-1)), joint.reshape(-1))
    return np.linalg.solve(np.eye(S) - gamma * P, r_pi)


def best_response_value(next_state, reward, probs, gamma, agent, V0, tol=1e-9, max_iter=5000):
    opp = probs[:, 1, :] if agent == 0 else probs[:, 0, :]
    Vb = V0.copy()
    for _ in range(max_iter):
        Qsa = reward + gamma * Vb[next_state]                       # [S, a0, a1]
        Q = (Qsa * opp[:, None, :]).sum(2) if agent == 0 else (Qsa * opp[:, :, None]).sum(1)
        Vn = Q.max(1)
        if np.abs(Vn - Vb).max() < tol:
            return Vn
        Vb = Vn
    return Vb


def optimal_team_value(next_state, reward, gamma, tol=1e-9, max_iter=5000):
    S = reward.shape[0]
    V = np.zeros(S)
    for _ in range(max_iter):
        Vn = (reward + gamma * V[next_state]).reshape(S, -1).max(1)
        if np.abs(Vn - V).max() < tol:
            return Vn
        V = Vn
    return V


def team_nash_stats(next_state, reward, probs, gamma):
    """
    probs: [S, 2, A] policy of both agents on every joint state.
    Returns NashConv = sum_i E_rho[V_i^BR - V], the team value V (= the potential),
    the optimal team value, and their gap. rho is uniform over joint states.
    """
    V = policy_value(next_state, reward, probs, gamma)
    br0 = best_response_value(next_state, reward, probs, gamma, 0, V)
    br1 = best_response_value(next_state, reward, probs, gamma, 1, V)
    Vopt = optimal_team_value(next_state, reward, gamma)
    g0, g1 = float((br0 - V).mean()), float((br1 - V).mean())
    return dict(
        nash_conv=max(g0 + g1, 0.0),
        gap_agent0=max(g0, 0.0), gap_agent1=max(g1, 0.0),
        value=float(V.mean()), optimal_value=float(Vopt.mean()),
        value_gap=float(Vopt.mean() - V.mean()),
    )
