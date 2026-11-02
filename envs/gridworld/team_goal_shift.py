import torch
from tensordict import TensorDict

from envs.gridworld.envs import GridWorldEnv, _REGISTRY
from envs.gridworld import team_nash as tn


class TeamGoalShiftEnv(GridWorldEnv):
    """
    Identical-interest (team-reward) navigation with an exogenously moving goal.
    A finite-state Markov potential game: every agent receives the SAME reward, so the shared
    value is an exact potential.

    * 2 agents on a K x K grid, actions: stay / up / down / left / right.
    * ONE active goal cell, chosen from a small fixed set of candidate cells and shared by all
      parallel envs. It moves every ``phase_length`` collector steps (``phase_order``:
      "cyclic" or "random", never repeating; ``phase_seed`` seeds the random order).
    * The goal is NOT in the observation (obs = own position + other agent's position), exactly
      like the phase is hidden in the matrix game. A policy must infer/relearn the goal from reward.
    * Shared reward per step: step_penalty + shaping_coef * (total Manhattan distance to the
      goal before - after) + goal_reward * [all agents on the goal]
      + agent_goal_reward * (number of agents on the goal).
      agent_goal_reward = 0 (default): pure coordination game, an old convention stays an
      approximate equilibrium after the goal moves (NashConv barely registers the switch).
      agent_goal_reward > 0 (e.g. 0.1): a lone agent on the goal is paid, so the old convention is
      exploitable and the unique equilibrium moves with the goal. Both versions are identical-interest.
    * No agent-specific out-of-bounds penalty (that would break the identical-interest property);
      moving into a wall just stays in place.

    ``exact_tables()`` / ``state_positions()`` / ``obs_from_positions()`` let the training script
    compute EXACT NashConv by value iteration (see envs/gridworld/team_nash.py).
    """

    n_agents = 2
    n_goals = 0               # no goal in the observation
    grid_size = 5
    extra_obs_dim = 0
    OUT_OF_BOUNDS_VIOLATION = 0.0

    def __init__(
            self,
            phase_length: int = 1500,
            phase_order: str = "cyclic",
            phase_seed: int = 0,
            grid_size: int = 5,
            goal_cells=None,
            step_penalty: float = -0.01,
            shaping_coef: float = 0.1,
            goal_reward: float = 1.0,
            agent_goal_reward: float = 0.0,
            **kwargs,
    ):
        if phase_order not in ("cyclic", "random"):
            raise ValueError(f"phase_order must be 'cyclic' or 'random', got {phase_order!r}")
        if goal_cells is None:
            goal_cells = [(0, 0), (0, grid_size - 1), (grid_size - 1, grid_size // 2)]
        if len(goal_cells) < 2:
            raise ValueError("need at least 2 candidate goal cells")
        self.grid_size = grid_size
        self._phase_length = phase_length
        self._phase_order = phase_order
        self._goal_cells = [tuple(int(v) for v in g) for g in goal_cells]
        self.step_penalty = step_penalty
        self.shaping_coef = shaping_coef
        self.goal_reward = goal_reward
        self.agent_goal_reward = agent_goal_reward

        super().__init__(**kwargs)

        self._phase_rng = torch.Generator().manual_seed(phase_seed)
        self.register_buffer(
            "_cand", torch.tensor(self._goal_cells, dtype=torch.long, device=self.device)
        )
        self.register_buffer(
            "_phase", torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        )
        self.register_buffer(
            "_total_steps", torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        )

    # ------------------------------------------------------------------ goal / phase
    def _active_goal(self) -> torch.Tensor:
        """[ne, 2] active goal cell for every env."""
        return self._cand[self._phase]

    def _next_phase(self, cur: int) -> int:
        G = len(self._goal_cells)
        if self._phase_order == "cyclic":
            return (cur + 1) % G
        others = [g for g in range(G) if g != cur]
        return others[int(torch.randint(len(others), (1,), generator=self._phase_rng).item())]

    # ------------------------------------------------------------------ dynamics hooks
    def _compute_rewards(self, positions, prev_positions, td_in):
        goal = self._active_goal()[:, None, :]                                   # [ne,1,2]
        d_prev = (prev_positions - goal).abs().sum(-1).float().sum(-1)
        d_new = (positions - goal).abs().sum(-1).float().sum(-1)
        on_each = (positions - goal).abs().sum(-1) == 0                          # [ne, N]
        on = on_each.all(-1).float()
        r = (self.step_penalty + self.shaping_coef * (d_prev - d_new)
             + self.goal_reward * on + self.agent_goal_reward * on_each.float().sum(-1))
        return r[:, None, None].expand(self._num_envs, self.n_agents, 1).clone()

    def _extra_reset_state(self, reset_mask) -> dict:
        return {
            "phase": self._phase.clone(),
            "phase_changed": torch.zeros(self._num_envs, dtype=torch.bool, device=self.device),
        }

    def _step(self, tensordict: TensorDict) -> TensorDict:
        self._total_steps.add_(1)
        old_phase = self._phase.clone()

        td = super()._step(tensordict)          # reward uses the OLD goal; the goal moves afterwards

        changed = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        if (self._total_steps[0] % self._phase_length == 0) and (self._total_steps[0] > 0):
            self._phase.fill_(self._next_phase(int(old_phase[0].item())))
            changed = self._phase != old_phase

        td.set("phase", self._phase.clone())
        td.set("phase_changed", changed)
        return td

    # ------------------------------------------------------------------ exact evaluation support
    def state_positions(self):
        """numpy [S, 2, 2]: positions of both agents for every joint state."""
        return tn.state_positions(self.grid_size)

    def obs_from_positions(self, pos: torch.Tensor) -> torch.Tensor:
        """Observation for arbitrary joint positions pos [S, N, 2] (same layout as _build_obs)."""
        gs = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        pf = pos.float() / gs
        obs = []
        for a in range(self.n_agents):
            others = torch.cat([pf[:, j, :] for j in range(self.n_agents) if j != a], dim=-1)
            obs.append(torch.cat([pf[:, a, :], others], dim=-1))
        return torch.stack(obs, dim=1)

    def exact_tables(self):
        """(next_state [S,A,A], reward [S,A,A]) of the CURRENT phase, as numpy arrays."""
        goal = self._active_goal()[0].cpu().numpy()
        return tn.build_tables(
            self.grid_size, goal, self.step_penalty, self.shaping_coef, self.goal_reward,
            self.agent_goal_reward,
        )

    @torch.no_grad()
    def get_extra_metrics(self) -> dict:
        goal = self._active_goal()[:, None, :]
        on = ((self._positions - goal).abs().sum(-1) == 0)
        phase = float(self._phase[0].item())
        return {
            "env/phase": phase,
            "current_phase": phase,
            "all_on_goal_rate": on.all(dim=1).float().mean().item(),
            "any_on_goal_rate": on.any(dim=1).float().mean().item(),
        }


_REGISTRY["team_goal_shift"] = TeamGoalShiftEnv
