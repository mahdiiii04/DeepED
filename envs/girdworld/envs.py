import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Bounded,
    Unbounded,
    OneHot,
    Composite,
)

N_MOVE_ACTIONS = 5 # stay / up / down / left / right

_DR = torch.tensor([0, -1, 1, 0, 0], dtype=torch.long) # row delta
_DC = torch.tensor([0, 0, 0, -1, 1], dtype=torch.long) # column delta

###################### Base Class #########################

class GridWorldEnv(EnvBase):
    """
    Abstract class for vectorised multi-agent grid-world environments.

    MUST Implement
    --------------
    ``_compute_rewards(positions, prev_positions, td_in) -> Tensor [ne, n_agents, 1]``
        Given current and previous positions, return per-agent rewards.
    
    ``_reset_positions(reset_mask) -> Tensor [ne, n_agents, 1]``
        Return new starting positions (row, col) for each agent for every
        environment flagged in ```reset_mask``.

    ``_reset_goals(reset_mask) -> Tensor [ne, n_goals, 2]``
        Return new goal positions for each goal for every environment
        flagged in ``reset_mask``.

    Optional overrides
    ------------------
    ``_extra_obs(positions, goals) -> Tensor [ne, n_agents, extra_obs_dim]``
        Additional features in the observation. 
        Default: empty.

    ``_extra_reset_state(reset_mask) -> dict``
        Extra Tensordict entries during reset.

    ``_extra_step_state(reset_mask) -> dict``
        Extra Tensordict entries during step.
    """

    n_agents: int = 2
    n_goals: int = 2
    grid_size: int = 8
    extra_obs_dim: int = 0

    OUT_OF_BOUNDS_VIOLATION = -0.1

    def __init__(
        self,
        num_envs: int = 1,
        max_steps: int = 50,
        device: str = "cpu",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(batch_size=torch.Size([num_envs]), device=device)

        self._num_envs = num_envs
        self._max_steps = max_steps
        self.n_actions = N_MOVE_ACTIONS

        # deltas
        self.register_buffer(
            "_dr", _DR.clone()
        )
        self.register_buffer(
            "_dc", _DC.clone()
        )

        self.register_buffer(
            "_step_count", 
            torch.zeros(num_envs, dtype=torch.long, device=device),
        )

        # (row, col) for each agents
        self.register_buffer(
            "_positions",
            torch.zeros(num_envs, self.n_agents, 2, dtype=torch.long, device=device),
        )

        # (row, col) for each goal
        if self.n_goals > 0:
            self.register_buffer(
                "_goals",
                torch.zeros(num_envs, self.n_goals, 2, dtype=torch.long, device=device),
            )
        else:
            self._goals = None

        self._obs_dim = (
            2                           # own pos
            + 2 * (self.n_agents - 1)    # others's pos
            + 2 * self.n_goals          # goals's pos
            + self.extra_obs_dim
        )

        self._make_specs()
        self.set_seed(seed)

    
    def _make_specs(self) -> None:

        ne, n, a, o = self._num_envs, self.n_agents, self.n_actions, self._obs_dim
        gs = self.grid_size

        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Unbounded(shape=(ne, n, o), dtype=torch.float32)
                    },
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

        self.action_spec = Composite(
            {
                "agents": Composite(
                    {
                        "action": OneHot(n=a, shape=(ne, n, a), dtype=torch.long)
                    },
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

        self.reward_spec = Composite(
            {
                "agents": Composite(
                    {
                       "reward": Unbounded(shape=(ne, n, 1), dtype=torch.float32)
                    },
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

        self.done_spec = Composite(
            {
                "agents": Composite(
                    {
                        "done": Bounded(low=0, high=1, shape=(ne, n, 1), dtype=torch.bool),
                        "terminated": Bounded(low=0, high=1, shape=(ne, n, 1), dtype=torch.bool),
                    },
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

    @property
    def reward_key(self):
        return ("agents", "reward")
    
    @property
    def action_key(self):
        return ("agents", "action")
    
    @property
    def done_keys(self):
        return [("agents", "done"), ("agents", "terminated")]
    
    @property
    def max_steps(self) -> int:
        return self._max_steps
    
    def _clamp_positions(
            self, pos: torch.Tensor
    ) -> torch.Tensor:
        """Keep positions within [0, grid_size-1]"""
        return pos.clamp(0, self.grid_size - 1)
    
    def _build_obs(
            self, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Build observation tensor

        Params
        ------
        positions : [ne, n_agents, 2] 

        Returns
        ------
        obs : [ne, n_agents, obs_dim]
        """

        ne = self._num_envs
        gs = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        pos_f = positions.float() / gs

        obs_parts = []
        for a_i in range(self.n_agents):
            own = pos_f[:, a_i, :]
            others = torch.cat(
                [pos_f[:, j, :] for j in range(self.n_agents) if j != a_i],
                dim=-1,
            )
            parts = [own, others]

            if self.n_goals > 0 and self._goals is not None:
                goals_f = self._goals.float() / gs
                goals_flat = goals_f.reshape(ne, -1)
                parts.append(goals_flat)

            obs_parts.append(torch.cat(parts, dim=-1))

        obs = torch.stack(obs_parts, dim=1)

        if self.extra_obs_dim > 0:
            extra = self._extra_obs(positions, self._goals)
            obs = torch.cat([obs, extra], dim=-1)

        return obs
    
    def _random_positions(
        self, ne: int, n: int, device
    ) -> torch.Tensor:
        """"Sample ``n`` random poistions across ``ne`` envs."""
        rows = torch.randint(0, self.grid_size, (ne, n), device=device)
        cols = torch.randint(0, self.grid_size, (ne, n), device=device)
        return torch.stack([rows, cols], dim=-1)
    
    def _reset(
        self, tensordict=None
    ) -> TensorDict:
        ne = self._num_envs
        dev = self.device

        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict.get("_reset").reshape(ne)
        else:
            reset_mask = torch.ones(ne, dtype=torch.bool, device=dev)
        
        self._step_count[reset_mask] = 0

        new_pos = self._reset_positions(reset_mask)
        self._positions[reset_mask] = new_pos[reset_mask]

        if self.n_goals > 0:
            new_goals = self._reset_goals(reset_mask)
            self._goals[reset_mask] = new_goals[reset_mask]
        
        obs = self._build_obs(self._positions)
        
        td = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation" : obs
                    },
                    batch_size=[ne, self.n_agents],
                    device=dev,
                )
            },
            batch_size=[ne],
            device=dev,
        )

        extra = self._extra_reset_state(reset_mask)
        for k,v in extra.items():
            td.set(k, v)

        return td
    
    def _step(
        self, tensordict: TensorDict
    ) -> TensorDict:
        ne = self._num_envs
        dev = self.device

        action_raw = tensordict.get(("agents", "action"))
        if action_raw.dim() == 3 and action_raw.shape[-1] == self.n_actions:
            actions = action_raw.argmax(dim=-1)
        else:
            actions = action_raw.long()

        prev_positions = self._positions.clone()

        dr = self._dr[actions]
        dc = self._dc[actions]
        intended_pos = self._positions.clone()
        intended_pos[..., 0] = intended_pos[..., 0] + dr
        intended_pos[..., 1] = intended_pos[..., 1] + dc

        new_pos = intended_pos.clone()
        new_pos = self._clamp_positions(new_pos)

        violation_mask = (new_pos != intended_pos).any(dim=-1)

        self._positions = new_pos
        self._step_count.add_(1)

        rewards = self._compute_rewards(new_pos, prev_positions, tensordict)

        violation_penalty = violation_mask.unsqueeze(-1).float() * self.OUT_OF_BOUNDS_VIOLATION
        rewards = rewards + violation_penalty

        done_env = self._step_count >= self._max_steps
        done_agent = done_env[:, None, None].expand(ne, self.n_agents, 1)

        obs = self._build_obs(new_pos)

        td = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation" : obs,
                        "reward": rewards,
                        "done": done_agent.clone(),
                        "terminated": done_agent.clone(),
                    },
                    batch_size=[ne, self.n_agents],
                    device=dev,
                )
            },
            batch_size=[ne],
            device=dev,
        )

        extra = self._extra_step_state(tensordict)
        for k, v in extra.items():
            td.set(k, v)
        
        return td
    
    def _set_seed(self, seed) -> None:
        torch.manual_seed(seed)
    
    ######################### To Override ###########################

    def _reset_positions(
            self, reset_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return [ne, n_agents, 2] long tensor of starting positions."""
        ne, dev = self._num_envs, self.device
        return self._random_positions(ne, self.n_agents, dev)
    
    def _reset_goals(
            self, reset_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return [ne, n_goals, 2] long tensor of goal positions."""
        ne, dev = self._num_envs, self.device
        return self._random_positions(ne, self.n_goals, dev)
    
    def _compute_rewards(
        self, 
        positions: torch.Tensor,
        prev_positions: torch.Tensor,
        td_in: TensorDict,
    ) -> torch.Tensor:
        """Return [ne, n_agents, 1] float reward tensor."""
        raise NotImplementedError
    
    def _extra_obs(
        self, positions, goals
    ) -> torch.Tensor:
        ne, dev = self._num_envs, self.device
        return torch.zeros(ne, self.n_agents, 0, device=dev)
    
    def _extra_reset_state(
        self, reset_mask
    ) -> dict:
        return {}
    
    def _extra_step_state(
        self, td
    ) -> dict:
        return {}
    

    @torch.no_grad()
    def get_extra_metrics(
        self
    ) -> dict:
        """
        Return a dict of scalar metrics for the *current* positions.
        Subclasses can override to add domain-specific metrics.
        """
        return {}

######################### Scenarios ############################

class CooperativeNavEnv(GridWorldEnv):
    """
    All agents must reach the same goal.

    Agents are rewarded when they are all simultaneously on the goal.
    They also get a small reward for getting close to it (shaping).
    """

    n_agents = 2
    n_goals = 1
    grid_size = 8
    extra_obs_dim = 0

    ALL_ON_GOAL_REWARD = 2.0
    STEP_PENALTY = -0.01
    SHAPING_COEF = 0.1

    def _dist_to_goal(
        self, positions
    ):
        """L1 distance from each agent to the goal"""
        goal = self._goals[:, 0, :] # [ne, 2]
        diff = (positions - goal[:, None, :]).abs()
        return diff.sum(-1).float()
    
    def _compute_rewards(
        self, positions, prev_positions, td_in
    ):
        ne = self._num_envs
        dev = self.device

        rewards = torch.full(
            (ne, self.n_agents, 1),
            self.STEP_PENALTY,
            dtype=torch.float32,
            device=dev,
        )

        #shaping
        prev_dist = self._dist_to_goal(prev_positions)
        curr_dist = self._dist_to_goal(positions)
        shaping = (prev_dist - curr_dist) * self.SHAPING_COEF
        rewards[:, :, 0] += shaping

        goal = self._goals[:, 0, :]
        on_goal = (
            (positions[:, :, 0] == goal[:, None, 0]) &
            (positions[:, :, 1] == goal[:, None, 1])
        )
        all_on = on_goal.all(dim=1, keepdim=True).expand_as(on_goal)
        rewards[:, :, 0] += all_on.float() * self.ALL_ON_GOAL_REWARD

        return rewards

    @torch.no_grad()
    def get_extra_metrics(self) -> dict:
        goal = self._goals[:, 0, :]
        on = (
            (self._positions[:, :, 0] == goal[:, None, 0]) &
            (self._positions[:, :, 1] == goal[:, None, 1])
        )
        all_on = on.all(dim=1).float().mean()
        any_on = on.any(dim=1).float().mean()

        return {
            "all_on_goal_rate": all_on.item(),
            "any_on_goal_rate": any_on.item(),
        }

class NSGridWorldMixin:
    """
    Adds non-stationarity (goal swap every phase_length steps)
    to any GridWorldEnv subclass.

    Usage:  class NSFoo(NSGridWorldMixin, FooEnv): ...
    """
    def __init__(self, phase_length: int = 550, **kwargs):
        self._phase_length = phase_length
        self._goals_initialized = False
        super().__init__(**kwargs)
        self.register_buffer("_total_steps",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device))
        self.register_buffer("_episode_count",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device))

    def _reset(self, tensordict=None):
        ne, dev = self._num_envs, self.device
        reset_mask = (
            tensordict.get("_reset").reshape(ne)
            if tensordict is not None and "_reset" in tensordict.keys()
            else torch.ones(ne, dtype=torch.bool, device=dev)
        )
        if self._goals_initialized:
            self._episode_count[reset_mask] += 1
        self._step_count[reset_mask] = 0
        new_pos = self._reset_positions(reset_mask)
        self._positions[reset_mask] = new_pos[reset_mask]
        if not self._goals_initialized:
            init_goals = self._reset_goals(reset_mask)
            self._goals[reset_mask] = init_goals[reset_mask]
            self._goals_initialized = True
        obs = self._build_obs(self._positions)
        return TensorDict(
            {
                "agents": TensorDict({"observation": obs},
                                     batch_size=[ne, self.n_agents], device=dev),
                "goals_changed":  torch.zeros(ne, dtype=torch.bool, device=dev),
                "episode_changed": torch.zeros(ne, dtype=torch.bool, device=dev),
                "total_steps":    self._total_steps.clone(),
                "episode_count":  self._episode_count.clone(),
            },
            batch_size=[ne], device=dev,
        )

    def _step(self, tensordict: TensorDict):
        self._total_steps.add_(1)
        goal_changed = (self._total_steps % self._phase_length == 0) & (self._total_steps > 0)
        if goal_changed.any():
            fresh = self._reset_goals(goal_changed)
            self._goals[goal_changed] = fresh[goal_changed]
        td = super()._step(tensordict)
        episode_changed = self._step_count >= self._max_steps
        td.set("goal_changed",    goal_changed)
        td.set("episode_changed", episode_changed)
        td.set("total_steps",     self._total_steps.clone())
        td.set("episode_count",   self._episode_count.clone())
        return td

class NSCooperativeNavEnv(NSGridWorldMixin, CooperativeNavEnv):
    """Non-stationary cooperative navigation."""

#################### Registery + Factory ######################

_REGISTRY: dict[str, type[GridWorldEnv]] = {
    "cooperative_nav": CooperativeNavEnv,
    "ns_cooperative_nav": NSCooperativeNavEnv,
}

def GridWorldFactory(
    scenario: str,
    num_envs: int = 1,
    max_steps: int = 50,
    device: str = "cpu",
    seed: int = 0,
    **kwargs,
):
    if scenario not in _REGISTRY:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {sorted(_REGISTRY.keys())}."
        )
    cls = _REGISTRY[scenario]
    return cls(
        num_envs=num_envs,
        max_steps=max_steps,
        device=device,
        seed=seed,
        **kwargs,
    )