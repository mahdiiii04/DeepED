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

        # deltas — must live on the same device as actions for indexing in _step
        self.register_buffer(
            "_dr", _DR.clone().to(device)
        )
        self.register_buffer(
            "_dc", _DC.clone().to(device)
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

    Adaptation Metrics
    ------------------
    Tracks *time to reconverge* after each goal change.  After a change, we
    wait until the per-env reward EMA recovers to ``reconverge_threshold``
    (default 0.95) of the pre-change EMA baseline.  Call
    ``get_adaptation_metrics()`` at any time to read:

    ``mean_time_to_reconverge``
        Average steps to reconverge over all completed adaptations.
    ``median_time_to_reconverge``
        Median steps (less sensitive to outliers).
    ``reconverge_rate``
        Fraction of goal-change events that have already reconverged.
    ``mean_steps_since_change``
        Average steps elapsed (across envs still adapting) — indicates
        how much of the current adaptation window has been spent.
    ``n_goal_changes``
        Total number of goal-change events recorded (summed over envs).
    ``n_reconverged``
        Total reconvergence events recorded so far.
    """

    # EMA smoothing factor for reward tracking.  Higher = faster response,
    # noisier; lower = smoother, slower to register recovery.
    REWARD_EMA_ALPHA: float = 0.05

    # An env is considered reconverged when its EMA >= this fraction of
    # the pre-change EMA baseline.
    RECONVERGE_THRESHOLD: float = 0.95

    def __init__(
        self,
        phase_length: int = 550,
        reward_ema_alpha: float = None,
        reconverge_threshold: float = None,
        **kwargs,
    ):
        self._phase_length = phase_length
        self._goals_initialized = False

        if reward_ema_alpha is not None:
            self.REWARD_EMA_ALPHA = reward_ema_alpha
        if reconverge_threshold is not None:
            self.RECONVERGE_THRESHOLD = reconverge_threshold

        super().__init__(**kwargs)

        ne, dev = self._num_envs, self.device

        self.register_buffer("_total_steps",
            torch.zeros(ne, dtype=torch.long, device=dev))
        self.register_buffer("_episode_count",
            torch.zeros(ne, dtype=torch.long, device=dev))

        # ── adaptation tracking ──────────────────────────────────────────────
        # Exponential moving average of mean-agent reward, per env.
        self.register_buffer("_reward_ema",
            torch.zeros(ne, dtype=torch.float32, device=dev))

        # Whether the EMA has been seeded with at least one real observation.
        self.register_buffer("_ema_seeded",
            torch.zeros(ne, dtype=torch.bool, device=dev))

        # Reward EMA captured just before the most recent goal change.
        # NaN for envs that have never changed goals.
        self.register_buffer("_baseline_reward",
            torch.full((ne,), float("nan"), dtype=torch.float32, device=dev))

        # Steps elapsed since the last goal change (0 = not currently adapting).
        self.register_buffer("_steps_since_change",
            torch.zeros(ne, dtype=torch.long, device=dev))

        # Whether each env is currently in an adaptation window.
        self.register_buffer("_adapting",
            torch.zeros(ne, dtype=torch.bool, device=dev))

        # Phase index: increments by 1 each time goals change (per env).
        self.register_buffer("_phase_index",
            torch.zeros(ne, dtype=torch.long, device=dev))

        # Cumulative counters (CPU lists — negligible overhead).
        self._reconverge_times: list[float] = []   # steps for each reconvergence
        self._n_goal_changes: int = 0               # total change events

    # ── helpers ──────────────────────────────────────────────────────────────

    def _mean_reward_per_env(self, td: TensorDict) -> torch.Tensor:
        """Return shape [ne] scalar reward (mean over agents)."""
        r = td.get(("agents", "reward"))   # [ne, n_agents, 1]
        return r.squeeze(-1).mean(dim=1)   # [ne]

    def _update_ema(self, mean_r: torch.Tensor) -> None:
        """Update the per-env reward EMA in-place."""
        alpha = self.REWARD_EMA_ALPHA
        # First observation: seed directly so we don't start from 0.
        seed_mask = ~self._ema_seeded
        self._reward_ema[seed_mask] = mean_r[seed_mask]
        self._ema_seeded[seed_mask] = True
        # Subsequent observations: standard EMA.
        update_mask = self._ema_seeded & ~seed_mask
        self._reward_ema[update_mask] = (
            alpha * mean_r[update_mask]
            + (1.0 - alpha) * self._reward_ema[update_mask]
        )

    def _record_goal_change(self, changed: torch.Tensor) -> None:
        """Snapshot baseline and open an adaptation window for changed envs."""
        n_changed = changed.sum().item()
        if n_changed == 0:
            return
        self._n_goal_changes += int(n_changed)
        # Snapshot EMA as the baseline for all envs whose goal just changed.
        # If an env was still adapting from a previous change, that window is
        # abandoned (we start fresh with the current EMA as the new baseline).
        self._baseline_reward[changed] = self._reward_ema[changed].clone()
        self._steps_since_change[changed] = 0
        self._adapting[changed] = True

    def _check_reconvergence(self) -> None:
        """
        For every adapting env, test whether the reward EMA has recovered
        to >= RECONVERGE_THRESHOLD * baseline.  Record and close windows
        where recovery is confirmed.
        """
        if not self._adapting.any():
            return

        baseline = self._baseline_reward   # [ne]
        ema = self._reward_ema             # [ne]
        threshold = self.RECONVERGE_THRESHOLD

        # We can only check envs with a valid (non-NaN, non-zero) baseline.
        valid_baseline = self._adapting & baseline.isfinite() & (baseline.abs() > 1e-8)

        if not valid_baseline.any():
            return

        # Recovery condition: ema >= threshold * baseline.
        # Works correctly for positive baselines (typical shaped rewards).
        # For negative baselines the inequality flips — we handle that below.
        positive_base = valid_baseline & (baseline > 0)
        negative_base = valid_baseline & (baseline <= 0)

        recovered = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        recovered[positive_base] = ema[positive_base] >= threshold * baseline[positive_base]
        # For negative baseline: "closer to 0" means better; 95% of a negative
        # number means less negative, i.e. ema >= threshold * baseline still holds
        # since threshold < 1 and baseline < 0 → threshold*baseline is less negative.
        recovered[negative_base] = ema[negative_base] >= threshold * baseline[negative_base]

        newly_converged = recovered & self._adapting
        if newly_converged.any():
            steps = self._steps_since_change[newly_converged].float()
            self._reconverge_times.extend(steps.tolist())
            self._adapting[newly_converged] = False
            self._steps_since_change[newly_converged] = 0

    # ── overrides ────────────────────────────────────────────────────────────

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
                "goals_changed":   torch.zeros(ne, dtype=torch.bool, device=dev),
                "episode_changed": torch.zeros(ne, dtype=torch.bool, device=dev),
                "total_steps":     self._total_steps.clone(),
                "episode_count":   self._episode_count.clone(),
            },
            batch_size=[ne], device=dev,
        )

    def _step(self, tensordict: TensorDict):
        self._total_steps.add_(1)

        # ── 1. determine which envs change goals this step ──────────────────
        goal_changed = (self._total_steps % self._phase_length == 0) & (self._total_steps > 0)
        if goal_changed.any():
            fresh = self._reset_goals(goal_changed)
            self._goals[goal_changed] = fresh[goal_changed]
            self._phase_index[goal_changed] += 1

        # ── 2. normal env step ───────────────────────────────────────────────
        td = super()._step(tensordict)

        # ── 3. update reward EMA with this step's rewards ────────────────────
        mean_r = self._mean_reward_per_env(td)
        self._update_ema(mean_r)

        # ── 4. open adaptation windows for envs whose goal just changed ──────
        #       (snapshot *after* the step so the EMA includes pre-change info)
        self._record_goal_change(goal_changed)

        # ── 5. advance step counter for adapting envs & check reconvergence ──
        self._steps_since_change[self._adapting] += 1
        self._check_reconvergence()

        episode_changed = self._step_count >= self._max_steps
        td.set("goal_changed",        goal_changed)
        td.set("episode_changed",     episode_changed)
        td.set("total_steps",         self._total_steps.clone())
        td.set("episode_count",       self._episode_count.clone())
        td.set("adapting",            self._adapting.clone())
        td.set("steps_since_change",  self._steps_since_change.clone())
        return td

    # ── metrics ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_extra_metrics(self) -> dict:
        """
        Merges the parent's domain metrics with non-stationarity adaptation
        metrics.  All keys are available from a single ``get_extra_metrics()``
        call.

        Adaptation keys
        ---------------
        phase_index : float
            Mean phase index across envs.  Increments by 1 for each env every
            time its goal changes, so it is a direct count of goal-change
            events per env.  Use this as an x-axis when plotting adaptation
            curves — a rising value means the non-stationarity is active.
        mean_time_to_reconverge : float | None
            Average steps from goal change to recovery (≥95 % of pre-change
            reward EMA).  None until the first reconvergence is recorded.
        median_time_to_reconverge : float | None
            Median of the same distribution (less sensitive to outliers).
        reconverge_rate : float
            Fraction of goal-change events that led to a confirmed
            reconvergence so far (0.0 before any changes).
        mean_steps_since_change : float
            Average steps elapsed across envs still inside an open adaptation
            window.  0.0 when no env is currently adapting.
        n_goal_changes : int
        n_reconverged : int
        reward_ema : list[float]
            Live per-env reward EMA — useful for debugging thresholds.
        """
        metrics = super().get_extra_metrics()

        times = self._reconverge_times
        n_rec = len(times)
        n_chg = self._n_goal_changes

        mean_t = float(sum(times) / n_rec) if n_rec > 0 else None
        if n_rec > 0:
            sorted_t = sorted(times)
            mid = n_rec // 2
            median_t = (
                float(sorted_t[mid])
                if n_rec % 2 == 1
                else float((sorted_t[mid - 1] + sorted_t[mid]) / 2)
            )
        else:
            median_t = None

        reconverge_rate = n_rec / n_chg if n_chg > 0 else 0.0

        adapting_steps = self._steps_since_change[self._adapting].float()
        mean_since = adapting_steps.mean().item() if self._adapting.any() else 0.0

        metrics.update({
            "phase_index":               self._phase_index.float().mean().item(),
            "mean_time_to_reconverge":   mean_t,
            "median_time_to_reconverge": median_t,
            "reconverge_rate":           reconverge_rate,
            "mean_steps_since_change":   mean_since,
            "n_goal_changes":            n_chg,
            "n_reconverged":             n_rec,
            "reward_ema":                self._reward_ema.tolist(),
        })
        return metrics

    # backward-compat alias
    def get_adaptation_metrics(self) -> dict:
        """Alias for ``get_extra_metrics()``."""
        return self.get_extra_metrics()

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