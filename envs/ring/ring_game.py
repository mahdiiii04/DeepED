import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Composite, OneHot, Unbounded

from envs.ring import ring_oracle as ro


class RingGameEnv(EnvBase):
    """
    Stateful, identical-interest (=> exact Markov potential) game built to TEST THE THEORY.

    * S states on a ring, 2 agents, M actions each; observation = one-hot state (the phase is hidden).
    * Shared stage reward  payoff_scale * ( v * [both play tau] + bridge * [exactly one plays tau] ),
      tau = (theta + s) % M; the ring advances only when BOTH play tau. (tau, tau) is the unique
      equilibrium of every state (checked numerically in ring_oracle tests).
    * Exogenous non-stationarity: every ``phase_length`` collector steps the phase offset theta changes
      ("cyclic" or "random", never repeating). The size of each change is
      v_t = payoff_scale * v (sup-norm reward change) and is exposed as ``env/v_t`` with its running sum
      ``env/V_T`` (the variation budget of the theory).
    * Everything is exact: exact_tables() gives the transition/reward tables of the CURRENT phase, so
      NashConv, the potential, the true Q-values (critic error) and the concentrability constant can be
      computed without sampling (see shared/ring_diag.py).
    Small payoffs keep the critic's action range B below the step-size limit 1/(alpha (M-1)) for
    alpha up to ~0.4, so the update that runs is the raw Smith update of the theory.
    """

    n_agents = 2

    def __init__(
            self,
            num_envs: int = 1,
            max_steps: int = 100,
            device: str = "cpu",
            seed: int = 0,
            n_states: int = 4,
            n_actions: int = 3,
            v: float = 1.0,
            bridge: float = 0.5,
            payoff_scale: float = 1.0,
            phase_length: int = 2000,
            phase_order: str = "random",
            phase_seed: int = 0,
            **kwargs,
    ):
        if not (v > bridge > 0):
            raise ValueError("need v > bridge > 0 so that (tau, tau) is the unique equilibrium")
        if phase_order not in ("cyclic", "random"):
            raise ValueError(f"phase_order must be 'cyclic' or 'random', got {phase_order!r}")
        self.n_states = n_states
        self.n_actions = n_actions
        self._v = v
        self._bridge = bridge
        self._scale = payoff_scale
        self._phase_length = phase_length
        self._phase_order = phase_order

        super().__init__(batch_size=torch.Size([num_envs]), device=device)

        self._num_envs = num_envs
        self._max_steps = max_steps
        self.register_buffer("_state", torch.zeros(num_envs, dtype=torch.long, device=self.device))
        self.register_buffer("_step_count", torch.zeros(num_envs, dtype=torch.long, device=self.device))
        self.register_buffer("_total_steps", torch.zeros(num_envs, dtype=torch.long, device=self.device))
        self._theta = 0
        self._v_last = 0.0
        self._V_T = 0.0
        self._phase_rng = torch.Generator().manual_seed(phase_seed)
        self._set_tables(0)
        self._make_specs()
        self._set_seed(seed)

    # ------------------------------------------------------------------ tables / phases
    def _set_tables(self, theta: int) -> None:
        ns, R = ro.build_tables(self.n_states, self.n_actions, theta, self._v, self._bridge, self._scale)
        self._ns_np, self._R_np = ns, R
        self._ns_t = torch.as_tensor(ns, dtype=torch.long, device=self.device)
        self._R_t = torch.as_tensor(R, dtype=torch.float32, device=self.device)

    def _next_phase(self, cur: int) -> int:
        M = self.n_actions
        if self._phase_order == "cyclic":
            return (cur + 1) % M
        others = [a for a in range(M) if a != cur]
        return others[int(torch.randint(len(others), (1,), generator=self._phase_rng).item())]

    # ------------------------------------------------------------------ specs
    def _make_specs(self) -> None:
        ne, n, a, o = self._num_envs, self.n_agents, self.n_actions, self.n_states
        self.observation_spec = Composite(
            {"agents": Composite(
                {"observation": Unbounded(shape=(ne, n, o), dtype=torch.float32)}, shape=(ne, n))},
            shape=(ne,),
        )
        self.action_spec = Composite(
            {"agents": Composite(
                {"action": OneHot(n=a, shape=(ne, n, a), dtype=torch.long)}, shape=(ne, n))},
            shape=(ne,),
        )
        self.reward_spec = Composite(
            {"agents": Composite(
                {"reward": Unbounded(shape=(ne, n, 1), dtype=torch.float32)}, shape=(ne, n))},
            shape=(ne,),
        )
        self.done_spec = Composite(
            {"agents": Composite(
                {"done": Bounded(low=0, high=1, shape=(ne, n, 1), dtype=torch.bool),
                 "terminated": Bounded(low=0, high=1, shape=(ne, n, 1), dtype=torch.bool)},
                shape=(ne, n))},
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
    def max_steps(self):
        return self._max_steps

    # ------------------------------------------------------------------ dynamics
    def _obs(self) -> torch.Tensor:
        oh = torch.nn.functional.one_hot(self._state, self.n_states).float()
        return oh[:, None, :].expand(self._num_envs, self.n_agents, self.n_states).clone()

    def _phase_tensor(self) -> torch.Tensor:
        return torch.full((self._num_envs,), self._theta, dtype=torch.long, device=self.device)

    def _reset(self, tensordict=None):
        ne, dev = self._num_envs, self.device
        if tensordict is not None and "_reset" in tensordict.keys():
            mask = tensordict.get("_reset").reshape(ne)
        else:
            mask = torch.ones(ne, dtype=torch.bool, device=dev)
        self._step_count[mask] = 0
        fresh = torch.randint(0, self.n_states, (ne,), device=dev)      # uniform start = rho of the oracle
        self._state[mask] = fresh[mask]
        return TensorDict(
            {
                "agents": TensorDict({"observation": self._obs()},
                                     batch_size=[ne, self.n_agents], device=dev),
                "phase": self._phase_tensor(),
                "phase_changed": torch.zeros(ne, dtype=torch.bool, device=dev),
            },
            batch_size=[ne], device=dev,
        )

    def _step(self, tensordict: TensorDict):
        ne, dev, N = self._num_envs, self.device, self.n_agents
        action_raw = tensordict.get(("agents", "action"))
        actions = action_raw.argmax(dim=-1) if action_raw.dim() == 3 else action_raw.long()
        a0, a1 = actions[:, 0], actions[:, 1]

        s = self._state.clone()
        r = self._R_t[s, a0, a1]                                  # shared reward, old phase
        self._state.copy_(self._ns_t[s, a0, a1])
        self._step_count.add_(1)
        self._total_steps.add_(1)
        done_env = self._step_count >= self._max_steps
        done_agent = done_env[:, None, None].expand(ne, N, 1)

        changed = torch.zeros(ne, dtype=torch.bool, device=dev)
        if (self._total_steps[0] % self._phase_length == 0) and (self._total_steps[0] > 0):
            old_R = self._R_np
            self._theta = self._next_phase(self._theta)
            self._set_tables(self._theta)
            self._v_last = ro.change_size(old_R, self._R_np)
            self._V_T += self._v_last
            changed = torch.ones(ne, dtype=torch.bool, device=dev)

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": self._obs(),
                        "reward": r[:, None, None].expand(ne, N, 1).clone(),
                        "done": done_agent.clone(),
                        "terminated": done_agent.clone(),
                    },
                    batch_size=[ne, N], device=dev),
                "phase": self._phase_tensor(),
                "phase_changed": changed,
            },
            batch_size=[ne], device=dev,
        )

    def _set_seed(self, seed) -> None:
        torch.manual_seed(seed)

    # ------------------------------------------------------------------ exact evaluation support
    def exact_tables(self):
        """(next_state [S,M,M], reward [S,M,M]) of the CURRENT phase (numpy)."""
        return self._ns_np.copy(), self._R_np.copy()

    def state_observations(self) -> torch.Tensor:
        """Observations of every state, [S, n_agents, n_states] (one-hot)."""
        S = self.n_states
        return torch.eye(S, device=self.device)[:, None, :].expand(S, self.n_agents, S).clone()

    def get_extra_metrics(self) -> dict:
        return {"env/phase": float(self._theta), "env/v_t": float(self._v_last), "env/V_T": float(self._V_T)}
