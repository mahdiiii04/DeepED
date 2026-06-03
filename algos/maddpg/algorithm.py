import torch
from torch import nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import (
    AdditiveGaussianModule,
    EGreedyModule,
    ProbabilisticActor,
    TanhDelta,
    ValueOperator,
)
from torchrl.modules.distributions import OneHotCategorical
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from omegaconf import DictConfig

class MADDPG:

    name: str = "maddpg"

    def __init__(
        self,
        env,
        cfg: DictConfig
    ):
        self.env = env
        self.cfg = cfg
        device = cfg.train.device

        #───── detect action space type ───────────────────────────
        action_spec = env.full_action_spec_unbatched[env.action_key]
        self._discrete = isinstance(action_spec.space, CategoricalBox)
        n_actions = action_spec.shape[-1]

        #───── policy ─────────────────────────────────────────────
        policy_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
                n_agent_outputs=n_actions,
                n_agents=env.n_agents,
                centralized=False,
                share_params=cfg.model.shared_params,
                device=device,
                depth=cfg.model.depth,
                num_cells=cfg.model.num_cells,
                activation_class=nn.Tanh,
            )
        )

        if self._discrete:
            policy_module = TensorDictModule(
                policy_net,
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "logits")],
            )
        else:
            policy_module = TensorDictModule(
                policy_net,
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "param")],
            )

        if self._discrete:
            self.policy = ProbabilisticActor(
                policy_module,
                spec=env.full_action_spec_unbatched,
                in_keys=[("agents", "logits")],
                out_keys=[env.action_key],
                distribution_class=OneHotCategorical,
                return_log_prob=False,
            )
            self.policy_explore = TensorDictSequential(
                self.policy,
                EGreedyModule(
                    spec=env.full_action_spec_unbatched,
                    annealing_num_steps=int(cfg.collector.total_frames * (1 / 2)),
                    action_key=env.action_key,
                    eps_init=1.0,
                    eps_end=0.05,
                ),
            )
        else:
            self.policy = ProbabilisticActor(
                policy_module,
                spec=env.full_action_spec_unbatched,
                in_keys=[("agents", "param")],
                out_keys=[env.action_key],
                distribution_class=TanhDelta,
                distribution_kwargs={
                    "low": action_spec.space.low,
                    "high": action_spec.space.high,
                },
                return_log_prob=False,
            )
            self.policy_explore = TensorDictSequential(
                self.policy,
                AdditiveGaussianModule(
                    spec=env.full_action_spec_unbatched,
                    annealing_num_steps=int(cfg.collector.total_frames * (1 / 2)),
                    action_key=env.action_key,
                    device=cfg.train.device,
                ),
            )

        #───── critic ────────────────────────────────────────
        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1]
            + env.full_action_spec_unbatched[env.action_key].shape[
                -1
            ],
            n_agent_outputs=1,
            n_agents=env.n_agents,
            centralized=cfg.model.centralized_critic,
            share_params=cfg.model.shared_params,
            device=device,
            depth=cfg.model.depth,
            num_cells=cfg.model.num_cells,
            activation_class=nn.Tanh,
        )

        self.critic = ValueOperator(
            module=critic_net,
            in_keys=[("agents", "observation"), env.action_key],
            out_keys=[("agents", "state_action_value")],
        )

        #───── loss ───────────────────────────────────────────
        self.loss_module = DDPGLoss(
            actor_network=self.policy, value_network=self.critic, delay_value=True
        )

        self.loss_module.set_keys(
            state_action_value=("agents", "state_action_value"),
            reward=env.reward_key,
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.loss.gamma)
        self.target_net_updater = SoftUpdate(self.loss_module, eps=1 - cfg.loss.tau)

        #───── replay buffer ───────────────────────────────────
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg.buffer.memory_size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg.train.minibatch_size,
        )

        #───── optim ───────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr = cfg.train.lr,
        )

    #───── public interfaces ───────────────────────────────────

    def after_collect(
        self,
        tensordict_data
    ):
        self.current_frames = tensordict_data.numel()

    def pre_update(
        self,
        tensordict_data
    ):
        self.replay_buffer.extend(tensordict_data.reshape(-1))

    def update(self, tensordict_data):
        cfg = self.cfg
        loss_logs = []
        grad_norms = []                         

        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = self.replay_buffer.sample()
                loss_vals = self.loss_module(subdata)

                total_loss = (
                    loss_vals["loss_actor"]
                    + loss_vals["loss_value"]
                )
                total_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(),
                    cfg.train.max_grad_norm,
                )
                grad_norms.append(grad_norm.item())

                loss_logs.append({
                    "loss_actor": loss_vals["loss_actor"].item(),
                    "loss_value": loss_vals["loss_value"].item(),
                })

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.target_net_updater.step()

        return {
            "loss/actor":  sum(d["loss_actor"]  for d in loss_logs) / len(loss_logs),
            "loss/value":  sum(d["loss_value"]  for d in loss_logs) / len(loss_logs),
            "grad/norm":   sum(grad_norms) / len(grad_norms),
        }


    def post_update(
        self,
        tensordict_data
    ):
        self.policy_explore[1].step(frames=self.current_frames)