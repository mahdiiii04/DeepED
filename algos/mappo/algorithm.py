import torch
from torch import nn

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from omegaconf import DictConfig

class MAPPO:

    name: str = "mappo"

    def __init__(
        self,
        env,
        cfg: DictConfig
    ):
        self.env = env
        self.cfg = cfg
        device = cfg.train.device

        #───── policy ─────────────────────────────────────────────
        policy_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
                n_agent_outputs=env.full_action_spec_unbatched[env.action_key].shape[-1],
                n_agents=env.n_agents,
                centralized=False,
                share_params=cfg.model.shared_params,
                device=device,
                depth=cfg.model.depth,
                num_cells=cfg.model.num_cells,
                activation_class=nn.Tanh,
            )
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )

        self.policy = ProbabilisticActor(
            policy_module,
            spec=env.full_action_spec_unbatched,
            in_keys=[("agents", "logits")],
            out_keys=[env.action_key],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )

        #───── critic ────────────────────────────────────────
        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
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
            critic_net,
            in_keys=[("agents", "observation")],
        )

        #───── loss ───────────────────────────────────────────
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=cfg.loss.clip_epsilon,
            entropy_coeff=cfg.loss.entropy_eps,
            normalize_advantage=False,
            separate_agent_loss=True,
        )

        self.loss_module.set_keys(
            reward=env.reward_key,
            action=env.action_key,
            done=("agents", "done"),
            terminated=("agents", "terminated"),
            sample_log_prob=("agents", "action_log_prob"),
        )

        self.loss_module.make_value_estimator(
            ValueEstimators.GAE,
            gamma=cfg.loss.gamma,
            lmbda=cfg.loss.lmbda,
        )

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

    def compute_advantage(
        self,
        tensordict_data
    ):
        with torch.no_grad():
            self.loss_module.value_estimator(
                tensordict_data,
                params=self.loss_module.critic_network_params,
                target_params=self.loss_module.target_critic_network_params,
            )

    def after_collect(
        self,
        tensordict_data
    ):
        self.compute_advantage(tensordict_data)

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
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                total_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(),
                    cfg.train.max_grad_norm,
                )
                grad_norms.append(grad_norm.item())      # ← plain float

                loss_logs.append({
                    "loss_objective": loss_vals["loss_objective"].item(),
                    "loss_critic":    loss_vals["loss_critic"].item(),
                    "loss_entropy":   loss_vals["loss_entropy"].item(),
                })

                self.optimizer.step()
                self.optimizer.zero_grad()

        return {
            "loss/objective": sum(d["loss_objective"] for d in loss_logs) / len(loss_logs),
            "loss/critic":    sum(d["loss_critic"]    for d in loss_logs) / len(loss_logs),
            "loss/entropy":   sum(d["loss_entropy"]   for d in loss_logs) / len(loss_logs),
            "grad/norm":      sum(grad_norms) / len(grad_norms),
        }


    def post_update(
        self,
        tensordict_data
    ):
        pass