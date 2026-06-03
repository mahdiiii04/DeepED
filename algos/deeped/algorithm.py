import torch
from torch import nn

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from algos.deeped.deeped_loss import DeepEDLoss

from omegaconf import DictConfig

class DeepED:
    name: str = "deep_ed"

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
            n_agent_outputs=env.full_action_spec_unbatched[env.action_key].shape[-1],
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
            out_keys=["q_value"],
        )

        #───── loss ───────────────────────────────────────────

        self.loss_module = DeepEDLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            entropy_coeff=cfg.loss.entropy_eps,
            alpha=cfg.loss.alpha,
            gamma=cfg.loss.gamma,
        )

        self.loss_module.set_keys(
            reward=env.reward_key,
            action=env.action_key,
            done=("agents", "done"),
            terminated=("agents", "terminated"),
            sample_log_prob=("agents", "action_log_prob"),
        )

        #───── replay buffer ───────────────────────────────────
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg.buffer.memory_size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg.train.minibatch_size,
        )

        #───── optim ───────────────────────────────────────────
        if self.loss_module.functional:
            self.actor_params = list(self.loss_module.actor_network_params.values(True, True))
            self.critic_params = list(self.loss_module.critic_network_params.values(True, True))
        else:
            self.actor_params = list(self.loss_module.actor_network.parameters())
            self.critic_params = list(self.loss_module.critic_network.parameters())

        self.actor_optim = torch.optim.Adam(self.actor_params, lr=cfg.train.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_params, lr=cfg.train.critic_lr)

    #───── public interfaces ───────────────────────────────────

    def after_collect(
        self,
        tensordict_data
    ):
        pass

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

                # --- Critic Step ---
                self.critic_optim.zero_grad()
                loss_vals["loss_critic"].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.critic_params, cfg.train.max_grad_norm)
                self.critic_optim.step()

                # --- Actor Step ---
                self.actor_optim.zero_grad()
                actor_loss = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_params, cfg.train.max_grad_norm)
                self.actor_optim.step()

                self.loss_module.soft_update_target(tau=cfg.train.tau)

                total_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.actor_params + self.critic_params
                    if p.grad is not None
                ) ** 0.5

                grad_norms.append(total_norm)      

                loss_logs.append({
                    "loss_objective": loss_vals["loss_objective"].item(),
                    "loss_critic":    loss_vals["loss_critic"].item(),
                    "loss_entropy":   loss_vals["loss_entropy"].item(),
                })

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
        self.loss_module.soft_update_avg_actor(tau=0.02)