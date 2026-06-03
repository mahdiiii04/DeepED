import torch
from torch import nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer, VDNMixer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss

from omegaconf import DictConfig

class QMIX:

    name: str = "qmix"

    def __init__(
        self,
        env,
        cfg: DictConfig
    ):
        self.env = env
        self.cfg = cfg
        device = cfg.train.device

        #───── policy ─────────────────────────────────────────────
        net = nn.Sequential(
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

        module = TensorDictModule(
            net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "action_value")],
        )

        self.value_module = QValueModule(
            action_value_key=("agents", "action_value"),
            out_keys=[
                env.action_key,
                ("agents", "action_value"),
                ("agents", "chosen_action_value"),
            ],
            spec=env.full_action_spec_unbatched,
            action_space=None,
        )

        self.qnet = SafeSequential(module, self.value_module)

        self.policy = TensorDictSequential(
            self.qnet,
            EGreedyModule(
                eps_init=cfg.loss.eps_init,
                eps_end=cfg.loss.eps_end,
                annealing_num_steps=int(cfg.collector.total_frames * cfg.loss.eps_anneal_frac),
                action_key=env.action_key,
                spec=env.full_action_spec_unbatched,
            ),
        )
        if cfg.loss.mixer_type == "qmix":
            self.mixer = TensorDictModule(
                module=QMixer(
                    state_shape=env.observation_spec_unbatched["agents", "observation"].shape,
                    mixing_embed_dim=32,
                    n_agents=env.n_agents,
                    device=cfg.train.device,
                ),
                in_keys=[("agents", "chosen_action_value"), ("agents", "observation")],
                out_keys=["chosen_action_value"],
            )
        elif cfg.loss.mixer_type == "vdn":
            self.mixer = TensorDictModule(
                module=VDNMixer(
                    n_agents=env.n_agents,
                    device=cfg.train.device,
                ),
                in_keys=[("agents", "chosen_action_value")],
                out_keys=["chosen_action_value"],
            )
        else:
            raise ValueError(f"Unknown mixer_type: {cfg.loss.mixer_type}. Use 'qmix' or 'vdn'.")

        #───── loss ───────────────────────────────────────────
        self.loss_module = QMixerLoss(self.qnet, self.mixer, delay_value=True)
        self.loss_module.set_keys(
            action_value=("agents", "action_value"),
            local_value=("agents", "chosen_action_value"),
            global_value="chosen_action_value",
            action=env.action_key,
            done="done",
            terminated="terminated",
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
        # Store original per-agent episode reward for logging in train.py
        # (we keep it instead of deleting)
        if ("next", "agents", "episode_reward") in tensordict_data.keys(True):
            per_agent_episode_reward = tensordict_data.get(("next", "agents", "episode_reward")).clone()
        else:
            per_agent_episode_reward = None

        # Global reward (for QMIX loss)
        tensordict_data.set(
            ("next", "reward"),
            tensordict_data.get(("next", self.env.reward_key)).mean(-2),
        )
        del tensordict_data["next", self.env.reward_key]

        # Global episode reward (for QMIX loss)
        tensordict_data.set(
            ("next", "episode_reward"),
            tensordict_data.get(("next", "agents", "episode_reward")).mean(-2),
        )

        # IMPORTANT: Restore per-agent episode_reward so train.py doesn't break
        if per_agent_episode_reward is not None:
            tensordict_data.set(
                ("next", "agents", "episode_reward"),
                per_agent_episode_reward
            )

        # Flatten done/terminated to top-level so they match reward's shape
        tensordict_data.set(
            ("next", "done"),
            tensordict_data.get(("next", "agents", "done")).any(-2),
        )
        tensordict_data.set(
            ("next", "terminated"),
            tensordict_data.get(("next", "agents", "terminated")).any(-2),
        )

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

                total_loss = loss_vals["loss"]
                total_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(),
                    cfg.train.max_grad_norm,
                )
                grad_norms.append(grad_norm.item())      # ← plain float

                loss_logs.append({
                    "loss": loss_vals["loss"].item(),
                })

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.target_net_updater.step()

        return {
            "loss/total": sum(d["loss"] for d in loss_logs) / len(loss_logs),
            "grad/norm":      sum(grad_norms) / len(grad_norms),
        }


    def post_update(
        self,
        tensordict_data
    ):
        self.policy[1].step(frames=self.current_frames)