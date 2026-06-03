import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

#──── helpers ────────────────────────────────────────────

def _rollout(
    env, 
    policy,
) -> tuple[TensorDict, int, int]:
    
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.rollout(
            max_steps=env.max_steps,
            policy=policy,
            auto_reset=True,
            break_when_any_done=False,
            tensordict=env.reset(),
        )
    num_episodes = td.batch_size[0]
    max_steps = env.max_steps
    return td, num_episodes, max_steps

def _flat_obs(
    td: TensorDict,
    n_agents: int,
    device
) -> TensorDict:
    observation = td.get(("agents", "observation"))
    flat = observation.reshape(-1, n_agents, observation.shape[-1])
    return TensorDict(
        {"agents": {"observation": flat}},
        batch_size=[flat.shape[0]],
        device=device,
    )


#──── public ───────────────────────────────────────────────────
@torch.no_grad()
def extract_avg_policy(
    env, 
    policy,
    *,
    policy_type: str = "actor",
    temperature: float = 1.0,
    device: str | torch.device | None = None
) -> torch.Tensor:
    
    if device is None:
        device = env.device

    n_agents = env.n_agents
    n_actions = env.n_actions

    td, num_episodes, max_steps = _rollout(env, policy)
    input_td = _flat_obs(td, n_agents, device)

    if policy_type == "actor":
        dist  = policy.get_dist(input_td)
        probs = dist.probs                                 
    elif policy_type == "qnet":
        out_td      = policy(input_td)
        action_vals = out_td.get(("agents", "action_value"))
        probs       = torch.softmax(action_vals / temperature, dim=-1)
    else:
        raise ValueError(f"Unknown policy_type '{policy_type}'. Use 'actor' or 'qnet'.")
    
    probs = probs.reshape(num_episodes, max_steps, n_agents, n_actions)
    avg_pi = probs.mean(dim=(0, 1))

    return avg_pi.clone().cpu()