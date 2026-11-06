import numpy as np
import torch
from tensordict import TensorDict


@torch.no_grad()
def policy_state_probs(env, policy, *, policy_type: str = "actor", temperature: float = 0.0) -> np.ndarray:
    """
    Action probabilities of `policy` on EVERY joint state of a small grid env (the env must expose
    state_positions() and obs_from_positions(), e.g. TeamGoalShiftEnv).
    Returns float64 numpy [S, n_agents, n_actions].

    policy_type "actor": the policy's own distribution.
    policy_type "qnet" : greedy one-hot if temperature <= 0 (default), else softmax(Q / temperature).
    """
    if hasattr(env, "state_observations"):          # e.g. RingGameEnv: observations of every state
        obs = env.state_observations()
    else:                                           # grid envs: build observations from joint positions
        pos = torch.as_tensor(np.asarray(env.state_positions()), dtype=torch.long, device=env.device)
        obs = env.obs_from_positions(pos)
    td = TensorDict(
        {"agents": {"observation": obs}}, batch_size=[obs.shape[0]], device=env.device
    )
    if policy_type == "actor":
        probs = policy.get_dist(td).probs
    elif policy_type == "qnet":
        vals = policy(td).get(("agents", "action_value"))
        if temperature <= 0:
            probs = torch.nn.functional.one_hot(vals.argmax(-1), vals.shape[-1]).to(vals.dtype)
        else:
            probs = torch.softmax(vals / temperature, dim=-1)
    else:
        raise ValueError(f"Unknown policy_type '{policy_type}'. Use 'actor' or 'qnet'.")
    return probs.detach().double().cpu().numpy()
