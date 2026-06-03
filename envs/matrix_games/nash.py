import torch
from tensordict import TensorDict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from envs.matrix_games.envs import MatrixGameFactory

@torch.no_grad()
def compute_nash_conv(
    env,
    avg_pi: torch.Tensor,
    *,
    device: str | torch.device | None = None
) -> float:
    
    if device is None:
        device = env.device
 
    avg_pi = avg_pi.to(device)
    payoff = env._payoff.to(device)  
 
    n_agents = env.n_agents

    u = torch.zeros(n_agents, device=device)
    u[0] = torch.einsum("a,ab,b->", avg_pi[0], payoff[0], avg_pi[1])
    u[1] = torch.einsum("a,ab,b->", avg_pi[0], payoff[1], avg_pi[1])

    br_0 = (payoff[0] @ avg_pi[1]).max() 
    br_1 = (avg_pi[0] @ payoff[1]).max()  
 
    nash_conv = (br_0 - u[0] + br_1 - u[1]).item()
 
    return nash_conv
