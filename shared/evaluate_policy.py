import torch
import numpy as np

def evaluate_policy(env_test, policy, n_episodes=10):
    """
    Evaluate the policy on the test environment.
    
    Returns:
        mean_reward (float): Mean episode reward across episodes.
    """
    rewards = []
    for _ in range(n_episodes):
        td = env_test.reset()
        done = False
        episode_reward = 0.0
        step = 0
        while not done and step < env_test.max_steps:
            with torch.no_grad():
                td = policy(td)
            td = env_test.step(td)
            reward = td.get(("next", "agents", "reward")).mean().item()
            episode_reward += reward
            done_val = td.get(("next", "agents", "done"), default=None)
            if done_val is None:
                done_val = td.get(("next", "done"))
            done = done_val.any().item()
            td = td.get("next").clone()
            step += 1
        rewards.append(episode_reward)
    return float(np.mean(rewards))