from torchrl.envs import Transform
from tensordict import TensorDictBase
import torch

class MinMaxRewardTransform(Transform):
    def __init__(self, epsilon: float = 1e-8, clip_range: float = 10.0):
        super().__init__(
            in_keys=[("agents", "reward")],  
            out_keys=[("agents", "reward")],
        )
        self.epsilon = epsilon
        self.clip_range = clip_range

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        reward = tensordict.get(("agents", "reward"), None)
        if reward is None or reward.numel() == 0:
            return tensordict

        r_min = reward.min()
        r_max = reward.max()

        scaled_reward = (reward - r_min) / (r_max - r_min + self.epsilon)

        if self.clip_range is not None:
            scaled_reward = torch.clamp(scaled_reward, -self.clip_range, self.clip_range)

        tensordict.set(("agents", "reward"), scaled_reward)
        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return tensordict_reset