from torchrl.envs import Transform
from tensordict import TensorDictBase
import torch

class MinMaxRewardTransform(Transform):
    """Simple per-batch min-max scaling:
    scaled_reward = (reward - r_min) / (r_max - r_min + 1e-8)
    """

    def __init__(self, epsilon: float = 1e-8, clip_range: float = 10.0):
        super().__init__(in_keys=["reward"], out_keys=["reward"])
        self.epsilon = epsilon
        self.clip_range = clip_range

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        reward = tensordict.get("reward", None)
        if reward is None or reward.numel() == 0:
            return tensordict

        r_min = reward.min()
        r_max = reward.max()

        # Your exact equation
        scaled_reward = (reward - r_min) / (r_max - r_min + self.epsilon)

        if self.clip_range is not None:
            scaled_reward = torch.clamp(scaled_reward, -self.clip_range, self.clip_range)

        tensordict.set("reward", scaled_reward)
        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        """Important: must return the reset tensordict"""
        return tensordict_reset