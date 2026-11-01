from omegaconf import DictConfig

_DEEPED_NAMES = (
    "deep_ed",
    "deep_ed_bnn",
    "deep_ed_replicator",
    "deep_ed_softmax",   # ablation: exponential-weights target
    "deep_ed_safe",      # ablation: Smith with per-state step so the clamp never acts
)

def build_algorithm(
    algo_name: str,
    env,
    cfg: DictConfig
):
    if algo_name in ("mappo", "ippo"):
        from algos.mappo.algorithm import MAPPO
        return MAPPO(env, cfg)
    elif algo_name in _DEEPED_NAMES:
        from algos.deeped.algorithm import DeepED
        return DeepED(env, cfg)
    elif algo_name == "maddpg":
        from algos.maddpg.algorithm import MADDPG
        return MADDPG(env, cfg)
    elif algo_name in ("qmix", "vdn"):
        from algos.qmix.algorithm import QMIX
        return QMIX(env, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
