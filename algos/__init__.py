from omegaconf import DictConfig

def build_algorithm(
    algo_name: str,
    env,
    cfg: DictConfig
):
    if algo_name in ("mappo", "ippo"):
        from algos.mappo.algorithm import MAPPO
        return MAPPO(env, cfg)
    elif algo_name in ("deep_ed", "deep_ed_bnn", "deep_ed_replicator"):
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