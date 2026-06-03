from omegaconf import DictConfig

def build_algorithm(
    algo_name: str,
    env,
    cfg: DictConfig
):
    if algo_name == "mappo":
        from algos.mappo.algorithm import MAPPO
        return MAPPO(env, cfg)
    elif algo_name == "deep_ed":
        from algos.deeped.algorithm import DeepED
        return DeepED(env, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")