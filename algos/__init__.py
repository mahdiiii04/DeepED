from omegaconf import DictConfig

def build_algorithm(
    algo_name: str,
    env,
    cfg: DictConfig
):
    if algo_name == "mappo":
        from algos.mappo.algorithm import MAPPO
        return MAPPO(env, cfg)
    else:
        pass