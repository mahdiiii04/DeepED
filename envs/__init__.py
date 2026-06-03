from omegaconf import DictConfig

def make_env(
    env_type: str,
    cfg: DictConfig,
    seed: int
):
    if env_type == "matrix_games":
        from envs.matrix_games.envs import MatrixGameFactory
        return MatrixGameFactory(
            scenario=cfg.env.scenario_name,
            num_envs=cfg.env.num_envs,
            max_steps=cfg.env.max_steps,
            device=cfg.env.device,
            seed=seed,
        )
    else:
        raise NotImplementedError("Not yet")