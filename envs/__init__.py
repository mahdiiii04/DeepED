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
    elif env_type == "gridworld":
        from envs.gridworld.envs import GridWorldFactory
        return GridWorldFactory(
            scenario=cfg.env.scenario_name,
            num_envs=cfg.env.num_envs,
            max_steps=cfg.env.max_steps,
            device=cfg.env.device,
            pref_phase_length=cfg.env.get("pref_phase_length", 550),
            goal_phase_length=cfg.env.get("goal_phase_length", 550),
            seed=seed,
        )
    elif env_type == "vmas":
        from torchrl.envs.libs.vmas import VmasEnv
        return VmasEnv(
            scenario=cfg.env.scenario_name,
            num_envs=cfg.env.num_envs,
            continuous_actions=False,
            max_steps=cfg.env.max_steps,
            device=cfg.env.device,
            n_agents=3,
            seed=seed,
        )
    else:
        raise NotImplementedError("Not yet")