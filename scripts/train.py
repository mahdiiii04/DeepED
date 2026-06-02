import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.envs import RewardSum, TransformedEnv
from torch.utils.tensorboard import SummaryWriter

from envs import make_env as _make_env_dispatch
from algos import build_algorithm
from shared import ResultsDB

def _resolve_cfg(cfg: DictConfig):
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device = cfg.train.device
    cfg.env.num_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch
    return cfg

def make_env(cfg: DictConfig, seed: int) -> TransformedEnv:
    env = _make_env_dispatch(cfg.env_type, cfg, seed)
    return TransformedEnv(
        env,
        RewardSum(
            in_keys=[env.reward_key],
            out_keys=[("agents", "episode_reward")]
        )
    )

@hydra.main(version_base="1.1", config_path=None)
def train(cfg: DictConfig):

    cfg = _resolve_cfg(cfg)

    env_type = cfg.env_type
    algo_name = cfg.algo_name

    #──── logging ────────────────────────────────────────────────
    orig_cwd = hydra.utils.get_original_cwd()
    db_path = os.path.join(orig_cwd, "outputs", cfg.experiment_name, "results.db")
    db = ResultsDB(db_path).connect()
    run_id = db.upsert_run(
        env_type, cfg.env.scenario_name, algo_name, cfg.seed, cfg
    )

    log_dir = os.path.join(
        orig_cwd, "outputs", cfg.experiment_name,
        "tb_logs", algo_name, f"seed{cfg.seed}"
    )
    writer = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"DB : {db_path} (run_id={run_id})")
    torchrl_logger.info(f"TensorBoard logs: {log_dir}")

    #──── environments ────────────────────────────────────────────
    env = make_env(cfg, cfg.seed)
    env_test = make_env(cfg, cfg.seed)

    #──── algorithm ────────────────────────────────────────────────
    algo = build_algorithm(algo_name, env, cfg)

    #──── data pipeline ────────────────────────────────────────────
    collector = Collector(
        env,
        algo.policy,
        device=cfg.train.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
    )

    #──── main loop ────────────────────────────────────────────────
    total_frames = 0
    
    for i, tensordict_data in enumerate(collector):
        t_collect_end = time.time()

        algo.after_collect(tensordict_data)
        algo.pre_update(tensordict_data)

        t_update_start = time.time()
        loss_metrics = algo.update(tensordict_data)
        t_update_end = time.time()

        algo.post_update(tensordict_data)

        collector.update_policy_weights_()

        total_frames += tensordict_data.numel()
        global_step = total_frames

        episode_r = (
            tensordict_data
            .get(("next", "agents", "episode_reward"))
            .reshape(cfg.env.num_envs, cfg.env.max_steps, env.n_agents, 1)
        )

        mean_episode_reward = episode_r[:, -1].mean().item()

        scalar_metrics = {
            "reward/mean_episode_reward": mean_episode_reward,
            "time/update": t_update_end - t_update_start,
            **loss_metrics,
            **env.get_extra_metrics(),
        }

        db.log_metric(run_id, global_step, scalar_metrics)
        
    #──── teardown ────────────────────────────────────────────────
    writer.close()
    collector.shutdown()
    db.close()
    for e in (env, env_test):
        if not e.is_closed:
            env.close()

    torchrl_logger.info(f"Done. Results in {db_path} (run_id={run_id})")