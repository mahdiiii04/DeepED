import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.envs import RewardSum, TransformedEnv, Compose
from torch.utils.tensorboard import SummaryWriter

from envs import make_env as _make_env_dispatch
from algos import build_algorithm
from shared import ResultsDB, extract_avg_policy, evaluate_policy, DoneTransform
from envs.matrix_games import compute_nash_conv, MinMaxRewardTransform

def _resolve_cfg(cfg: DictConfig):
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device = cfg.train.device
    cfg.env.num_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch
    return cfg

def make_env(cfg: DictConfig, seed: int) -> TransformedEnv:
    base_env = _make_env_dispatch(cfg.env_type, cfg, seed)
    
    transforms = [
        RewardSum(
            in_keys=[base_env.reward_key],
            out_keys=[("agents", "episode_reward")]
        )
    ]

    if cfg.env_type == "matrix_games" and cfg.env.scenario_name == "biased_rps":
        transforms.append(MinMaxRewardTransform())
    
    env = TransformedEnv(base_env, Compose(*transforms))
    return env

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
    postproc = None
    if cfg.env_type == "vmas":
        postproc = DoneTransform(
            reward_key=env.reward_key,
            done_keys=env.done_keys,
        )
    collector = Collector(
        env,
        algo.policy,
        device=cfg.train.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=postproc,
    )

    #──── main loop ────────────────────────────────────────────────
    total_frames = 0
    
    for i, tensordict_data in enumerate(collector):
        algo.after_collect(tensordict_data)
        algo.pre_update(tensordict_data)

        t_update_start = time.time()
        loss_metrics = algo.update(tensordict_data)
        t_update_end = time.time()

        algo.post_update(tensordict_data)

        collector.update_policy_weights_()

        total_frames += tensordict_data.numel()
        global_step = total_frames

        if cfg.env_type in ("gridworld", "matrix_games"):
            episode_r = (
                tensordict_data
                .get(("next", "agents", "episode_reward"))
                .reshape(cfg.env.num_envs, cfg.env.max_steps, env.n_agents, 1)
            )

            mean_episode_reward = episode_r[:, -1].mean().item()
        else:
            done = tensordict_data.get(("agents", "done"))
            final_rewards = tensordict_data.get(("agents", "episode_reward"))[done]
            mean_episode_reward = final_rewards.mean().item()

        scalar_metrics = {
            "reward/mean_episode_reward": mean_episode_reward,
            "time/update": t_update_end - t_update_start,
            **loss_metrics,
            **(env.get_extra_metrics() if hasattr(env, "get_extra_metrics") else {}),
        }
        if cfg.env_type == "matrix_games":
            avg_policy = extract_avg_policy(
                env,
                algo.policy,
                policy_type=cfg.get("policy_type", "actor"),
            )
            nash = compute_nash_conv(
                env,
                avg_policy,
            )
            db.log_policy(run_id, global_step, avg_policy)

            scalar_metrics["nash/nash_conv"] = nash

        #──── logging ────────────────────────────────────────────
        db.log_metrics(run_id, global_step, scalar_metrics)

        for k, v in scalar_metrics.items():
            writer.add_scalar(k, v, global_step=global_step)

        #──── evaluation ─────────────────────────────────────────
        if (i % cfg.eval.frequency == 0 or i == cfg.collector.n_iters - 1):
            eval_reward = evaluate_policy(env_test=env_test, policy=algo.policy)
            db.log_metrics(
                run_id, global_step, {"eval/mean_episode_reward": eval_reward}
            )
            writer.add_scalar("eval/mean_episode_reward", eval_reward, global_step=global_step)
            torchrl_logger.info(f"Eval reward: {eval_reward:.3f}")

        torchrl_logger.info(f"Iteration {i} | Frames {total_frames:>8d} | Mean Reward {mean_episode_reward:.3f}")
        
    #──── teardown ────────────────────────────────────────────────
    writer.close()
    collector.shutdown()
    db.close()
    for e in (env, env_test):
        if not e.is_closed:
            env.close()

    torchrl_logger.info(f"Done. Results in {db_path} (run_id={run_id})")

if __name__ == "__main__":
    train()