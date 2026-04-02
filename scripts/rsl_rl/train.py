# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the target aiming task with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import os
import platform
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import target_aiming.tasks  # noqa: F401

RSL_RL_VERSION = "3.0.1"
DEFAULT_TASK = "Template-Target-Aiming-Direct-v0"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train the target aiming task with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL config entry point."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Number of policy training iterations.")
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # Phase 2+ uses TiledCamera: pass --enable_cameras when needed.
    # Phase 1 uses geometric projection only, no camera rendering required.
    return args_cli, hydra_args


args_cli, hydra_args = _parse_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _check_rsl_rl_version() -> str:
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) >= version.parse(RSL_RL_VERSION):
        return installed_version

    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]

    raise RuntimeError(
        "Please install the correct version of RSL-RL.\n"
        f"Existing version is: '{installed_version}' and required version is: '{RSL_RL_VERSION}'.\n"
        "To install the correct version, run:\n\n"
        f"\t{' '.join(cmd)}\n"
    )


installed_version = _check_rsl_rl_version()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with an RSL-RL PPO runner."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported on CPU. Use a CUDA device instead.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    if getattr(agent_cfg, "class_name", "OnPolicyRunner") != "OnPolicyRunner":
        raise ValueError("This project currently supports only the RSL-RL OnPolicyRunner training path.")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir_name}")
    if agent_cfg.run_name:
        log_dir_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir_name)

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = None

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
