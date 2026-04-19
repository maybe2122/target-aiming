#!/usr/bin/env python3
"""Dump RGB observations from the target-aiming env with the projected
target pixel overlaid. Useful to verify that:
  - the camera actually sees the car at reset,
  - the pinhole projection used by the reward matches where the car really is,
  - the target's size in pixels is reasonable (i.e. the CNN has something to latch onto).

Usage:
    ./isaaclab.sh -p scripts/diagnose/save_obs_images.py \
        --task Isaac-Target-Aiming-Direct-v0 --num_envs 16 --steps 40

Writes:
    outputs/diagnose/<ts>/obs_reset.png      -- 4x4 grid at t=0 (just after reset)
    outputs/diagnose/<ts>/obs_grid.png       -- 4x4 grid at a later step
    outputs/diagnose/<ts>/env0_seq.png       -- time series for env 0
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-Target-Aiming-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--out_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from datetime import datetime

import gymnasium as gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

import target_aiming.tasks  # noqa: F401


def make_env():
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    holder = {}

    @hydra_task_config(args_cli.task, args_cli.agent)
    def _grab(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        holder["env_cfg"] = env_cfg
        holder["agent_cfg"] = agent_cfg

    _grab()
    env = gym.make(args_cli.task, cfg=holder["env_cfg"], render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=holder["agent_cfg"].clip_actions)
    return env, holder["env_cfg"]


def grab(env):
    """Return (rgb (N,H,W,3) float[0,1], u (N,), v (N,), visible (N,) bool)."""
    unwrapped = env.unwrapped
    unwrapped._compute_intermediate_values()
    rgb_raw = unwrapped.camera.data.output["rgb"]
    rgb = (rgb_raw[..., :3].float() / 255.0).detach().cpu().numpy()
    u = unwrapped._target_u.detach().cpu().numpy()
    v = unwrapped._target_v.detach().cpu().numpy()
    vis = unwrapped._target_visible.detach().cpu().numpy()
    return rgb, u, v, vis


def save_grid(rgb, u, v, vis, out_path, cols=4, title=""):
    N = rgb.shape[0]
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= N:
            continue
        ax.imshow(np.clip(rgb[i], 0, 1))
        color = "lime" if vis[i] else "red"
        if 0 <= u[i] < rgb.shape[2] and 0 <= v[i] < rgb.shape[1]:
            ax.add_patch(mpatches.Circle((u[i], v[i]), 6, fill=False, color=color, linewidth=2))
        ax.text(5, 15, f"env{i} {'V' if vis[i] else 'X'}", color=color,
                fontsize=10, weight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    env, env_cfg = make_env()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args_cli.out_dir or os.path.join("outputs", "diagnose", f"{ts}_obs")
    os.makedirs(out_dir, exist_ok=True)

    # Drive env forward a tiny bit so _compute_intermediate_values has latest data
    obs = env.get_observations()
    n_actions = env_cfg.action_space
    num_envs = env.unwrapped.num_envs

    # Initial frame (after env was constructed, pre-first-step)
    rgb0, u0, v0, vis0 = grab(env)
    save_grid(rgb0, u0, v0, vis0,
              os.path.join(out_dir, "obs_reset.png"),
              cols=min(4, num_envs),
              title="Observations immediately after reset. Green=projected target visible, red=not visible")

    # Save a sequence from env 0 every few steps
    seq_frames: list[tuple[np.ndarray, float, float, bool]] = []
    for t in range(args_cli.steps):
        idx = torch.randint(0, n_actions, (num_envs,), device=env.unwrapped.device)
        actions = torch.nn.functional.one_hot(idx, n_actions).float()
        obs, _, _, _ = env.step(actions)
        if t % max(1, args_cli.steps // 8) == 0:
            rgb, u, v, vis = grab(env)
            seq_frames.append((rgb[0].copy(), float(u[0]), float(v[0]), bool(vis[0])))

    # Grid of mid-rollout frames across envs
    rgb, u, v, vis = grab(env)
    save_grid(rgb, u, v, vis,
              os.path.join(out_dir, "obs_grid.png"),
              cols=min(4, num_envs),
              title=f"Observations at step {args_cli.steps} (random actions)")

    # env0 sequence
    n = len(seq_frames)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, (rgb_i, ui, vi, visi) in zip(axes, seq_frames):
            ax.imshow(np.clip(rgb_i, 0, 1))
            col = "lime" if visi else "red"
            ax.add_patch(mpatches.Circle((ui, vi), 6, fill=False, color=col, linewidth=2))
            ax.axis("off")
        fig.suptitle("env 0 over time (random actions)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "env0_seq.png"), dpi=110)
        plt.close(fig)

    env.close()
    print(f"[INFO] Images saved to {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
