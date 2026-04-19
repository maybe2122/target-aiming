#!/usr/bin/env python3
"""Roll out the target-aiming env with a random / zero / checkpoint policy and
log per-step diagnostics: reward components, action distribution, pixel error,
visibility fraction, joint angles, episode lengths.

This is the diagnostic most likely to pin down *why* training is not converging.

Typical usage (requires Isaac Sim python):

    ./isaaclab.sh -p scripts/diagnose/rollout_diagnose.py \
        --task Isaac-Target-Aiming-Direct-v0 --policy random --steps 400 --num_envs 32

    ./isaaclab.sh -p scripts/diagnose/rollout_diagnose.py \
        --task Isaac-Target-Aiming-Direct-v0 --policy checkpoint \
        --checkpoint logs/rsl_rl/target_aiming_direct/<run>/model_750.pt \
        --steps 400 --num_envs 32

Outputs:
    outputs/diagnose/<timestamp>/metrics.npz
    outputs/diagnose/<timestamp>/diagnose.png        (per-step & distribution plots)
    outputs/diagnose/<timestamp>/reward_breakdown.png
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Target-Aiming-Direct-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400, help="number of policy steps to roll out")
parser.add_argument("--policy", choices=["random", "zero", "checkpoint"], default="random")
parser.add_argument("--checkpoint", type=str, default=None, help="required when --policy checkpoint")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--out_dir", type=str, default=None, help="where to write plots/metrics")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- rest after Isaac app is up ----
import os
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import target_aiming.tasks  # noqa: F401 -- register envs

ACTION_NAMES = ["left(yaw+)", "right(yaw-)", "up(pitch+)", "down(pitch-)", "stay"]


def make_env():
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    # Use hydra entry to get env_cfg & agent_cfg, just like train.py does.
    holder = {}

    @hydra_task_config(args_cli.task, args_cli.agent)
    def _grab(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        holder["env_cfg"] = env_cfg
        holder["agent_cfg"] = agent_cfg

    _grab()
    env_cfg = holder["env_cfg"]
    agent_cfg = holder["agent_cfg"]

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return env, env_cfg, agent_cfg


def load_policy(env, agent_cfg, checkpoint_path):
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    return runner.get_inference_policy(device=env.unwrapped.device)


def reward_components(env_unwrapped):
    """Recompute reward components from env state (mirrors compute_rewards)."""
    cfg = env_unwrapped.cfg
    pe = env_unwrapped._pixel_error_normalized
    prev = env_unwrapped._prev_pixel_error
    actions = env_unwrapped.actions
    is_idle = env_unwrapped._is_idle
    visible = env_unwrapped._target_visible
    terminated = env_unwrapped.reset_terminated
    visible_f = visible.float()

    rew_center = cfg.rew_scale_pixel_center * (1.0 - pe) * visible_f
    error_delta = (prev - pe).clamp(-0.1, 0.1)
    rew_pixel = cfg.rew_scale_pixel_error * error_delta
    rew_smooth = cfg.rew_scale_action_smooth * torch.sum(actions ** 2, dim=-1)
    rew_success = cfg.rew_scale_success * (pe < cfg.success_threshold).float()
    rew_alive = cfg.rew_scale_alive * visible_f * (1.0 - terminated.float())
    not_centered = (pe > cfg.success_threshold).float()
    rew_idle = cfg.rew_scale_idle * is_idle.float() * not_centered
    total = rew_center + rew_pixel + rew_smooth + rew_success + rew_alive + rew_idle
    return {
        "center": rew_center.mean().item(),
        "pixel": rew_pixel.mean().item(),
        "smooth": rew_smooth.mean().item(),
        "success": rew_success.mean().item(),
        "alive": rew_alive.mean().item(),
        "idle": rew_idle.mean().item(),
        "total": total.mean().item(),
    }


def main():
    env, env_cfg, agent_cfg = make_env()
    unwrapped = env.unwrapped
    device = unwrapped.device
    print(f"[INFO] num_envs={unwrapped.num_envs} device={device}")
    print(f"[INFO] fixed_step_rad={env_cfg.fixed_step_rad}  decimation={env_cfg.decimation}  "
          f"dt={env_cfg.sim.dt}  episode_length_s={env_cfg.episode_length_s}")
    print(f"[INFO] effective rad/policy-step = "
          f"{env_cfg.fixed_step_rad * env_cfg.decimation * env_cfg.sim.dt:.5f} rad "
          f"(max per 10s episode ≈ "
          f"{env_cfg.fixed_step_rad * env_cfg.episode_length_s:.3f} rad)")

    policy = None
    if args_cli.policy == "checkpoint":
        if not args_cli.checkpoint:
            raise SystemExit("--checkpoint required when --policy checkpoint")
        policy = load_policy(env, agent_cfg, args_cli.checkpoint)

    obs = env.get_observations()
    n_actions = env.unwrapped.cfg.action_space
    num_envs = unwrapped.num_envs
    T = args_cli.steps

    # Buffers (mean over envs per step)
    rec_reward_total = np.zeros(T)
    rec_reward = {k: np.zeros(T) for k in ["pixel", "smooth", "success", "alive", "idle"]}
    rec_pixel_err = np.zeros(T)
    rec_visible_frac = np.zeros(T)
    rec_idle_frac = np.zeros(T)
    rec_action_hist = np.zeros((T, n_actions), dtype=np.int64)
    rec_yaw = np.zeros(T)
    rec_pitch = np.zeros(T)
    rec_done_frac = np.zeros(T)
    rec_episode_lengths: list[int] = []

    for t in range(T):
        with torch.inference_mode():
            if policy is not None:
                actions = policy(obs)
            elif args_cli.policy == "zero":
                actions = torch.zeros(num_envs, n_actions, device=device)
                actions[:, -1] = 1.0  # choose "stay"
            else:  # random
                idx = torch.randint(0, n_actions, (num_envs,), device=device)
                actions = torch.nn.functional.one_hot(idx, n_actions).float()

            obs, rew, dones, extras = env.step(actions)

        rec_reward_total[t] = rew.mean().item()
        comps = reward_components(unwrapped)
        for k in rec_reward:
            rec_reward[k][t] = comps[k]

        rec_pixel_err[t] = unwrapped._pixel_error_normalized.mean().item()
        rec_visible_frac[t] = unwrapped._target_visible.float().mean().item()
        rec_idle_frac[t] = unwrapped._is_idle.float().mean().item()

        act_idx = actions.argmax(dim=-1).detach().cpu().numpy()
        for a in act_idx:
            rec_action_hist[t, a] += 1

        yaw_idx = unwrapped._yaw_dof_idx[0]
        pitch_idx = unwrapped._pitch_dof_idx[0]
        rec_yaw[t] = unwrapped.gimbal.data.joint_pos[:, yaw_idx].mean().item()
        rec_pitch[t] = unwrapped.gimbal.data.joint_pos[:, pitch_idx].mean().item()

        rec_done_frac[t] = dones.float().mean().item()
        ep_lens = unwrapped.episode_length_buf[dones.bool()].detach().cpu().numpy().tolist()
        rec_episode_lengths.extend(ep_lens)

        if t % 50 == 0:
            print(f"step {t:4d}  rew={rec_reward_total[t]:+.3f}  "
                  f"pix={rec_pixel_err[t]:.3f}  vis={rec_visible_frac[t]:.2f}  "
                  f"done_frac={rec_done_frac[t]:.2f}")

    env.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args_cli.out_dir or os.path.join("outputs", "diagnose", f"{ts}_{args_cli.policy}")
    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        os.path.join(out_dir, "metrics.npz"),
        reward_total=rec_reward_total,
        reward_pixel=rec_reward["pixel"],
        reward_smooth=rec_reward["smooth"],
        reward_success=rec_reward["success"],
        reward_alive=rec_reward["alive"],
        reward_idle=rec_reward["idle"],
        pixel_error=rec_pixel_err,
        visible_frac=rec_visible_frac,
        idle_frac=rec_idle_frac,
        action_hist=rec_action_hist,
        yaw=rec_yaw,
        pitch=rec_pitch,
        done_frac=rec_done_frac,
        episode_lengths=np.array(rec_episode_lengths),
    )

    # ---- plot 1: overview ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    x = np.arange(T)

    axes[0, 0].plot(x, rec_reward_total, label="total")
    axes[0, 0].set_title("Per-step mean reward (over envs)")
    axes[0, 0].grid(alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(x, rec_pixel_err)
    axes[0, 1].axhline(env_cfg.success_threshold, color="g", ls="--", label="success threshold")
    axes[0, 1].set_title("Pixel error (normalized)"); axes[0, 1].grid(alpha=0.3); axes[0, 1].legend()

    axes[1, 0].plot(x, rec_visible_frac, label="visible frac")
    axes[1, 0].plot(x, rec_done_frac, label="done frac")
    axes[1, 0].plot(x, rec_idle_frac, label="idle action frac")
    axes[1, 0].set_title("Visibility / dones / idle"); axes[1, 0].grid(alpha=0.3); axes[1, 0].legend()

    axes[1, 1].plot(x, rec_yaw, label="yaw")
    axes[1, 1].plot(x, rec_pitch, label="pitch")
    axes[1, 1].set_title("Mean joint angles (rad)"); axes[1, 1].grid(alpha=0.3); axes[1, 1].legend()

    # action histogram aggregated over all steps
    agg = rec_action_hist.sum(axis=0)
    axes[2, 0].bar(ACTION_NAMES, agg)
    axes[2, 0].set_title("Action frequency (all steps)")
    axes[2, 0].tick_params(axis="x", rotation=20)

    if rec_episode_lengths:
        axes[2, 1].hist(rec_episode_lengths, bins=30)
        axes[2, 1].set_title(f"Episode length histogram "
                             f"(mean={np.mean(rec_episode_lengths):.1f}, "
                             f"max_ep={env_cfg.episode_length_s/(env_cfg.decimation*env_cfg.sim.dt):.0f})")
    else:
        axes[2, 1].text(0.5, 0.5, "no episodes ended", transform=axes[2, 1].transAxes, ha="center")
        axes[2, 1].set_title("Episode lengths")

    fig.suptitle(f"Rollout diagnose — policy={args_cli.policy} steps={T} envs={num_envs}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diagnose.png"), dpi=120)
    plt.close(fig)

    # ---- plot 2: reward breakdown ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, v in rec_reward.items():
        ax.plot(x, v, label=k, linewidth=1.2)
    ax.plot(x, rec_reward_total, label="total", color="k", linewidth=1.5, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Per-step reward breakdown (mean over envs)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_breakdown.png"), dpi=120)
    plt.close(fig)

    # ---- print summary ----
    print("\n===== Summary =====")
    print(f"out_dir: {out_dir}")
    print(f"mean total reward       : {rec_reward_total.mean():+.4f}")
    for k, v in rec_reward.items():
        print(f"mean reward component {k:8s}: {v.mean():+.4f}")
    print(f"mean pixel error        : {rec_pixel_err.mean():.4f}")
    print(f"mean visible fraction   : {rec_visible_frac.mean():.4f}")
    print(f"mean idle fraction      : {rec_idle_frac.mean():.4f}")
    print(f"mean done fraction/step : {rec_done_frac.mean():.4f}")
    if rec_episode_lengths:
        print(f"episode length mean/median/min/max: "
              f"{np.mean(rec_episode_lengths):.1f} / {np.median(rec_episode_lengths):.1f} / "
              f"{np.min(rec_episode_lengths)} / {np.max(rec_episode_lengths)}")

    action_freq = rec_action_hist.sum(axis=0) / rec_action_hist.sum()
    print("action freq              :", {n: f"{f:.3f}" for n, f in zip(ACTION_NAMES, action_freq)})


if __name__ == "__main__":
    main()
    simulation_app.close()
