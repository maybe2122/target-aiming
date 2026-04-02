# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-1 target aiming environment: coordinate-based observation, no camera rendering."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .target_aiming_env_cfg import TargetAimingEnvCfg


class TargetAimingEnv(DirectRLEnv):
    cfg: TargetAimingEnvCfg

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, cfg: TargetAimingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices
        self._yaw_dof_idx, _ = self.gimbal.find_joints(self.cfg.yaw_dof_name)
        self._pitch_dof_idx, _ = self.gimbal.find_joints(self.cfg.pitch_dof_name)

        # Virtual camera geometry (for geometric projection)
        self._img_cx = self.cfg.camera_width / 2.0
        self._img_cy = self.cfg.camera_height / 2.0
        self._img_diag = (self.cfg.camera_width ** 2 + self.cfg.camera_height ** 2) ** 0.5
        # focal_px = width * focal_length / horizontal_aperture
        self._focal_px = (
            self.cfg.camera_width * self.cfg.camera_focal_length / self.cfg.camera_horizontal_aperture
        )

        # Runtime caches (filled in _compute_intermediate_values, used by dones/rewards/observations)
        self._pixel_error_normalized = torch.zeros(self.num_envs, device=self.device)
        self._target_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_u = torch.zeros(self.num_envs, device=self.device)
        self._target_v = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # 1. Gimbal (Articulation)
        self.gimbal = Articulation(self.cfg.robot_cfg)

        # 2. Target (static red sphere)
        self.target = RigidObject(self.cfg.target_cfg)

        # 3. Ground plane + light
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 4. Clone parallel environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 5. Register assets
        self.scene.articulations["gimbal"] = self.gimbal
        self.scene.rigid_objects["target"] = self.target

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: (N, 2) in [-1, 1] → [Δyaw, Δpitch]
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        velocity_targets = torch.zeros(
            self.num_envs, self.gimbal.num_joints, device=self.device
        )
        velocity_targets[:, self._yaw_dof_idx[0]] = (
            self.actions[:, 0] * self.cfg.action_scale
        )
        velocity_targets[:, self._pitch_dof_idx[0]] = (
            self.actions[:, 1] * self.cfg.action_scale
        )
        self.gimbal.set_joint_velocity_target(velocity_targets)

    # ------------------------------------------------------------------
    # Intermediate computation (called once per step, before dones/rewards/obs)
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self):
        """Project target to virtual image plane and cache results."""
        u, v, visible = self._project_target_to_image()
        self._target_u = u
        self._target_v = v
        self._target_visible = visible

        pixel_dist = torch.sqrt((u - self._img_cx) ** 2 + (v - self._img_cy) ** 2)
        self._pixel_error_normalized = pixel_dist / self._img_diag
        self._pixel_error_normalized = torch.where(
            visible,
            self._pixel_error_normalized,
            torch.ones_like(self._pixel_error_normalized),
        )

    # ------------------------------------------------------------------
    # Dones (called FIRST in the step — see DirectRLEnv.step)
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute intermediate values for this step (also used by rewards and obs)
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_view = (
            ~self._target_visible
            & (self._pixel_error_normalized > self.cfg.max_pixel_error)
        )
        return out_of_view, time_out

    # ------------------------------------------------------------------
    # Rewards (called SECOND — uses cached intermediate values)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            self.cfg.rew_scale_pixel_error,
            self.cfg.rew_scale_action_smooth,
            self.cfg.rew_scale_success,
            self.cfg.rew_scale_alive,
            self.cfg.success_threshold,
            self._pixel_error_normalized,
            self.actions,
            self._target_visible,
            self.reset_terminated,
        )

    # ------------------------------------------------------------------
    # Observations (called LAST — uses cached intermediate values)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # Normalize u, v: center of image → 0, edge → ±1
        u_norm = ((self._target_u - self._img_cx) / self._img_cx).clamp(-2.0, 2.0)
        v_norm = ((self._target_v - self._img_cy) / self._img_cy).clamp(-2.0, 2.0)

        yaw_pos = self.gimbal.data.joint_pos[:, self._yaw_dof_idx[0]]
        pitch_pos = self.gimbal.data.joint_pos[:, self._pitch_dof_idx[0]]

        obs = torch.stack([u_norm, v_norm, yaw_pos, pitch_pos], dim=-1)  # (N, 4)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.gimbal._ALL_INDICES
        super()._reset_idx(env_ids)

        # 1. Reset gimbal joint state
        joint_pos = self.gimbal.data.default_joint_pos[env_ids].clone()
        joint_vel = self.gimbal.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self._yaw_dof_idx[0]] += sample_uniform(
            self.cfg.initial_yaw_range[0],
            self.cfg.initial_yaw_range[1],
            (len(env_ids),),
            joint_pos.device,
        )
        joint_pos[:, self._pitch_dof_idx[0]] += sample_uniform(
            self.cfg.initial_pitch_range[0],
            self.cfg.initial_pitch_range[1],
            (len(env_ids),),
            joint_pos.device,
        )

        default_root_state = self.gimbal.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.gimbal.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.gimbal.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.gimbal.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 2. Randomize target position
        target_root_state = self.target.data.default_root_state[env_ids].clone()
        target_root_state[:, :3] += self.scene.env_origins[env_ids]

        target_root_state[:, 0] += sample_uniform(
            self.cfg.target_pos_range["x"][0],
            self.cfg.target_pos_range["x"][1],
            (len(env_ids),), self.device,
        )
        target_root_state[:, 1] += sample_uniform(
            self.cfg.target_pos_range["y"][0],
            self.cfg.target_pos_range["y"][1],
            (len(env_ids),), self.device,
        )
        target_root_state[:, 2] = sample_uniform(
            self.cfg.target_pos_range["z"][0],
            self.cfg.target_pos_range["z"][1],
            (len(env_ids),), self.device,
        )

        self.target.write_root_pose_to_sim(target_root_state[:, :7], env_ids)
        self.target.write_root_velocity_to_sim(target_root_state[:, 7:], env_ids)

    # ------------------------------------------------------------------
    # Utility: project target 3D position to virtual image plane
    # ------------------------------------------------------------------
    def _project_target_to_image(self):
        """Return target pixel coords (u, v) and visibility mask via geometric projection."""
        target_pos_w = self.target.data.root_pos_w   # (N, 3)
        gimbal_pos_w = self.gimbal.data.root_pos_w   # (N, 3)

        delta = target_pos_w - gimbal_pos_w           # (N, 3)
        dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]

        dist_xy = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1e-6)

        target_yaw = torch.atan2(dy, dx)
        target_pitch = torch.atan2(dz, dist_xy)

        cur_yaw = self.gimbal.data.joint_pos[:, self._yaw_dof_idx[0]]
        cur_pitch = self.gimbal.data.joint_pos[:, self._pitch_dof_idx[0]]

        u = self._img_cx + (target_yaw - cur_yaw) * self._focal_px
        v = self._img_cy - (target_pitch - cur_pitch) * self._focal_px

        visible = (
            (u >= 0) & (u < self.cfg.camera_width)
            & (v >= 0) & (v < self.cfg.camera_height)
        )
        return u, v, visible


# ----------------------------------------------------------------------
# Reward computation (JIT compiled — no self access)
# ----------------------------------------------------------------------
@torch.jit.script
def compute_rewards(
    rew_scale_pixel_error: float,
    rew_scale_action_smooth: float,
    rew_scale_success: float,
    rew_scale_alive: float,
    success_threshold: float,
    pixel_error: torch.Tensor,      # (N,) normalized [0, 1]
    actions: torch.Tensor,           # (N, 2)
    target_visible: torch.Tensor,    # (N,) bool
    reset_terminated: torch.Tensor,  # (N,) bool
) -> torch.Tensor:
    # 1. Pixel error penalty (main signal)
    rew_pixel = rew_scale_pixel_error * pixel_error

    # 2. Action smoothness penalty
    rew_smooth = rew_scale_action_smooth * torch.sum(actions ** 2, dim=-1)

    # 3. Success reward (sparse)
    rew_success = rew_scale_success * (pixel_error < success_threshold).float()

    # 4. Alive reward (target in view)
    rew_alive = rew_scale_alive * target_visible.float() * (1.0 - reset_terminated.float())

    return rew_pixel + rew_smooth + rew_success + rew_alive
