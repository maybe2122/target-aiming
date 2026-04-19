# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Target aiming environment: RGB image observation, yaw/pitch action output."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
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

        # Virtual camera geometry (for geometric projection used in reward)
        self._img_cx = self.cfg.camera_width / 2.0
        self._img_cy = self.cfg.camera_height / 2.0
        self._img_diag = (self.cfg.camera_width ** 2 + self.cfg.camera_height ** 2) ** 0.5
        self._focal_px = (
            self.cfg.camera_width * self.cfg.camera_focal_length / self.cfg.camera_horizontal_aperture
        )

        # Runtime caches
        # Runtime caches
        self._pixel_error_normalized = torch.zeros(self.num_envs, device=self.device)
        self._prev_pixel_error = torch.zeros(self.num_envs, device=self.device)
        self._target_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_u = torch.zeros(self.num_envs, device=self.device)
        self._is_idle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Counts consecutive steps the target has been out of view (reset on visibility).
        self._invisible_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Discrete action mapping: 0=left, 1=right, 2=up, 3=down, 4=stay
        step = self.cfg.fixed_step_rad
        self._action_map = torch.tensor([
            [+step, 0.0],   # 0: left  (yaw+)
            [-step, 0.0],   # 1: right (yaw-)
            [0.0, +step],   # 2: up    (pitch+)
            [0.0, -step],   # 3: down  (pitch-)
            [0.0,  0.0],    # 4: stay
        ], device=self.device)
       


    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # 1. Gimbal (Articulation)
        self.gimbal = Articulation(self.cfg.robot_cfg)

        # 2. Car (target) — spawn USD, then patch RigidBodyAPI before RigidObject resolves
        self.target = RigidObject(self.cfg.target_cfg)

        # car USD has no RigidBodyAPI natively; ensure it's applied before cloning
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        car_prim = stage.GetPrimAtPath("/World/envs/env_0/Car")
        if car_prim.IsValid() and not car_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(car_prim)
            UsdPhysics.RigidBodyAPI(car_prim).CreateKinematicEnabledAttr(True)
        if car_prim.IsValid() and not car_prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(car_prim)
            UsdPhysics.MassAPI(car_prim).CreateMassAttr(1.0)

        # 3. TiledCamera mounted on gimbal pitch_link
        self.camera = TiledCamera(self.cfg.tiled_camera_cfg)

        # 4. Ground plane + light (matched to validated spawncargimbal.py)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(500.0, 500.0)))
        light_cfg = sim_utils.DomeLightCfg(intensity=6000.0, color=(0.85, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 5. Clone parallel environments (RigidBodyAPI is now on env_0/Car, will be replicated)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 6. Register assets
        self.scene.articulations["gimbal"] = self.gimbal
        self.scene.rigid_objects["target"] = self.target
        self.scene.sensors["camera"] = self.camera

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: (N, 5) one-hot encoded discrete actions
        action_indices = actions.argmax(dim=-1)  # (N,)
        self.actions = self._action_map[action_indices]  # (N, 2) -> (yaw_vel, pitch_vel)
        self._is_idle = (action_indices == 4)  # True when agent chose "stay"

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
    # Intermediate computation (reward uses geometric projection)
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self):
        # save previous error
        self._prev_pixel_error = self._pixel_error_normalized.clone()

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

        # Track consecutive invisible steps for grace-period termination.
        self._invisible_counter = torch.where(
            visible,
            torch.zeros_like(self._invisible_counter),
            self._invisible_counter + 1,
        )


    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_view = self._invisible_counter > self.cfg.invisible_grace_steps
        return out_of_view, time_out

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            self.cfg.rew_scale_pixel_center,
            self.cfg.rew_scale_pixel_error,
            self.cfg.rew_scale_action_smooth,
            self.cfg.rew_scale_success,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_idle,
            self.cfg.success_threshold,
            self._pixel_error_normalized,
            self._prev_pixel_error,
            self.actions,
            self._is_idle,
            self._target_visible,
            self.reset_terminated,
        )
    # ------------------------------------------------------------------
    # Observations: RGB image + joint angles
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # RGB from TiledCamera: (N, H, W, 4) uint8 -> (N, 3, H, W) float [0, 1]
        rgb_raw = self.camera.data.output["rgb"]  # (N, H, W, 4) with alpha
        rgb = rgb_raw[..., :3].float() / 255.0    # (N, H, W, 3)
        image = rgb.permute(0, 3, 1, 2)           # (N, 3, H, W)

        # Joint angles
        yaw = self.gimbal.data.joint_pos[:, self._yaw_dof_idx[0]].unsqueeze(-1)
        pitch = self.gimbal.data.joint_pos[:, self._pitch_dof_idx[0]].unsqueeze(-1)
        state = torch.cat([yaw, pitch], dim=-1)    # (N, 2)

        return {"image": image, "state": state}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.gimbal._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # 1. Randomize car position first
        target_root_state = self.target.data.default_root_state[env_ids].clone()
        target_root_state[:, :3] += self.scene.env_origins[env_ids]

        target_root_state[:, 0] += sample_uniform(
            self.cfg.target_pos_range["x"][0],
            self.cfg.target_pos_range["x"][1],
            (n,), self.device,
        )
        target_root_state[:, 1] += sample_uniform(
            self.cfg.target_pos_range["y"][0],
            self.cfg.target_pos_range["y"][1],
            (n,), self.device,
        )
        target_root_state[:, 2] = sample_uniform(
            self.cfg.target_pos_range["z"][0],
            self.cfg.target_pos_range["z"][1],
            (n,), self.device,
        )

        self.target.write_root_pose_to_sim(target_root_state[:, :7], env_ids)
        self.target.write_root_velocity_to_sim(target_root_state[:, 7:], env_ids)

        # 2. Reset gimbal: point at car, then add small random offset
        default_root_state = self.gimbal.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        gimbal_pos = default_root_state[:, :3]

        # Compute yaw to aim at the car
        delta = target_root_state[:, :3] - gimbal_pos
        dx, dy = delta[:, 0], delta[:, 1]
        dist_xy = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1e-6)
        aim_yaw = torch.atan2(dy, dx)

        # Compute pitch so the short camera arm (≤ 0.5m here) looks at the car.
        # URDF puts pitch_link 0.175m above gimbal root (yaw_joint 0.075 + pitch_joint 0.1);
        # the TiledCamera offset adds another offset.pos.z on top.
        # With a short arm the simple right-triangle aim is accurate enough — no fudge needed.
        pitch_link_z = 0.175
        cam_z_from_gimbal = pitch_link_z + self.cfg.tiled_camera_cfg.offset.pos[2]
        cam_z_tensor = torch.tensor(cam_z_from_gimbal, device=self.device)
        target_z = target_root_state[:, 2]
        aim_pitch = torch.atan2(cam_z_tensor - target_z, dist_xy)

        # Random offset for yaw (within 80% of half-FOV so target is usually off-center)
        half_fov = math.atan(self.cfg.camera_horizontal_aperture / (2.0 * self.cfg.camera_focal_length))
        yaw_offset = half_fov * 0.8

        joint_pos = self.gimbal.data.default_joint_pos[env_ids].clone()
        joint_vel = self.gimbal.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self._yaw_dof_idx[0]] = aim_yaw + sample_uniform(
            -yaw_offset, yaw_offset, (n,), joint_pos.device,
        )
        joint_pos[:, self._pitch_dof_idx[0]] = aim_pitch + sample_uniform(
            -0.15, 0.15, (n,), joint_pos.device,
        )

        self.gimbal.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.gimbal.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.gimbal.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Recompute projection with the new gimbal/target state and sync caches so
        # the first reward after reset uses a consistent prev/current pair
        # (otherwise pixel_error would spike from the old episode's value).
        u, v, visible = self._project_target_to_image()
        pixel_dist = torch.sqrt((u - self._img_cx) ** 2 + (v - self._img_cy) ** 2)
        fresh_error = torch.where(
            visible,
            pixel_dist / self._img_diag,
            torch.ones_like(u),
        )
        self._pixel_error_normalized[env_ids] = fresh_error[env_ids]
        self._prev_pixel_error[env_ids] = fresh_error[env_ids]
        self._target_visible[env_ids] = visible[env_ids]
        self._invisible_counter[env_ids] = 0

    # ------------------------------------------------------------------
    # Utility: project target 3D position to virtual image plane
    # ------------------------------------------------------------------
    def _project_target_to_image(self):
        """Return target pixel coords (u, v) and visibility mask.

        Uses full pinhole projection from the actual camera world position
        and orientation, accounting for the 2m arm offset on pitch_link.

        Camera basis vectors (derived from URDF + camera quaternion):
            forward = Rz(yaw) * Ry(pitch) * [1, 0, 0]
            right   = Rz(yaw) * [0, -1, 0]
            up      = Rz(yaw) * Ry(pitch) * [0, 0, 1]
        """
        target_pos_w = self.target.data.root_pos_w   # (N, 3)
        gimbal_pos_w = self.gimbal.data.root_pos_w   # (N, 3)

        cur_yaw = self.gimbal.data.joint_pos[:, self._yaw_dof_idx[0]]
        cur_pitch = self.gimbal.data.joint_pos[:, self._pitch_dof_idx[0]]

        cos_y, sin_y = torch.cos(cur_yaw), torch.sin(cur_yaw)
        cos_p, sin_p = torch.cos(cur_pitch), torch.sin(cur_pitch)

        # Camera offset in pitch_link frame, read from cfg so projection stays
        # in sync with the actual TiledCamera placement.
        cam_off_x, _, cam_off_z = self.cfg.tiled_camera_cfg.offset.pos
        # Fixed URDF offsets (base → pitch_link) along gimbal Z:
        #   yaw_joint origin at 0.075, pitch_joint origin at 0.1 → sum 0.175.
        pitch_link_z = 0.175

        # After Ry(pitch) applied to the pitch_link-frame offset:
        #   arm_x = cam_off_x*cos_p + cam_off_z*sin_p
        #   arm_z = -cam_off_x*sin_p + cam_off_z*cos_p
        cam_offset_x = cam_off_x * cos_p + cam_off_z * sin_p   # in yaw-rotated XY plane
        cam_offset_z = -cam_off_x * sin_p + cam_off_z * cos_p + pitch_link_z

        cam_x = gimbal_pos_w[:, 0] + cos_y * cam_offset_x
        cam_y = gimbal_pos_w[:, 1] + sin_y * cam_offset_x
        cam_z = gimbal_pos_w[:, 2] + cam_offset_z

        # Vector from camera to target
        dx = target_pos_w[:, 0] - cam_x
        dy = target_pos_w[:, 1] - cam_y
        dz = target_pos_w[:, 2] - cam_z

        # Project onto camera basis vectors
        d_forward = dx * cos_y * cos_p + dy * sin_y * cos_p - dz * sin_p
        d_right   = dx * sin_y         - dy * cos_y
        d_up      = dx * cos_y * sin_p + dy * sin_y * sin_p + dz * cos_p

        # Pinhole projection
        d_forward_safe = d_forward.clamp(min=1e-6)
        u = self._img_cx + self._focal_px * d_right / d_forward_safe
        v = self._img_cy - self._focal_px * d_up / d_forward_safe

        visible = (
            (d_forward > 0)
            & (u >= 0) & (u < self.cfg.camera_width)
            & (v >= 0) & (v < self.cfg.camera_height)
        )
        return u, v, visible


# ----------------------------------------------------------------------
# Reward computation (JIT compiled)
# ----------------------------------------------------------------------
@torch.jit.script
def compute_rewards(
    rew_scale_pixel_center: float,
    rew_scale_pixel_error: float,
    rew_scale_action_smooth: float,
    rew_scale_success: float,
    rew_scale_alive: float,
    rew_scale_idle: float,
    success_threshold: float,
    pixel_error: torch.Tensor,
    prev_pixel_error: torch.Tensor,
    actions: torch.Tensor,
    is_idle: torch.Tensor,
    target_visible: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:

    visible_f = target_visible.float()

    # Absolute centering shaping: always dense while visible, no dependence on prev state.
    rew_center = rew_scale_pixel_center * (1.0 - pixel_error) * visible_f

    # Delta improvement, clipped to avoid spikes at reset / visibility flips.
    error_delta = (prev_pixel_error - pixel_error).clamp(-0.1, 0.1)
    rew_pixel = rew_scale_pixel_error * error_delta

    # Smooth (kept for config compatibility; default scale is 0).
    rew_smooth = rew_scale_action_smooth * torch.sum(actions ** 2, dim=-1)

    # Success bonus.
    rew_success = rew_scale_success * (pixel_error < success_threshold).float()

    # Alive: only credit when visible and not just terminated.
    rew_alive = rew_scale_alive * visible_f * (1.0 - reset_terminated.float())

    # Idle penalty (default scale is 0).
    not_centered = (pixel_error > success_threshold).float()
    rew_idle = rew_scale_idle * is_idle.float() * not_centered

    return rew_center + rew_pixel + rew_smooth + rew_success + rew_alive + rew_idle