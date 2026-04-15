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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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
        self._pixel_error_normalized = torch.zeros(self.num_envs, device=self.device)
        self._target_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_u = torch.zeros(self.num_envs, device=self.device)
        self._target_v = torch.zeros(self.num_envs, device=self.device)

        # ---- Debug visualization markers (only during play, disabled for training) ----
        if self.cfg.show_debug_markers:
            # Target status: green = centered, yellow = in FOV, red = out of FOV
            self._target_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/target_status",
                markers={
                    "centered": sim_utils.SphereCfg(
                        radius=0.3,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "in_fov": sim_utils.SphereCfg(
                        radius=0.3,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                    ),
                    "out_fov": sim_utils.SphereCfg(
                        radius=0.3,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            ))
            # Camera aim point: blue sphere 10 m along camera forward
            self._aim_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/aim_point",
                markers={
                    "aim": sim_utils.SphereCfg(
                        radius=0.15,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                    ),
                },
            ))

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
        self.actions = actions.clone().clamp(-self.cfg.max_action_rad, self.cfg.max_action_rad)

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

        # Update debug markers (only during play)
        if self.cfg.show_debug_markers:
            self._update_debug_vis()

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------
    def _update_debug_vis(self):
        """Draw markers: target status sphere + camera aim-point sphere."""
        # 1. Target status sphere (above car)
        target_pos = self.target.data.root_pos_w.clone()  # (N, 3)
        target_pos[:, 2] += 1.0  # lift above car

        centered = self._pixel_error_normalized < self.cfg.success_threshold  # green
        in_fov = self._target_visible & ~centered                             # yellow
        # remaining = out of FOV                                              # red

        # marker_indices: 0=centered(green), 1=in_fov(yellow), 2=out_fov(red)
        marker_idx = torch.full((self.num_envs,), 2, dtype=torch.int32, device=self.device)
        marker_idx[in_fov] = 1
        marker_idx[centered] = 0

        self._target_markers.visualize(translations=target_pos, marker_indices=marker_idx)

        # 2. Aim-point sphere (10 m along camera forward direction)
        cur_yaw = self.gimbal.data.joint_pos[:, self._yaw_dof_idx[0]]
        cur_pitch = self.gimbal.data.joint_pos[:, self._pitch_dof_idx[0]]
        gimbal_pos = self.gimbal.data.root_pos_w  # (N, 3)

        aim_dist = 10.0
        aim_pos = gimbal_pos.clone()
        aim_pos[:, 0] += aim_dist * torch.cos(cur_yaw) * torch.cos(cur_pitch)
        aim_pos[:, 1] += aim_dist * torch.sin(cur_yaw) * torch.cos(cur_pitch)
        aim_pos[:, 2] += aim_dist * torch.sin(cur_pitch)

        self._aim_markers.visualize(translations=aim_pos)

        # 3. OpenCV camera feed window (env 0)
        if self.cfg.show_camera_feed:
            self._render_camera_feed()

    def _render_camera_feed(self):
        """Show env-0 camera image with crosshair, target dot, and HUD via matplotlib."""
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        rgb_raw = self.camera.data.output.get("rgb")
        if rgb_raw is None:
            return

        img = rgb_raw[0, :, :, :3].cpu().numpy().astype(np.uint8)
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Lazy-init matplotlib figure
        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1, figsize=(6, 6))
            self._im = self._ax.imshow(img)
            # Static crosshair
            self._ax.axhline(cy, color="lime", linewidth=0.8, linestyle="--")
            self._ax.axvline(cx, color="lime", linewidth=0.8, linestyle="--")
            cross_r = min(w, h) * 0.15
            self._cross_circle = patches.Circle((cx, cy), cross_r, fill=False, edgecolor="lime", linewidth=0.8)
            self._ax.add_patch(self._cross_circle)
            # Dynamic elements
            self._target_dot, = self._ax.plot([], [], "ro", markersize=6)
            self._target_line, = self._ax.plot([], [], "r-", linewidth=0.8)
            self._hud_text = self._ax.text(
                2, 5, "", fontsize=9, color="white", fontweight="bold",
                verticalalignment="top", bbox=dict(facecolor="black", alpha=0.6, pad=2),
            )
            self._ax.set_xlim(0, w)
            self._ax.set_ylim(h, 0)
            self._ax.set_axis_off()
            self._fig.tight_layout(pad=0)
            self._fig.show()

        # Update image
        self._im.set_data(img)

        # Update target dot + line
        u = self._target_u[0].item()
        v = self._target_v[0].item()
        visible = self._target_visible[0].item()
        if visible:
            self._target_dot.set_data([u], [v])
            self._target_line.set_data([cx, u], [cy, v])
        else:
            self._target_dot.set_data([], [])
            self._target_line.set_data([], [])

        # Update HUD
        err = self._pixel_error_normalized[0].item()
        status = "CENTERED" if err < self.cfg.success_threshold else ("IN FOV" if visible else "LOST")
        yaw = self.gimbal.data.joint_pos[0, self._yaw_dof_idx[0]].item()
        pitch = self.gimbal.data.joint_pos[0, self._pitch_dof_idx[0]].item()
        self._hud_text.set_text(f"{status}  err={err:.3f}\nyaw={yaw:.2f}  pitch={pitch:.2f}")
        self._hud_text.set_color("lime" if status == "CENTERED" else ("yellow" if status == "IN FOV" else "red"))

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_view = (
            ~self._target_visible
            & (self._pixel_error_normalized > self.cfg.max_pixel_error)
        )
        return out_of_view, time_out

    # ------------------------------------------------------------------
    # Rewards
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

        # 2. Randomize car position
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
# Reward computation (JIT compiled)
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

    # 3. Success reward (target centered)
    rew_success = rew_scale_success * (pixel_error < success_threshold).float()

    # 4. Alive reward (target in view)
    rew_alive = rew_scale_alive * target_visible.float() * (1.0 - reset_terminated.float())

    return rew_pixel + rew_smooth + rew_success + rew_alive
