# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .gimbal_cfg import GIMBAL_CFG


@configclass
class TargetAimingEnvCfg(DirectRLEnvCfg):
    """Phase-1 configuration: coordinate-based observation, no camera rendering."""

    decimation = 4
    episode_length_s = 10.0

    # Phase 1: obs = (u_norm, v_norm, yaw, pitch), action = (Δyaw, Δpitch)
    action_space = 2
    observation_space = 4
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)

    robot_cfg: ArticulationCfg = GIMBAL_CFG.replace(prim_path="/World/envs/env_.*/Gimbal")

    yaw_dof_name = "yaw_joint"
    pitch_dof_name = "pitch_joint"

    # Virtual camera parameters (for geometric projection, no actual rendering in phase 1)
    camera_width: int = 84
    camera_height: int = 84
    # Pinhole camera intrinsics (match phase 2 TiledCamera settings)
    camera_focal_length: float = 12.0
    camera_horizontal_aperture: float = 20.955

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                roughness=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 1.5)),
    )

    target_pos_range = {
        "x": (2.0, 8.0),
        "y": (-3.0, 3.0),
        "z": (0.5, 3.0),
    }

    action_scale = 1.0

    rew_scale_pixel_error: float = -1.0
    rew_scale_action_smooth: float = -0.01
    rew_scale_success: float = 5.0
    rew_scale_alive: float = 0.1

    max_pixel_error: float = 0.95
    success_threshold: float = 0.05

    initial_yaw_range = (-0.5, 0.5)
    initial_pitch_range = (-0.3, 0.3)
