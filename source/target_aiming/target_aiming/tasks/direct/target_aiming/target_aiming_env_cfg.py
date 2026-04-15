# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .gimbal_cfg import GIMBAL_CFG

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)


@configclass
class TargetAimingEnvCfg(DirectRLEnvCfg):
    """Target aiming with RGB observation: gimbal tracks a car and keeps it centered."""

    decimation = 4
    episode_length_s = 10.0 #10s

    # obs = {"image": (3,128,128), "state": (2,)}
    # action = (delta_yaw, delta_pitch)
    camera_width: int = 128
    camera_height: int = 128
    action_space = 2
    observation_space = {"image": [3, 128, 128], "state": 2}
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=20.0, replicate_physics=True
    )

    # ---- Gimbal (robot) ----
    robot_cfg: ArticulationCfg = GIMBAL_CFG.replace(
        prim_path="/World/envs/env_.*/Gimbal"
    )
    yaw_dof_name = "yaw_joint"
    pitch_dof_name = "pitch_joint"

    # ---- Car (target) ----
    # car USD 无 RigidBodyAPI，spawn 时通过 rigid_props 补上（kinematic，不参与动力学）
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Car",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_PROJECT_ROOT, "assets/smallcar.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Car spawn randomization range (offset from default pos)
    target_pos_range = {
        "x": (2.0, 3.0),
        "y": (-2.0, 2.0),
        "z": (0.05, 0.05),  # car stays on ground
    }

    # ---- TiledCamera on pitch_link (matches spawncargimbal.py) ----
    tiled_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Gimbal/pitch_link/camera",
        update_period=0.0,  # update every sim step
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 2.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # Pinhole intrinsics for geometric projection (reward computation only)
    camera_focal_length: float = 18.0
    camera_horizontal_aperture: float = 20.955

    # ---- Action ----
    action_scale = 1.0
    max_action_rad: float = 0.05236  # max delta per step in radians (~3 degrees)

    # ---- Reward ----
    rew_scale_pixel_error: float = -1.0
    rew_scale_action_smooth: float = -0.01
    rew_scale_success: float = 5.0
    rew_scale_alive: float = 0.1

    # pixel_error > max_pixel_error when target not visible → terminate
    max_pixel_error: float = 0.95
    # pixel_error < success_threshold → success bonus
    success_threshold: float = 0.08

    # ---- Initial joint randomization ----
    initial_yaw_range = (-0.5, 0.5)
    initial_pitch_range = (-0.3, 0.3)

    # ---- Debug ----
    show_debug_markers: bool = False  # scene markers (visible to camera, disable during training)
    show_camera_feed: bool = False  # OpenCV window showing env-0 camera with HUD
