# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 VLA-RL 环境配置 — 支持 yaw/pitch/zoom 三轴控制。"""

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
class TargetAimingVisualEnvCfg(DirectRLEnvCfg):
    """VLA 在线 RL 微调环境配置。

    动作空间扩展为 3 维：[delta_yaw_vel, delta_pitch_vel, delta_zoom]
    zoom 通过运行时修改相机 focal_length 实现（不需要机械变焦关节）。
    """

    decimation = 2
    episode_length_s = 15.0

    # ---- 图像分辨率（SmolVLA 需要 224×224）----
    camera_width: int = 224
    camera_height: int = 224

    # ---- 动作空间：yaw_vel, pitch_vel, zoom_delta ----
    action_space = 3
    # observation_space 不用于 VLA 模式（图像直接发给 VLA 服务）
    observation_space = 0
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
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Car",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_PROJECT_ROOT, "assets/lamborghini_revuelto.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(6.0, 6.0, 0.05),
            rot=(0.7071, 0.7071, 0.0, 0.0),
        ),
    )

    # Car spawn randomization range
    target_pos_range = {
        "x": (3.0, 10.0),
        "y": (-6.0, 6.0),
        "z": (0.05, 0.05),
    }

    # ---- TiledCamera（复用 spawncargimbal.py 验证过的参数）----
    tiled_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Gimbal/pitch_link/camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,          # 初始焦距，运行时动态修改
            horizontal_aperture=20.955,
            clipping_range=(0.1, 50.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # ---- zoom 范围 ----
    focal_length_min: float = 6.0     # 广角端
    focal_length_max: float = 50.0    # 长焦端
    focal_length_default: float = 12.0

    # ---- 动作缩放 ----
    yaw_speed_scale: float = 1.0
    pitch_speed_scale: float = 1.0
    zoom_speed_scale: float = 2.0     # focal_length 每步最大变化量

    # ---- 奖励权重 ----
    center_weight: float = 10.0
    smooth_weight: float = 0.1
    alive_weight: float = 0.5
    zoom_weight: float = 2.0
    zoom_smooth_weight: float = 0.05
    target_ratio_ideal: float = 0.15  # 目标占画面面积理想比例

    # ---- 终止条件 ----
    max_pixel_error: float = 0.95

    # ---- VLA 服务 URL（同机部署用 127.0.0.1）----
    vla_server_url: str = "http://127.0.0.1:8000"
    vla_instruction: str = "Track the vehicle and keep it centered in frame."

    # ---- 初始关节随机化 ----
    initial_yaw_range = (-0.5, 0.5)
    initial_pitch_range = (-0.3, 0.3)
