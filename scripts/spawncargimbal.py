# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="This script demonstrates adding a gimbal to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim import UsdFileCfg


GIMBAL_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/gimbal/gimbal.usd"
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "yaw_joint": 0.0,
            "pitch_joint": 0.0,
        },
    ),
    actuators={
        "gimbal_acts": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint", "pitch_joint"],
            damping=None,
            stiffness=None,
        )
    },
)

# 用 AssetBaseCfg 加载 Car，绕过 RigidBodyAPI 检查
CAR_CONFIG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Car",
    spawn=UsdFileCfg(
        usd_path="/home/sz/code/rl/target_aiming/assets/smallcar.usd",
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(3.5, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


class SceneCfg(InteractiveSceneCfg):
    """场景：每个环境包含一个 Gimbal 和一辆 Car。"""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=6000.0, color=(0.85, 0.75, 0.75)),
    )

    Gimbal = GIMBAL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Gimbal")

    Car = CAR_CONFIG  # AssetBaseCfg 不需要 .replace()，prim_path 已写好

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Gimbal/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 2.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    num_envs = scene.num_envs

    os.makedirs("camera_output", exist_ok=True)

    while simulation_app.is_running():
        # ---------- reset ----------
        if count % 500 == 0:
            count = 0

            root_gimbal_state = scene["Gimbal"].data.default_root_state.clone()
            root_gimbal_state[:, :3] += scene.env_origins
            scene["Gimbal"].write_root_pose_to_sim(root_gimbal_state[:, :7])
            scene["Gimbal"].write_root_velocity_to_sim(root_gimbal_state[:, 7:])

            joint_pos, joint_vel = (
                scene["Gimbal"].data.default_joint_pos.clone(),
                scene["Gimbal"].data.default_joint_vel.clone(),
            )
            scene["Gimbal"].write_joint_state_to_sim(joint_pos, joint_vel)

            # AssetBaseCfg 没有 write_root_pose，Car 不需要 reset

            scene.reset()
            print("[INFO]: Resetting scene state...")

        # ---------- 打印 body 名称（仅第1帧）----------
        if count == 1:
            print("[INFO]: Gimbal body names:", scene["Gimbal"].data.body_names)

        # ---------- 控制 Gimbal 缓慢扫描 ----------
        wave_action = scene["Gimbal"].data.default_joint_pos.clone()
        wave_action[:, 0] = 0.3 * np.sin(2 * np.pi * 0.05 * sim_time)  # yaw
        # wave_action[:, 1] = 0.1 * np.sin(2 * np.pi * 0.1 * sim_time)   # pitch
        scene["Gimbal"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # ---------- 每100帧保存一次，每个环境单独存图 ----------
        if count % 100 == 0:
            rgb_data = scene["camera"].data.output["rgb"]  # (num_envs, H, W, 4)
            if rgb_data is not None:
                for env_idx in range(num_envs):
                    rgb_np = rgb_data[env_idx].cpu().numpy()[:, :, :3]
                    img = Image.fromarray(rgb_np.astype(np.uint8))
                    filename = f"camera_output/env{env_idx}_frame_{count:06d}.png"
                    img.save(filename)
                print(f"[INFO]: Saved frame {count:06d} for {num_envs} envs")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([10.5, 10.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = SceneCfg(args_cli.num_envs, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()