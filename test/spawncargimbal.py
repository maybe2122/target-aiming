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
import math
import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import UsdFileCfg


# ==============================================================================
# Asset 配置
# ==============================================================================
GIMBAL_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/gimbal/gimbal.usd"
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "yaw_joint":   0.0,
            "pitch_joint": 0.0,
        },
    ),
    actuators={
        "gimbal_acts": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint", "pitch_joint"],
            stiffness=10000.0,
            damping=100.0,
        )
    },
)

CAR_CONFIG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Car",
    spawn=UsdFileCfg(
        usd_path="/home/sz/code/rl/target_aiming/assets/lamborghini_revuelto.usd",
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(6.0, 6.0, 0.05),
        rot=(0.7071, 0.7071, 0.0, 0.0),
    ),
)


# ==============================================================================
# Scene 配置
# ==============================================================================
class NewRobotsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    Gimbal = GIMBAL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Gimbal")
    Car = CAR_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Car")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Gimbal/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,  # 减小焦距，增大 FOV
            focus_distance=400.0,
            horizontal_aperture=40.0,  # 增大水平孔径，增大 FOV
            clipping_range=(0.1, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


# ==============================================================================
# 主循环
# ==============================================================================
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    os.makedirs("camera_output", exist_ok=True)

    device = sim.device
    num_envs = scene.num_envs

    # yaw 旋转参数
    yaw_speed = 0.1  # 降低速度（原来是 0.5）
    pitch_target = 0.0

    while simulation_app.is_running():
        # ------------------------------------------------------------------
        # Reset
        # ------------------------------------------------------------------
        if count % 1000 == 0:
            count = 0
            root_state = scene["Gimbal"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["Gimbal"].write_root_pose_to_sim(root_state[:, :7])
            scene["Gimbal"].write_root_velocity_to_sim(root_state[:, 7:])

            jp, jv = (
                scene["Gimbal"].data.default_joint_pos.clone(),
                scene["Gimbal"].data.default_joint_vel.clone(),
            )
            scene["Gimbal"].write_joint_state_to_sim(jp, jv)
            scene.reset()
            print("[INFO]: Resetting scene...")

        if count == 1:
            print("[INFO]: Gimbal body names:", scene["Gimbal"].data.body_names)

        # ------------------------------------------------------------------
        # Yaw 持续旋转
        # ------------------------------------------------------------------
        yaw_target = math.sin(sim_time * yaw_speed) * math.pi
        
        joint_targets = torch.tensor([[yaw_target, pitch_target]], device=device).expand(num_envs, -1)
        scene["Gimbal"].set_joint_position_target(joint_targets)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # ------------------------------------------------------------------
        # 每50帧保存相机图像
        # ------------------------------------------------------------------
        if count % 50 == 0:
            rgb_data = scene["camera"].data.output["rgb"]
            if rgb_data is not None:
                rgb_np = rgb_data[0].cpu().numpy()[:, :, :3]
                img = Image.fromarray(rgb_np.astype(np.uint8))
                save_path = f"camera_output/frame_{count:06d}.png"
                img.save(save_path)
                print(f"[INFO]: Saved {save_path}, yaw={math.degrees(yaw_target):.1f}°")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([10.5, 10.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()