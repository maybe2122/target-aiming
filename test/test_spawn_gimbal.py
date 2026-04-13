# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="This script demonstrates adding a gimbal to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")



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

#{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd

GIMBAL_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/gimbal/gimbal.usd"
    ),
    actuators={
        "gimbal_acts": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint", "pitch_joint"],
            damping=None,
            stiffness=None
        )
    },
)

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
CAR_CONFIG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Car",
    spawn=UsdFileCfg(
    usd_path="/home/sz/code/rl/target_aiming/assets/lamborghini_revuelto.usd",
    # scale=(1000.0, 1000.0, 1000.0)
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(6.0, 6.0, 0.05),   # x y z
        rot=(0.7071, 0.7071, 0.0, 0.0),  # quaternion (w x y z)
    ),
)

# CAR_CONFIG = AssetBaseCfg(
#     prim_path="{ENV_REGEX_NS}/Car",
#     spawn=sim_utils.CuboidCfg(
#         size=(0.5, 0.5, 0.5),   # cube 尺寸 (x,y,z)
#         visual_material=sim_utils.PreviewSurfaceCfg(
#             diffuse_color=(1.0, 0.0, 0.0),  # 红色
#         ),
#     ),
#     init_state=AssetBaseCfg.InitialStateCfg(
#         pos=(1.5, 0.0, 1.0),   # 放空中更容易看到
#         rot=(1.0, 0.0, 0.0, 0.0),
#     ),
# )

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        },
        pos=(0.25, -0.25, 0.0),
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robots
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    # Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    Gimbal = GIMBAL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Gimbal")
    Car = CAR_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Car")
    # RGB camera mounted on Dofbot end effector
    # NOTE: replace "link5" with the actual end effector link name from:
    #       print(scene["Dofbot"].data.body_names)
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Gimbal/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    os.makedirs("camera_output", exist_ok=True)

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            count = 0

            root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            root_jetbot_state[:, :3] += scene.env_origins
            root_gimbal_state = scene["Gimbal"].data.default_root_state.clone()
            root_gimbal_state[:, :3] += scene.env_origins
            root_car_state = scene["Car"].data.default_root_state.clone()
            root_car_state[:, :3] += scene.env_origins

            scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
            scene["Gimbal"].write_root_pose_to_sim(root_gimbal_state[:, :7])
            scene["Gimbal"].write_root_velocity_to_sim(root_gimbal_state[:, 7:])
            scene["Car"].write_root_pose_to_sim(root_car_state[:, :7])
            scene["Car"].write_root_velocity_to_sim(root_car_state[:, 7:])

            joint_pos, joint_vel = (
                scene["Jetbot"].data.default_joint_pos.clone(),
                scene["Jetbot"].data.default_joint_vel.clone(),
            )
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)

            joint_pos, joint_vel = (
                scene["Gimbal"].data.default_joint_pos.clone(),
                scene["Gimbal"].data.default_joint_vel.clone(),
            )
            scene["Gimbal"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO]: Resetting scene state...")

        # 第一帧打印 Gimbal body 名称
        if count == 1:
            print("[INFO]: Gimbal body names:", scene["Gimbal"].data.body_names)

        # drive Jetbot
        if count % 100 < 75:
            action = torch.Tensor([[10.0, 10.0]])
        else:
            action = torch.Tensor([[5.0, -5.0]])
        scene["Jetbot"].set_joint_velocity_target(action)

        # wave Gimbal
        wave_action = scene["Gimbal"].data.default_joint_pos.clone()
        wave_action[:, :] = 0.01 * np.sin(2 * np.pi * 0.1 * sim_time)
        scene["Gimbal"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # 每100帧保存一次相机图像（只保存第0个环境）
        if count % 100 == 0:
            rgb_data = scene["camera"].data.output["rgb"]  # (num_envs, H, W, 4)
            if rgb_data is not None:
                rgb_np = rgb_data[0].cpu().numpy()[:, :, :3]  # 取第0个环境，去掉alpha
                img = Image.fromarray(rgb_np.astype(np.uint8))
                img.save(f"camera_output/frame_{count:06d}.png")
                print(f"[INFO]: Saved camera_output/frame_{count:06d}.png")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([10.5, 10.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()