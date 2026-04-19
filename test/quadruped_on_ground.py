# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
默认地面上加载 ANYmal-C 四足 + Franka 机械臂，使用预训练 locomotion 策略自由行走。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quadruped + arm on default ground.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import io
import torch

import omni

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
    AnymalCRoughEnvCfg_PLAY,
)


POLICY_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/HeightScan/policy.pt"

# SO-101 机械臂 USD（Isaac Nucleus 下）
ARM_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd"

ROBOT_SPAWN_POS = (0.0, 0.0, 0.6)

# 前进指令 [lin_vel_x, lin_vel_y, ang_vel_z]
VELOCITY_COMMAND = (1.0, 0.0, 0.0)

# 机械臂相对狗底盘的偏移（狗 base 坐标系下）和缩放
ARM_OFFSET = (-0.1, 0.0, 0.25)
ARM_SCALE = 1.0


SO101_ARM_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Arm",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARM_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
)


def build_env_cfg() -> AnymalCRoughEnvCfg_PLAY:
    cfg = AnymalCRoughEnvCfg_PLAY()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 4.0
    cfg.curriculum = None

    # 默认平地地形
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    cfg.scene.robot.init_state.pos = ROBOT_SPAWN_POS

    # DomeLight
    cfg.scene.light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )

    # SO-101 机械臂（独立 Articulation，每步重写根位姿跟随四足）
    cfg.scene.arm = SO101_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Arm")
    cfg.scene.arm.init_state.pos = (
        ROBOT_SPAWN_POS[0] + ARM_OFFSET[0],
        ROBOT_SPAWN_POS[1] + ARM_OFFSET[1],
        ROBOT_SPAWN_POS[2] + ARM_OFFSET[2],
    )
    cfg.scene.arm.spawn.scale = (ARM_SCALE, ARM_SCALE, ARM_SCALE)
    cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        cfg.sim.use_fabric = False

    return cfg


def load_policy(device: str):
    content = omni.client.read_file(POLICY_PATH)[2]
    buf = io.BytesIO(memoryview(content).tobytes())
    return torch.jit.load(buf, map_location=device)


def main():
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = build_env_cfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    policy = load_policy(args_cli.device)

    cmd_term = env.command_manager.get_term("base_velocity")
    cmd_tensor = torch.tensor([VELOCITY_COMMAND], device=env.device).repeat(env.num_envs, 1)

    obs, _ = env.reset()

    robot = env.scene["robot"]
    arm = env.scene["arm"]
    arm_offset = torch.tensor(ARM_OFFSET, device=env.device)
    robot_pos = robot.data.root_pos_w[0].cpu().numpy()
    print(f"[INFO]: robot root world pos = {robot_pos}")

    env.sim.set_camera_view(
        eye=(robot_pos[0] + 4.0, robot_pos[1] + 4.0, robot_pos[2] + 2.0),
        target=(robot_pos[0], robot_pos[1], robot_pos[2]),
    )

    def follow_robot():
        """把机械臂根位姿贴到狗 base 上（base 坐标系下偏移 ARM_OFFSET）。"""
        from isaaclab.utils.math import quat_rotate

        base_pos = robot.data.root_pos_w  # (N, 3)
        base_quat = robot.data.root_quat_w  # (N, 4) wxyz
        world_offset = quat_rotate(base_quat, arm_offset.expand_as(base_pos))
        arm_pos = base_pos + world_offset
        pose = torch.cat([arm_pos, base_quat], dim=-1)
        zeros = torch.zeros((base_pos.shape[0], 6), device=env.device)
        arm.write_root_pose_to_sim(pose)
        arm.write_root_velocity_to_sim(zeros)

    follow_robot()

    step_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            cmd_term.command[:] = cmd_tensor
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
            follow_robot()

            step_count += 1
            if step_count % 50 == 0:
                p = robot.data.root_pos_w[0].cpu().numpy()
                env.sim.set_camera_view(
                    eye=(p[0] + 4.0, p[1] + 4.0, p[2] + 2.0),
                    target=(p[0], p[1], p[2]),
                )


if __name__ == "__main__":
    main()
    simulation_app.close()
