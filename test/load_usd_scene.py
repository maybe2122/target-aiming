# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
在 USD 场景中加载 ANYmal-C 四足 + Franka 机械臂，使用预训练 locomotion 策略自由行走。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quadruped + arm in USD scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import io
import os
import torch

import omni
from pxr import UsdPhysics, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
    AnymalCRoughEnvCfg_PLAY,
)
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


# TODO: 填入你的 USD 场景路径
USD_SCENE_PATH = "/media/maybe/新加卷/ue/usdmp/2048usda/Showreel_Scene.usda"

# ANYmal-C 的 HeightScan 预训练策略
POLICY_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/HeightScan/policy.pt"

# 四足初始位置（在 USD 场景里找个空旷的落脚点，按需调整）
ROBOT_SPAWN_POS = (0.0, 0.0, 10.0)

# 前进指令 [lin_vel_x, lin_vel_y, ang_vel_z]
VELOCITY_COMMAND = (1.0, 0.0, 0.0)


def build_env_cfg() -> AnymalCRoughEnvCfg_PLAY:
    """基于 Isaac-Velocity-Rough-Anymal-C PLAY 环境，把地形换成用户的 USD。"""
    cfg = AnymalCRoughEnvCfg_PLAY()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 8.0
    cfg.curriculum = None

    # 用 USD 作为带碰撞的地形
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=USD_SCENE_PATH,
        collision_group=-1,
    )

    # 机器人初始位置
    cfg.scene.robot.init_state.pos = ROBOT_SPAWN_POS

    # 追加一盏 DomeLight（替换原 distant light）
    cfg.scene.light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )

    # 在四足背上挂一只 Franka 机械臂（作为独立 Articulation）
    # 注意：这是简化挂载，机械臂不会物理跟随四足移动。
    # 真正刚性绑定需要自定义 USD（四足 + 机械臂 + fixed joint）。
    cfg.scene.arm = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Arm",
    )
    cfg.scene.arm.init_state.pos = (
        ROBOT_SPAWN_POS[0],
        ROBOT_SPAWN_POS[1],
        ROBOT_SPAWN_POS[2] + 0.3,  # 粗略放在四足背部上方
    )
    # 让机械臂底座固定在空中（不会掉落），便于先跑通流程
    cfg.scene.arm.spawn.rigid_props.disable_gravity = True

    cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        cfg.sim.use_fabric = False

    return cfg


def load_policy(device: str):
    content = omni.client.read_file(POLICY_PATH)[2]
    buf = io.BytesIO(memoryview(content).tobytes())
    return torch.jit.load(buf, map_location=device)


def apply_collision_to_all_meshes(root_prim_path: str, approximation: str = "meshSimplification"):
    """递归给 root_prim_path 下的所有 Mesh 应用碰撞 API。

    approximation 可选: "none"(三角网格,最精确但慢)、"convexHull"、"convexDecomposition"、
    "meshSimplification"、"boundingCube"、"boundingSphere"。
    复杂静态场景推荐 "meshSimplification" 或 "none"(地面走路用 "none" 最准)。
    """
    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        print(f"[WARN]: Prim {root_prim_path} not found; skip collision setup.")
        return
    count = 0
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root.GetPath()):
            continue
        if prim.GetTypeName() != "Mesh":
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set(approximation)
        count += 1
    print(f"[INFO]: Applied collision to {count} meshes under {root_prim_path}.")


def main():
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = build_env_cfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 给 USD 场景里所有 Mesh 应用碰撞（UE/Blender 导出的 USD 通常没带物理）
    apply_collision_to_all_meshes("/World/ground", approximation="none")

    policy = load_policy(args_cli.device)

    # 把默认的速度指令覆盖为恒定向前
    cmd_term = env.command_manager.get_term("base_velocity")
    cmd_tensor = torch.tensor([VELOCITY_COMMAND], device=env.device).repeat(env.num_envs, 1)

    obs, _ = env.reset()

    # 打印机器人实际位置并把相机对准它
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[0].cpu().numpy()
    print(f"[INFO]: env_origins = {env.scene.env_origins[0].cpu().numpy()}")
    print(f"[INFO]: robot root world pos = {robot_pos}")

    env.sim.set_camera_view(
        eye=(robot_pos[0] + 4.0, robot_pos[1] + 4.0, robot_pos[2] + 2.0),
        target=(robot_pos[0], robot_pos[1], robot_pos[2]),
    )

    step_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            # 每次步进都锁定指令，避免 resample 改掉
            cmd_term.command[:] = cmd_tensor
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)

            # 每 50 步把相机跟到机器人身上
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
