# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load a USD scene and add a DomeLight.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


# TODO: 填入你的 USD 场景路径
USD_SCENE_PATH = "/media/maybe/新加卷/ue/usdmp/2048usda/Showreel_Scene.usda"


class SceneCfg(InteractiveSceneCfg):
    """场景：加载 USD 场景并添加 DomeLight。"""

    scene = AssetBaseCfg(
        prim_path="/World/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_SCENE_PATH),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=5000.0, color=(1.0, 1.0, 1.0)),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([5.0, 5.0, 3.0], [0.0, 0.0, 0.5])

    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
