# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../assets/gimbal")
)


GIMBAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "gimbal.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "yaw_joint": 0.0,
            "pitch_joint": 0.0,
        },
    ),
    actuators={
        "gimbal_joints": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint", "pitch_joint"],
            stiffness=0.0,
            damping=100.0,
            effort_limit=10.0,
            velocity_limit=3.14,
        ),
    },
)
