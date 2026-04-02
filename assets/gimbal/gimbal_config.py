# gimbal_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

GIMBAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sz/code/rl/target_aiming/assets/gimbal/gimbal.usd",
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
        pos=(0.0, 0.0, 0.0),           # 相对于环境原点的位置
        joint_pos={
            "yaw_joint": 0.0,           # 初始 yaw 角（弧度）
            "pitch_joint": 0.0,         # 初始 pitch 角（弧度）
        },
    ),
    actuators={
        # 速度控制模式：stiffness=0, damping>0
        "gimbal_joints": ImplicitActuatorCfg(
            joint_names_expr=["yaw_joint", "pitch_joint"],
            stiffness=0.0,      # 速度控制不需要位置刚度
            damping=100.0,      # damping 决定速度跟踪能力
            effort_limit=10.0,
            velocity_limit=3.14,
        ),
    },
)