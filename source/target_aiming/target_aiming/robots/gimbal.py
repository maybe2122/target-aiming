import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

GIMBAL_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"assets/gimbal/gimbal.usd"),
    actuators={"gimbal_acts": ImplicitActuatorCfg(joint_names_expr=["gimbal_yaw", "gimbal_pitch"], damping=None, stiffness=None)},
)