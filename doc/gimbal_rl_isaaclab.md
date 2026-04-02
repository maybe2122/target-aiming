# 云台视觉追踪强化学习方案（基于 Isaac Lab）

---

## 一、任务建模

| 要素 | 描述 |
|------|------|
| **输入 (Observation)** | 相机图像（RGB / instance segmentation）+ 云台关节角度 |
| **输出 (Action)** | 云台控制量 `[Δyaw, Δpitch]`（角速度） |
| **目标 (Goal)** | 让目标出现在图像中心 |
| **奖励 (Reward)** | 目标越接近图像中心，奖励越高 |

本质是一个**基于视觉的连续控制问题（vision-based continuous control）**。

---

## 二、在 Isaac Lab 中搭建仿真环境

### 1. 工作流选择：Direct Workflow

视觉输入的 RL 环境**推荐使用 Direct Workflow**（`DirectRLEnv`），因为：

- Manager-Based Workflow 的 `mdp` 子模块目前**不包含**相机相关的 observation term。
- Direct Workflow 可以直接在 `_get_observations()` 中访问 `TiledCamera` 数据。

参考示例：`Isaac-Cartpole-RGB-Camera-Direct-v0`

```bash
# 训练时必须加 --enable_cameras 启用离屏渲染
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Cartpole-RGB-Camera-Direct-v0 \
    --headless --enable_cameras
```

---

### 2. 环境配置类（`@configclass`）

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

@configclass
class GimbalEnvCfg(DirectRLEnvCfg):
    # 仿真设置
    decimation = 2
    episode_length_s = 10.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # 动作空间：[Δyaw, Δpitch]，归一化到 [-1, 1]
    action_space = 2
    # 观测空间：图像像素 + 关节角度
    # 84×84 grayscale → 7056 dims，或使用 segmentation
    observation_space = 84 * 84 + 2
    state_space = 0

    # 相机配置（使用 TiledCamera 支持并行渲染）
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Gimbal/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # 四元数 (w, x, y, z)，ROS 惯例
            convention="ros",
        ),
        # 推荐初期使用 instance_segmentation_fast，训练更稳定
        data_types=["instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 50.0),
        ),
        width=84,
        height=84,
    )
```

> **关键说明：**
> - `TiledCamera` 使用 RTX 渲染器的 **Tiled Rendering** 技术，将所有并行环境的相机帧拼接为一张大图进行批量渲染，极大减少 CPU-GPU 带宽开销，是 RL 场景下的推荐方案。
> - 支持的 `data_types`：`"rgb"`, `"rgba"`, `"depth"`, `"distance_to_image_plane"`, `"instance_segmentation_fast"`, `"semantic_segmentation"` 等。
> - RGB 数据类型返回 `torch.uint8`，shape 为 `(N, H, W, 3)`；使用前需除以 `255.0` 转换为 `float32`。

---

### 3. 云台模型（`Articulation`）

云台为两自由度关节结构（yaw + pitch），使用 `ArticulationCfg` 加载：

```python
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

GIMBAL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Gimbal",
    spawn=sim_utils.UsdFileCfg(
        usd_path="path/to/gimbal.usd",
    ),
    actuators={
        "yaw_joint": ...,
        "pitch_joint": ...,
    },
)
```

> 如需从 URDF 导入，参考官方文档 [Importing a New Asset](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)。

---

## 三、强化学习环境设计

### 1. Observation（观测）

**推荐方案：图像 + 关节角度**

```python
def _get_observations(self) -> dict:
    # 获取 segmentation 图像，shape: (N, H, W, 1)，类型 uint8
    seg_img = self.tiled_camera.data.output["instance_segmentation_fast"]

    # 展平为 (N, H*W)，归一化到 [0, 1]
    img_flat = seg_img.float().view(self.num_envs, -1) / 255.0

    # 获取关节角度 (N, 2)
    joint_pos = self.gimbal.data.joint_pos  # [yaw, pitch]

    obs = torch.cat([img_flat, joint_pos], dim=-1)
    return {"policy": obs}
```

> 加入关节角度可以**显著加速收敛**，避免纯视觉策略的探索困难问题。

---

### 2. Action（动作）

使用**速度控制**（增量控制），输出归一化角速度：

```python
def _pre_physics_step(self, actions: torch.Tensor):
    # actions: (N, 2)，范围 [-1, 1]
    self._actions = actions.clone()

def _apply_action(self):
    # 映射到实际角速度，单位 rad/s
    max_vel = 1.0  # rad/s
    velocity_targets = self._actions * max_vel
    self.gimbal.set_joint_velocity_target(velocity_targets)
```

---

### 3. Reward（奖励函数）

核心思路：**让目标像素坐标趋近图像中心**

设图像分辨率为 `(W, H)`，目标像素坐标为 `(u, v)`，图像中心为 `(cx, cy) = (W/2, H/2)`：

```python
def _get_rewards(self) -> torch.Tensor:
    cx, cy = self.cfg.tiled_camera.width / 2, self.cfg.tiled_camera.height / 2

    # 归一化像素误差，范围 [0, 1]
    pixel_error = torch.sqrt(
        ((self.target_u - cx) / cx) ** 2 +
        ((self.target_v - cy) / cy) ** 2
    )

    # 距离惩罚
    r_distance = -pixel_error

    # 平滑惩罚（防抖）
    r_smooth = -0.01 * torch.sum(self._actions ** 2, dim=-1)

    # 完成奖励
    r_done = torch.where(pixel_error < 0.05, torch.ones_like(pixel_error) * 10.0, torch.zeros_like(pixel_error))

    return r_distance + r_smooth + r_done
```

**奖励组合总结：**

```
reward = -归一化像素误差              # 核心：驱动对准
       - 0.01 × ||action||²          # 平滑：防抖动
       + 10.0 × (error < threshold)  # 稀疏完成奖励
```

---

### 4. 终止条件（`_get_dones`）

```python
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    # 超时终止
    time_out = self.episode_length_buf >= self.max_episode_length - 1

    # 成功终止（可选）
    success = pixel_error < 0.05

    return success, time_out
```

---

## 四、视觉处理策略

### 路线一：感知 + 控制分离（推荐，适合初期）

```
Segmentation Mask → 提取目标坐标 (u, v) → RL 策略输入 (u, v, yaw, pitch)
```

优点：收敛快、稳定、工程常用  
适用于：快速验证控制逻辑

```python
# 从 segmentation mask 中提取目标质心
def extract_target_center(seg_mask, target_id):
    # seg_mask: (N, H, W, 1)
    mask = (seg_mask[..., 0] == target_id).float()  # (N, H, W)
    h_coords = torch.arange(H, device=mask.device).view(1, H, 1)
    w_coords = torch.arange(W, device=mask.device).view(1, 1, W)
    pixel_count = mask.sum(dim=[1, 2]).clamp(min=1)
    u = (mask * w_coords).sum(dim=[1, 2]) / pixel_count  # (N,)
    v = (mask * h_coords).sum(dim=[1, 2]) / pixel_count  # (N,)
    return u, v
```

---

### 路线二：端到端 RL（End-to-End）

```
TiledCamera RGB/Seg → CNN 特征提取 → PPO 策略
```

优点：无需手工特征  
缺点：训练慢，需要更多环境并行数

```python
# CNN 特征提取示例（在策略网络中）
import torch.nn as nn

class CameraEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),  # segmentation 单通道
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
    def forward(self, x):
        # x: (N, H, W) → (N, 1, H, W)
        return self.cnn(x.unsqueeze(1))
```

---

## 五、Domain Randomization（领域随机化）

使用 `EventTermCfg` 配置随机化，提升 sim-to-real 能力：

```python
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import mdp
from isaaclab.utils import configclass

@configclass
class EventCfg:
    # 物理属性随机化
    gimbal_physics = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("gimbal", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # 目标位置随机化（每个 episode 重置）
    reset_target_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (0.5, 2.0)},
            "velocity_range": {},
        },
    )
```

> **视觉随机化**：Isaac Lab 通过 Replicator API 支持 MDL 材质、场景光照的随机化，提升外观泛化能力（参考 `omni.replicator.core`）。

---

## 六、强化学习算法选择

Isaac Lab 支持多个主流 RL 库，均可通过 Wrapper 直接对接：

| 库 | 算法 | 适用场景 |
|----|------|---------|
| **RSL-RL** | PPO | 首选，Isaac Lab 原生支持 |
| **RL-Games** | PPO / SAC | 高性能，支持 LSTM |
| **Stable-Baselines3** | PPO / SAC / TD3 | 调试友好 |
| **SKRL** | PPO / SAC / TD3 | 多智能体支持 |

**视觉 RL 推荐：PPO + CNN Encoder（端到端）**或**PPO + 坐标输入（感知分离）**

---

## 七、训练与启动

```bash
# 训练（必须开启相机）
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Gimbal-Camera-Direct-v0 \
    --num_envs 512 \
    --headless \
    --enable_cameras

# 可视化（回放）
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task=Isaac-Gimbal-Camera-Direct-v0 \
    --num_envs 16 \
    --enable_cameras
```

> **性能参考**：Isaac Lab 官方建议在 RTX 4090 上运行约 512 个并行相机环境（84×84 分辨率）。

---

## 八、常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 收敛慢 | 纯图像输入探索困难 | 使用 segmentation + 加入关节角度作为观测 |
| 控制抖动 | 动作惩罚缺失 | reward 中加 `- α × ||action||²` |
| 完全学不会 | 目标随机范围过大 | Curriculum Learning：从固定目标开始，逐步增加随机范围 |
| sim-to-real 失败 | 过拟合仿真外观 | Domain Randomization（光照、材质、噪声注入） |
| 相机数据为空 | 未启用 RTX 渲染 | 启动命令加 `--enable_cameras` |

---

## 九、推荐开发路径

### 阶段一：最小可行（坐标输入）
- Segmentation 提取目标坐标 `(u, v)`
- 输入 `(u, v, yaw, pitch)` 到 MLP 策略
- 目标：系统能稳定对准固定目标 ✅

### 阶段二：视觉输入
- 替换为 `TiledCamera` RGB/Seg 图像
- CNN Encoder + PPO
- 目标：端到端视觉对准 ✅

### 阶段三：鲁棒性提升
- 加入 Domain Randomization（物理 + 视觉）
- 目标位置完全随机化
- 噪声注入（`observation_noise_model`）
- 目标：具备 sim-to-real 迁移能力 ✅

---

## 十、整体架构

```
TiledCamera
    │
    ▼
Segmentation / RGB
    │
    ├─── [路线A] CV提取 (u, v) ──► MLP Policy ──► [Δyaw, Δpitch] ──► Gimbal Articulation
    │
    └─── [路线B] CNN Encoder ─────► PPO Policy ──► [Δyaw, Δpitch] ──► Gimbal Articulation
```

---

## 参考资料

- [Isaac Lab 官方文档 - Camera Sensor](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html)
- [Isaac Lab 官方文档 - Tiled Rendering](https://isaac-sim.github.io/IsaacLab/main/source/features/tiled_rendering.html)
- [Isaac Lab 官方文档 - Direct Workflow RL Environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)
- [Isaac Lab 官方文档 - Domain Randomization](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)
- [Isaac Lab 官方文档 - Adding Sensors on Robot](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/04_sensors/add_sensors_on_robot.html)
- 参考任务源码：`isaaclab_tasks/direct/cartpole/cartpole_camera_env.py`
