# VLA 在线强化学习微调方案（修订版）

## 项目概述

VLA 模型已通过 SFT 完成初步训练（输入：图片 → 输出：yaw, pitch, zoom），但动作精度不足。本方案在 Isaac Sim 中搭建在线 RL 环境，**将 VLA 直接作为策略网络**，通过 PPO + LoRA 进行在线微调，利用仿真奖励信号提升动作质量。

### 与原方案的核心变更

| 项目 | 原方案 | 修订方案 |
|------|--------|----------|
| VLA 角色 | Phase 3 教师（蒸馏） | **策略网络本体**，直接参与 RL |
| 微调方式 | 训练独立学生策略 | **LoRA 微调 VLA 自身** |
| 动作空间 | 2D (yaw, pitch) | **3D (yaw, pitch, zoom)** |
| zoom 实现 | 无 | **控制相机 focal_length** |
| 训练架构 | 单进程 | **VLA 推理服务 + Isaac Sim 训练进程分离** |

---

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                   Isaac Sim 训练进程                      │
│                                                         │
│  TiledCamera ──→ RGB (N,H,W,3) ──→ HTTP Client ──────┐ │
│                                                       │ │
│       ┌──── actions (yaw, pitch, zoom) ◄──────────────┘ │
│       │                                                 │
│       ▼                                                 │
│  Gimbal Joint Control (yaw, pitch)                      │
│  Camera focal_length Control (zoom)                     │
│       │                                                 │
│       ▼                                                 │
│  Reward Computation ──→ PPO Gradient ──→ HTTP ──────────┤
│  (像素误差 + 平滑 + 存活)    (传回 LoRA 梯度)            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│               VLA 推理服务 (独立 GPU 进程)                │
│                                                         │
│  VLA Model (frozen backbone + LoRA adapters)            │
│       │                                                 │
│  /predict    ← 接收图片，返回 (yaw, pitch, zoom)         │
│  /gradient   ← 接收 PPO loss，返回更新后的 LoRA 权重      │
│  /checkpoint ← 保存/加载 LoRA 权重                       │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1：坐标投影 RL 基线（保留不变）

Phase 1 作为环境验证基线保留，不做修改。确认仿真环境、奖励函数、Gimbal 控制逻辑正确后再进入 Phase 2。

---

## Phase 2：VLA 在线 RL 微调

### 2.1 动作空间扩展：zoom 轴

**zoom 通过运行时修改相机 `focal_length` 实现**，不需要机械变焦关节。

```python
# target_aiming_env_visual.py

class TargetAimingVisualEnv(DirectRLEnv):
    """VLA 在线 RL 微调环境，支持 yaw/pitch/zoom 三轴控制。"""

    # zoom 参数范围
    FOCAL_LENGTH_MIN = 6.0    # 广角端
    FOCAL_LENGTH_MAX = 50.0   # 长焦端
    FOCAL_LENGTH_DEFAULT = 12.0

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 当前焦距状态 (N,)
        self._current_focal_length = torch.full(
            (self.num_envs,), self.FOCAL_LENGTH_DEFAULT,
            device=self.device, dtype=torch.float32
        )

    def _apply_action(self, actions: torch.Tensor):
        """
        actions: (N, 3) — [delta_yaw_vel, delta_pitch_vel, delta_zoom]
        delta_zoom ∈ [-1, 1]，映射为焦距增量。
        """
        # yaw, pitch → 关节速度目标（与 Phase 1 相同）
        yaw_vel = actions[:, 0] * self.cfg.yaw_speed_scale
        pitch_vel = actions[:, 1] * self.cfg.pitch_speed_scale

        joint_vel_targets = torch.zeros(self.num_envs, 2, device=self.device)
        joint_vel_targets[:, 0] = yaw_vel
        joint_vel_targets[:, 1] = pitch_vel
        self.gimbal.set_joint_velocity_target(joint_vel_targets)

        # zoom → 修改 focal_length
        zoom_delta = actions[:, 2] * self.cfg.zoom_speed_scale  # e.g. scale=2.0
        self._current_focal_length = torch.clamp(
            self._current_focal_length + zoom_delta,
            self.FOCAL_LENGTH_MIN,
            self.FOCAL_LENGTH_MAX,
        )
        self._update_camera_focal_lengths()

    def _update_camera_focal_lengths(self):
        """逐环境设置相机焦距。

        Isaac Sim 中 TiledCamera 不直接支持批量修改 focal_length，
        需要通过 USD API 逐 prim 设置。可在子步之间调用。
        """
        for i in range(self.num_envs):
            camera_prim_path = f"/World/envs/env_{i}/Gimbal/pitch_link/camera"
            focal_attr = self.camera._sensor_prims[i].GetAttribute("focalLength")
            focal_attr.Set(float(self._current_focal_length[i]))

    def _get_observations(self) -> dict:
        rgb = self.camera.data.output["rgb"]  # (N, H, W, 3) uint8
        rgb_norm = rgb.float() / 255.0
        return {
            "policy": {
                "image": rgb_norm,                          # (N, H, W, 3)
                "joint_state": torch.stack([
                    self.gimbal.data.joint_pos[:, self._yaw_idx],
                    self.gimbal.data.joint_pos[:, self._pitch_idx],
                    self._current_focal_length / self.FOCAL_LENGTH_MAX,  # 归一化
                ], dim=-1),                                 # (N, 3)
            }
        }
```

### 2.2 奖励函数修改

新增 zoom 相关奖励项，鼓励目标在画面中占据合理比例：

```python
def _compute_rewards(self) -> torch.Tensor:
    # === 原有奖励 ===
    # 1. 像素误差惩罚（目标中心 vs 画面中心）
    pixel_error = self._compute_pixel_error()  # (N,)
    r_center = -self.cfg.center_weight * pixel_error

    # 2. 动作平滑惩罚
    action_diff = self.actions - self.prev_actions
    r_smooth = -self.cfg.smooth_weight * action_diff.pow(2).sum(dim=-1)

    # 3. 存活奖励
    r_alive = self.cfg.alive_weight

    # === 新增 zoom 奖励 ===
    # 4. 目标尺寸奖励：鼓励目标在画面中占合理比例
    #    target_size_ratio: 目标 bbox 面积 / 画面面积，理想值 ~0.1-0.3
    target_ratio = self._compute_target_size_ratio()  # (N,)
    r_zoom = -self.cfg.zoom_weight * (target_ratio - self.cfg.target_ratio_ideal).pow(2)

    # 5. zoom 变化平滑惩罚
    zoom_change = actions[:, 2].pow(2)
    r_zoom_smooth = -self.cfg.zoom_smooth_weight * zoom_change

    return r_center + r_smooth + r_alive + r_zoom + r_zoom_smooth
```

### 2.3 VLA 推理服务设计

VLA 作为独立服务部署，提供前向推理和梯度回传两个接口。

**服务端 `vla_rl_server.py`：**

```python
"""VLA RL 微调服务端 — 提供推理 + LoRA 梯度更新接口。"""
import torch
from fastapi import FastAPI
from peft import get_peft_model, LoraConfig

app = FastAPI()

# ---- 模型初始化 ----
base_model = load_vla_model("path/to/sft_checkpoint")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "action_head.linear"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 预期 < 1% 参数可训练

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-5, weight_decay=0.01
)


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    输入: batch of RGB images (base64 编码)
    输出: actions (N, 3) — [yaw, pitch, zoom]

    前向推理，保留计算图用于后续梯度回传。
    """
    images = decode_images(request.images)            # (N, H, W, 3)
    with torch.enable_grad():
        actions, log_probs, values = model.forward_rl(
            images, instruction=request.instruction
        )
    # 缓存当前 step 的计算图
    step_cache[request.step_id] = {
        "actions": actions,
        "log_probs": log_probs,
        "values": values,
    }
    return {
        "actions": actions.detach().cpu().tolist(),
        "log_probs": log_probs.detach().cpu().tolist(),
        "values": values.detach().cpu().tolist(),
    }


@app.post("/update")
async def update(request: UpdateRequest):
    """
    接收 PPO 计算好的 loss 组件，执行反向传播更新 LoRA 权重。

    输入: step_id, advantages, returns, old_log_probs, clip_param
    """
    cached = step_cache.pop(request.step_id)

    # PPO loss 计算
    ratio = torch.exp(cached["log_probs"] - request.old_log_probs)
    surr1 = ratio * request.advantages
    surr2 = torch.clamp(ratio, 1 - request.clip_param, 1 + request.clip_param) * request.advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = 0.5 * (cached["values"] - request.returns).pow(2).mean()

    loss = policy_loss + request.value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


@app.post("/checkpoint")
async def save_checkpoint(request: CheckpointRequest):
    """保存 LoRA adapter 权重。"""
    model.save_pretrained(request.save_path)
    return {"status": "saved", "path": request.save_path}
```

**客户端 `vla_rl_client.py`（Isaac Sim 侧）：**

```python
"""Isaac Sim 端的 VLA 客户端，替代本地策略网络。"""
import torch
import requests
import numpy as np
from PIL import Image
import io, base64


class VLARLClient:
    """封装 VLA 服务调用，对 Isaac Sim RL 训练循环透明。"""

    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        self.server_url = server_url
        self.session = requests.Session()
        self.step_counter = 0

    def predict(self, rgb_tensor: torch.Tensor, instruction: str) -> dict:
        """
        Args:
            rgb_tensor: (N, H, W, 3) float32 [0, 1]
            instruction: 任务指令文本
        Returns:
            dict with 'actions', 'log_probs', 'values'
        """
        images_b64 = self._encode_batch(rgb_tensor)
        step_id = self.step_counter
        self.step_counter += 1

        resp = self.session.post(f"{self.server_url}/predict", json={
            "images": images_b64,
            "instruction": instruction,
            "step_id": step_id,
        })
        result = resp.json()
        return {
            "actions": torch.tensor(result["actions"], device=rgb_tensor.device),
            "log_probs": torch.tensor(result["log_probs"], device=rgb_tensor.device),
            "values": torch.tensor(result["values"], device=rgb_tensor.device),
            "step_id": step_id,
        }

    def update(self, step_id, advantages, returns, old_log_probs, clip_param=0.2):
        """将 PPO 梯度信息发回服务端执行更新。"""
        resp = self.session.post(f"{self.server_url}/update", json={
            "step_id": step_id,
            "advantages": advantages.cpu().tolist(),
            "returns": returns.cpu().tolist(),
            "old_log_probs": old_log_probs.cpu().tolist(),
            "clip_param": clip_param,
            "value_coef": 0.5,
        })
        return resp.json()

    def _encode_batch(self, rgb_tensor: torch.Tensor) -> list:
        """将 (N,H,W,3) float tensor 编码为 base64 列表。"""
        images_b64 = []
        for i in range(rgb_tensor.shape[0]):
            img_np = (rgb_tensor[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            images_b64.append(base64.b64encode(buf.getvalue()).decode())
        return images_b64
```

### 2.4 训练循环（Isaac Sim 侧）

由于 VLA 在远端服务上，Isaac Sim 侧的训练循环需要自定义，不能直接使用 RSL-RL 的 `OnPolicyRunner`：

```python
"""自定义 PPO 训练循环 — 适配远端 VLA 推理服务。"""

class VLAOnlineTrainer:
    """
    核心循环：
    1. Isaac Sim step → 渲染图像
    2. 图像发送至 VLA 服务 → 获取 action, log_prob, value
    3. action 应用到仿真 → 获取 reward, done
    4. 收集 rollout buffer
    5. 计算 GAE advantage
    6. 将 PPO 更新信号发送至 VLA 服务
    """

    def __init__(self, env, client: VLARLClient, cfg):
        self.env = env
        self.client = client
        self.cfg = cfg
        self.instruction = "Track the vehicle and keep it centered in frame."

    def train(self):
        obs = self.env.reset()

        for iteration in range(self.cfg.max_iterations):
            # ---- Rollout Phase ----
            rollout = self._collect_rollout(obs)

            # ---- Compute GAE ----
            advantages, returns = self._compute_gae(rollout)

            # ---- PPO Update (发送至 VLA 服务) ----
            for epoch in range(self.cfg.num_epochs):
                for mb in self._make_minibatches(rollout, advantages, returns):
                    loss_info = self.client.update(
                        step_id=mb["step_id"],
                        advantages=mb["advantages"],
                        returns=mb["returns"],
                        old_log_probs=mb["old_log_probs"],
                        clip_param=self.cfg.clip_param,
                    )

            # ---- Logging ----
            if iteration % self.cfg.log_interval == 0:
                self._log_metrics(iteration, rollout, loss_info)

            # ---- Checkpoint ----
            if iteration % self.cfg.save_interval == 0:
                self.client.session.post(
                    f"{self.client.server_url}/checkpoint",
                    json={"save_path": f"checkpoints/iter_{iteration}"}
                )

    def _collect_rollout(self, obs):
        """收集 N 步 rollout 数据。"""
        buffer = {
            "obs": [], "actions": [], "rewards": [],
            "dones": [], "log_probs": [], "values": [], "step_ids": []
        }

        for step in range(self.cfg.num_steps_per_env):
            # VLA 前向推理
            pred = self.client.predict(obs["policy"]["image"], self.instruction)

            actions = pred["actions"]  # (N, 3)

            # 仿真 step
            next_obs, rewards, dones, infos = self.env.step(actions)

            buffer["obs"].append(obs)
            buffer["actions"].append(actions)
            buffer["rewards"].append(rewards)
            buffer["dones"].append(dones)
            buffer["log_probs"].append(pred["log_probs"])
            buffer["values"].append(pred["values"])
            buffer["step_ids"].append(pred["step_id"])

            obs = next_obs

        return buffer

    def _compute_gae(self, rollout):
        """广义优势估计 (GAE-Lambda)。"""
        gamma = self.cfg.gamma        # 0.99
        lam = self.cfg.gae_lambda     # 0.95
        rewards = torch.stack(rollout["rewards"])   # (T, N)
        values = torch.stack(rollout["values"])     # (T, N)
        dones = torch.stack(rollout["dones"])       # (T, N)

        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = self.client.predict(
                    rollout["obs"][-1]["policy"]["image"], self.instruction
                )["values"]
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns
```

### 2.5 环境配置 `target_aiming_env_cfg_visual.py`

```python
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

@configclass
class TargetAimingVisualEnvCfg(DirectRLEnvCfg):
    """VLA 在线 RL 微调环境配置。"""

    # ---- 相机（复用 spawncargimbal.py 验证过的参数）----
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Gimbal/pitch_link/camera",
        update_period=0.0,
        height=224,          # VLA 通常需要 224x224 输入
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,              # 初始焦距，运行时动态修改
            horizontal_aperture=20.955,
            clipping_range=(0.1, 50.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # ---- 动作空间 ----
    action_space = 3  # yaw_vel, pitch_vel, zoom_delta
    yaw_speed_scale = 1.0
    pitch_speed_scale = 1.0
    zoom_speed_scale = 2.0  # focal_length 每步最大变化量

    # ---- 奖励权重 ----
    center_weight = 10.0
    smooth_weight = 0.1
    alive_weight = 0.5
    zoom_weight = 2.0
    zoom_smooth_weight = 0.05
    target_ratio_ideal = 0.15  # 目标占画面比例理想值

    # ---- 训练 ----
    episode_length_s = 15.0
    decimation = 2
    num_envs = 128  # 受 VLA 推理吞吐限制，不宜过大
```

---

## Phase 3：部署优化（可选）

VLA 在线微调完成后，如果部署时延迟不满足要求（VLA 推理 > 30ms），可将微调后的 VLA 作为教师，蒸馏到轻量学生网络：

```
微调后的 VLA (LoRA merged) → Teacher
         │
         ▼  MSE 蒸馏
Student MLP (3层, 256-128-64) → 部署推理 < 5ms
```

这一步复用原方案 Phase 3 的 `DistillationRunner` 框架即可。

---

## 训练流程命令

```bash
# ===== 步骤 1：启动 VLA RL 推理服务 =====
# 在 GPU-0 上启动 VLA 服务（LoRA 模式）
CUDA_VISIBLE_DEVICES=0 python vla_rl_server.py \
    --checkpoint path/to/sft_checkpoint \
    --lora_r 16 \
    --lora_alpha 32 \
    --port 8000

# ===== 步骤 2：启动 Isaac Sim 训练 =====
# 在 GPU-1 上启动仿真环境和训练循环
CUDA_VISIBLE_DEVICES=1 python train_vla_online.py \
    --task=TargetAiming-Direct-Visual-v0 \
    --num_envs=128 \
    --max_iterations=3000 \
    --vla_server_url=http://127.0.0.1:8000 \
    --enable_cameras \
    --device=cuda

# ===== 步骤 3（可选）：合并 LoRA 并导出 =====
python merge_lora.py \
    --base_model path/to/sft_checkpoint \
    --lora_adapter checkpoints/iter_3000 \
    --output path/to/merged_model
```

---

## 关键设计决策与说明

### 为什么 VLA 作为独立服务？

VLA 模型体量大（通常 7B+ 参数），与 Isaac Sim 渲染共享 GPU 显存容易 OOM。分离部署允许各自独占 GPU，且 VLA 服务可以水平扩展（多副本）应对并行环境数增加。

### LoRA 微调的优势

LoRA 只更新约 0.5-1% 的参数，保留 SFT 预训练的视觉理解能力，避免灾难性遗忘。微调后可以通过 merge 得到标准模型权重，部署时无额外开销。

### 并行环境数的权衡

VLA 推理延迟（约 50-100ms/batch）是瓶颈。128 个并行环境意味着每个 RL step 需要等待一次 VLA 推理。可以通过以下方式缓解：

- 批量推理：VLA 服务端一次处理整个 batch
- 异步流水线：Isaac Sim 渲染下一帧的同时，VLA 处理当前帧
- 多副本服务：如有多 GPU，部署多个 VLA 服务实例

### zoom 控制的注意事项

Isaac Sim 中运行时修改 `focal_length` 需要通过 USD API，可能有性能开销。如果逐 prim 设置太慢，可以：

- 降低 zoom 更新频率（每 N 步更新一次）
- 离散化 zoom 档位（如 5 档），减少切换频率
- 先验证单环境下的 focal_length 动态修改是否可行

---

## 文件结构（修订后）

```
source/target_aiming/target_aiming/tasks/direct/target_aiming/
├── target_aiming_env_cfg.py              # Phase 1 配置（保留）
├── target_aiming_env_cfg_visual.py       # Phase 2 VLA-RL 环境配置（修改）
├── target_aiming_env.py                  # Phase 1 环境（保留）
├── target_aiming_env_visual.py           # Phase 2 VLA-RL 环境（修改）
├── vla_rl_client.py                      # VLA 服务客户端（新增，替代 vla_teacher.py）
├── vla_rl_trainer.py                     # 自定义 PPO 训练循环（新增）
└── agents/
    └── rsl_rl_ppo_cfg.py                 # Phase 1 PPO 配置（保留）

vla_service/
├── vla_rl_server.py                      # VLA RL 推理+梯度服务（新增）
├── merge_lora.py                         # LoRA 合并导出脚本（新增）
└── configs/
    └── lora_config.yaml                  # LoRA 超参数配置

scripts/
└── train_vla_online.py                   # 训练入口脚本（新增）
```

---

## 里程碑与评估指标

| Phase | 里程碑 | 成功指标 |
|-------|--------|----------|
| P1 | PPO 坐标投影训练收敛 | 成功率 > 85%，平均步数 > 8s |
| P2-a | VLA 服务联通验证 | Isaac Sim ↔ VLA 服务端到端跑通，action 可控制云台 |
| P2-b | VLA 在线 RL 收敛 | 瞄准成功率（pixel_error < 0.05）较 SFT 提升 ≥ 15% |
| P2-c | zoom 控制生效 | 目标占画面比例稳定在 10-30% |
| P3 | （可选）蒸馏部署 | 学生策略成功率 ≥ P2，推理延迟 < 5ms |

---

## 关键风险与应对

| 风险 | 应对 |
|------|------|
| VLA 推理延迟过高导致训练过慢 | 批量推理 + 异步流水线；降低 num_envs；预渲染图像离线训练 |
| HTTP 传输图像带宽瓶颈 | JPEG 压缩（quality=85）；或改用共享内存 / gRPC + protobuf |
| LoRA 微调导致灾难性遗忘 | 降低 LoRA rank (r=8)；添加 SFT 数据回放 loss 正则化 |
| focal_length 运行时修改不稳定 | 先单环境验证；如不可行，改用裁剪+缩放模拟 zoom |
| PPO 在服务分离架构下梯度不准确 | 服务端保留完整计算图；确保 step_id 对齐无数据错位 |
| 奖励稀疏导致 zoom 学不到有效策略 | 为 zoom 添加课程学习：先固定 zoom 训练 yaw/pitch，再逐步解锁 zoom |