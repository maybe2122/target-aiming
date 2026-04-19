# 训练诊断脚本 (Training Diagnostics)

这个目录里是帮你排查 RL 训练为什么不收敛的工具。

## 诊断概览：当前配置里已经能看出的主要问题

所有数字基于 `target_aiming_env_cfg.py` 和 `rsl_rl_ppo_cfg.py` 的当前值。

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 1 | **动作速度过慢**：`fixed_step_rad=0.03` 作为 velocity target (rad/s)，`decimation=4, dt=1/120` 下每个 policy step 只转 `0.03·(4/120)≈0.001 rad (≈0.057°)`。整个 10 s episode 累计也只能转 `0.03·10=0.3 rad (≈17°)`。 | `target_aiming_env_cfg.py:101` + `target_aiming_env.py:116-122` | reset 时 yaw 随机偏差是 `half_fov·0.8≈24°`，**初始偏差大于整集可转范围**，agent 几乎无法把目标拉回中心，reward 信号自然学不到。 |
| 2 | **出 FOV 立即 terminate**：`out_of_view = ~visible & (pixel_error>0.95)`。`pixel_error` 在不可见时被强制置 1.0，所以条件等价于 `~visible`，一瞬间就 done。 | `target_aiming_env.py:151-156` | 结合问题 1，episode 经常很短，样本里多是失败轨迹，PPO 很难拿到正向信号。 |
| 3 | **pixel 奖励信号弱**：`rew_pixel=10·(prev-cur)`。因为转得慢，每步 `error_delta≈0.003`，信号量 `≈0.03`；而 `rew_idle=-1.0`、`rew_success=5.0`、`rew_alive=0.1`，pixel 信号完全被其他项淹没。 | `target_aiming_env.py:336-338` + cfg 权重 | 梯度方向主要由 idle/success 决定，pixel tracking 信号太弱。 |
| 4 | **reward 尖峰噪声**：可见→不可见的一步 `prev_pixel_error` 可能是 0.1、当前被 clamp 到 1.0，`rew_pixel = 10·(0.1-1.0) = -9`。反过来 reset 后首步又会有 +9 的假进步。 | `target_aiming_env.py:128-142` | PPO 的 critic 很难拟合 value，看 TB 里 `Loss/value_function` 通常会居高不下。 |
| 5 | **CNN 输入无归一化、分辨率偏大**：`obs_normalization=False`，`image=(3,256,256)`。car 在 FOV 大(≈60°)的相机里像素占比很小。 | `rsl_rl_ppo_cfg.py:33,44` + `_env_cfg.py:32-35` | CNN 需要更多样本才能学到稳定的目标表征；建议 128×128 或加 `obs_normalization=True`。 |
| 6 | **PPO batch 偏小 (图像 obs)**：`num_envs=16 (cfg) / 64 (实际 yaml)`，`num_steps_per_env=48`，`num_mini_batches=4`，图像观测下 minibatch 约 192~768。 | `_env_cfg.py:41` + `rsl_rl_ppo_cfg.py:19,54` | 图像策略通常需要 minibatch ≥ 4k，否则 value/surrogate loss 噪声很大。 |
| 7 | **`init_noise_std=0.3` 对 Categorical 无效**：actor 是 `CategoricalDistribution`，这个字段只对高斯生效。无 bug，但容易误以为在调探索。 | `rsl_rl_ppo_cfg.py:35` | 真正控制探索的是 `entropy_coef=0.02`，已经不算低，不是瓶颈。 |

## 推荐先做的最小修复

1. 让动作每步真正能转得动：把 `fixed_step_rad` 改成 "每 policy step 期望转动的角度"，再换算成 velocity：
   ```python
   # 想让每步转 ~0.03 rad，而 policy step = decimation*dt = 0.0333 s
   fixed_step_rad = 0.03
   # 在 _apply_action 里用 velocity = fixed_step_rad / (decimation*dt) ≈ 0.9 rad/s
   ```
2. reset 时不要直接贴 FOV 边缘：`yaw_offset = half_fov*0.4`，并在非可见时用连续惩罚而非直接 done。
3. 暂时把 `rew_scale_idle` 降到 `-0.05` 或 `0`，看 pixel 信号能不能占上风；也可以把 `rew_scale_pixel_error` 提到 `50~100`。
4. 图像先降到 `128×128`，`num_envs` 提到 256+，`num_mini_batches=8`，对图像 PPO 更友好。
5. 打开 `obs_normalization=True`（至少 state 归一化），或把 RGB 做 `(x-0.5)/0.5`。

下面三个脚本就是用来验证/回归这些猜测的。

---

## 脚本用法

### 1. `plot_tfevents.py` — 画训练曲线（不需要 Isaac）

从 `logs/rsl_rl/target_aiming_direct/<run>/events.*` 解析出来，plot reward / loss / entropy / KL 等。

```bash
# 默认画最新一次 run
python scripts/diagnose/plot_tfevents.py

# 画指定 run
python scripts/diagnose/plot_tfevents.py --run 2026-04-16_00-06-56

# 多次 run 叠加对比
python scripts/diagnose/plot_tfevents.py --compare 2026-04-15_22-56-56 2026-04-16_00-06-56

# 列出所有 run
python scripts/diagnose/plot_tfevents.py --list
```

输出到 `logs/rsl_rl/target_aiming_direct/<run>/training_curves.png`。

关注 6 条线：
- `Train/mean_reward`：应该稳步上升。现在多半在零附近震荡。
- `Train/mean_episode_length`：应逐渐接近最大值 (`10s / 0.0333 ≈ 300`)。若一直在很小的值 → 问题 2 已经 confirm。
- `Loss/value_function`：高噪声不下降 → 问题 4 已 confirm。
- `Policy/mean_kl`：应在 0.01 附近。若反复触发 learning-rate 自适应调到很小 → 动力太糟。
- `Loss/entropy`：应缓慢下降。若一直在 ~log(5)=1.6 不降 → policy 没学到东西。
- `Episode_Reward/*`：各 reward 项平均值，看哪个占主导。

### 2. `rollout_diagnose.py` — 随机/零/checkpoint 策略跑 env 看 reward 分量 & 动作分布

```bash
# random policy（用来校准环境本身）
./isaaclab.sh -p scripts/diagnose/rollout_diagnose.py \
    --task Isaac-Target-Aiming-Direct-v0 --policy random --steps 400 --num_envs 32

# zero policy (一直 stay)
./isaaclab.sh -p scripts/diagnose/rollout_diagnose.py \
    --task Isaac-Target-Aiming-Direct-v0 --policy zero --steps 400 --num_envs 32

# 加载训练好的 checkpoint
./isaaclab.sh -p scripts/diagnose/rollout_diagnose.py \
    --task Isaac-Target-Aiming-Direct-v0 --policy checkpoint \
    --checkpoint logs/rsl_rl/target_aiming_direct/2026-04-16_00-06-56/model_750.pt \
    --steps 400 --num_envs 32
```

输出到 `outputs/diagnose/<ts>_<policy>/`：
- `diagnose.png` — 6 联图：per-step reward、pixel error、visibility/done/idle、joint angles、action histogram、episode length 分布。
- `reward_breakdown.png` — 5 个 reward 分量叠加，一眼看出谁主导。
- `metrics.npz` — 原始数据。

**期待观察**：
- random policy 下 `reward_breakdown`，若 `idle` 接近 0、`alive` 正、`pixel` 在 0 附近 → 环境基本正常。
- checkpoint policy 的 `pixel error` 曲线若**与 random 区别不大**，说明 policy 根本没学到；结合 action histogram 看是不是卡在某个动作。
- 如果 `mean episode length` 远小于 300，confirm 问题 2。

### 3. `save_obs_images.py` — 把 RGB 观察 + 投影的目标像素位置画出来

```bash
./isaaclab.sh -p scripts/diagnose/save_obs_images.py \
    --task Isaac-Target-Aiming-Direct-v0 --num_envs 16 --steps 40
```

输出到 `outputs/diagnose/<ts>_obs/`：
- `obs_reset.png` — 16 个 env reset 后的 RGB，绿圈 = `_project_target_to_image` 计算出的目标像素位置（可见），红圈 = 认为不可见但位置仍画出来。
- `obs_grid.png` — 若干步随机动作后的观察。
- `env0_seq.png` — env 0 的时间序列。

如果绿圈**没有落在 car 的图像上**，说明 `_project_target_to_image` 里相机外参/内参有偏差，reward 基于错误的"真值"算 — 无论 PPO 多强也学不到真正的瞄准。这是要优先排除的 bug。
