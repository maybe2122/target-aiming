# Target Aiming — Projection Geometry

## 1. Gimbal 机械结构 (URDF)

```
base (地面)
 └─ yaw_joint  (绕 Z 轴旋转, ±180°)
     └─ yaw_link
         └─ pitch_joint  (绕 Y 轴旋转, ±90°)
             └─ pitch_link
                 └─ camera (TiledCamera, 固定偏移)
```

### 关节轴与方向

| 关节 | 旋转轴 | 正方向 (右手定则) |
|------|--------|-------------------|
| yaw_joint | Z (竖直) | 俯视逆时针 (X→Y) |
| pitch_joint | Y (水平) | 相机向**下**看 (X→-Z) |

> 关键: **pitch_joint > 0 = 相机向下**, pitch_joint < 0 = 相机向上

### 各连杆在 pitch=0, yaw=0 时的世界位置

```
base           → (0, 0, 0)
yaw_link       → (0, 0, 0.075)      # yaw_joint origin offset
pitch_link     → (0, 0, 0.175)      # + pitch_joint origin offset (0,0,0.1)
camera         → (0.1, 0, 2.175)    # + camera offset (0.1, 0, 2.0)
```

## 2. 相机参数

### 内参 (Pinhole Model)

| 参数 | 值 | 说明 |
|------|------|------|
| focal_length | 18.0 mm | 焦距 |
| horizontal_aperture | 20.955 mm | 传感器水平尺寸 |
| image_width | 128 px | 图像宽度 |
| image_height | 128 px | 图像高度 |

### 推导量

```
focal_px = image_width × focal_length / horizontal_aperture
         = 128 × 18.0 / 20.955
         ≈ 110 px

img_cx = image_width  / 2 = 64 px    (图像中心 u)
img_cy = image_height / 2 = 64 px    (图像中心 v)
img_diag = sqrt(128² + 128²) ≈ 181 px

half_fov = atan(horizontal_aperture / (2 × focal_length))
         = atan(20.955 / 36)
         ≈ 0.527 rad ≈ 30.2°
```

### 外参 (相对于 pitch_link)

```
pos = (0.1, 0.0, 2.0)            # pitch_link 坐标系下: X前 0.1m, Z上 2.0m
rot = (0.5, -0.5, 0.5, -0.5)     # 四元数 (w, x, y, z), ROS convention
```

四元数的旋转矩阵:

```
R = [[ 0,  0,  1],
     [-1,  0,  0],
     [ 0, -1,  0]]
```

ROS 相机坐标系 +Z 方向经此旋转后指向 pitch_link 的 +X 方向,
即 **yaw=0, pitch=0 时相机朝世界 +X 方向看** (水平向前).

## 3. 坐标变换链 (精确投影)

### 3.1 相机世界坐标

相机在 pitch_link 上偏移 `(0.1, 0, 2.0)`, 经 yaw/pitch 关节旋转后:

```python
cos_y, sin_y = cos(cur_yaw), sin(cur_yaw)
cos_p, sin_p = cos(cur_pitch), sin(cur_pitch)

# 相机在 yaw 旋转平面内的前向偏移
cam_offset_x = 0.1 × cos_p + 2.0 × sin_p

# 相机高度 (含关节偏移 0.175m)
cam_offset_z = -0.1 × sin_p + 2.0 × cos_p + 0.175

# 相机世界坐标
cam_x = gx + cos_y × cam_offset_x
cam_y = gy + sin_y × cam_offset_x
cam_z = gz + cam_offset_z
```

### 3.2 相机基向量

由 URDF 关节轴 + 相机四元数推导:

```python
forward = (cos_y × cos_p,  sin_y × cos_p,  -sin_p)   # 相机光轴
right   = (sin_y,          -cos_y,           0)        # 图像 +u 方向
up      = (cos_y × sin_p,  sin_y × sin_p,   cos_p)    # 图像 -v 方向
```

### 3.3 针孔投影

```python
# 目标相对于相机的方向向量
dx = tx - cam_x
dy = ty - cam_y
dz = tz - cam_z

# 投影到相机坐标系
d_forward = dx × forward_x + dy × forward_y + dz × forward_z
d_right   = dx × right_x   + dy × right_y
d_up      = dx × up_x      + dy × up_y      + dz × up_z

# 针孔模型 → 像素坐标
u = img_cx + focal_px × d_right   / d_forward
v = img_cy - focal_px × d_up      / d_forward
```

含义:
- `u = img_cx` → 目标在图像水平中心
- `v = img_cy` → 目标在图像垂直中心
- `u > img_cx` → 目标在图像右侧 (target_yaw > cur_yaw)
- `v > img_cy` → 目标在图像下方 (target_pitch < cur_pitch, 即目标在相机光轴下方)

### 3.3 像素误差归一化

```python
pixel_dist = sqrt((u - img_cx)² + (v - img_cy)²)
pixel_error_normalized = pixel_dist / img_diag    # 范围 [0, ~0.7]
```

当目标不可见时, `pixel_error_normalized = 1.0`.

### 3.4 可见性判断

```python
visible = (0 ≤ u < image_width) and (0 ≤ v < image_height)
```

## 4. Reset 时的角度计算

### 4.1 Yaw 计算 (精确)

```python
aim_yaw = atan2(dy, dx)
```

水平面内的指向, 不受相机高度影响, 精确.

### 4.2 Pitch 计算 (需要补偿)

相机在 pitch_link 上方 2m (摇臂结构). 直接用 `atan2(dz, dist_xy)` 不行,
因为:
1. dz 从 gimbal root 算 ≈ 0.05 (目标几乎同高), 给出 pitch ≈ 0° (水平看)
2. 但相机实际在 2m 高处水平看, 地面目标在 FOV 之外 (低于 30° 半 FOV)

**也不能直接减 camera_height**:
- `atan2(dz - 2.0, dist)` ≈ -0.66 rad, pitch 为负数
- pitch < 0 = 相机朝上看 (更糟!)
- 即使翻转符号, 大角度 pitch 会让 2m 摇臂大幅摆动, 相机位置偏移严重

**正确做法: 正向补偿 + 缩放**

```python
camera_height = 2.0  # 相机在 pitch_link 上方的偏移

# 从相机高度到地面的俯角
aim_pitch = atan2(camera_height, dist_xy) × 0.7
```

- `atan2(camera_height, dist_xy)` → 正值 → 相机向下看 ✓
- `× 0.7` → 补偿摇臂旋转时相机位置前移效应
  - pitch 旋转使相机沿弧线移动: 前移 + 降低
  - 实际需要的 pitch 角小于几何计算角

#### 数值验证 (dist_xy = 2.5m)

| 方法 | aim_pitch | 相机实际行为 | 目标是否可见 |
|------|-----------|------------|------------|
| 不补偿 `atan2(0.05, 2.5)` | +0.02 rad | 几乎水平看 | 不可见 (目标在 FOV 下方 41°) |
| 减 camera_height `atan2(-1.95, 2.5)` | -0.66 rad | 朝上看 | 不可见 (方向完全反) |
| 正向补偿 `atan2(2.0, 2.5) × 0.7` | +0.50 rad | 向下看 29° | **可见** (目标在画面中下部) |

## 5. 摇臂效应详解

当 pitch_joint = θ 时, 相机位置从 pitch_link 原点经旋转矩阵 Ry(θ) 变换:

```
cam_pos_in_yaw_frame:
  x' =  0.1 × cos(θ) + 2.0 × sin(θ)
  y' =  0
  z' = -0.1 × sin(θ) + 2.0 × cos(θ) + 0.175
```

相机前方向:

```
forward = (cos(θ), 0, -sin(θ))
```

| pitch (rad) | pitch (deg) | 相机 X 偏移 | 相机 Z 高度 | 前方向 Z 分量 |
|-------------|-------------|------------|------------|--------------|
| 0.0 | 0° | +0.10 | 2.18 | 0.000 (水平) |
| +0.3 | 17° | +0.69 | 2.06 | -0.296 (向下) |
| +0.5 | 29° | +1.05 | 1.88 | -0.479 (向下) |
| +0.7 | 40° | +1.40 | 1.70 | -0.644 (向下) |
| -0.3 | -17° | -0.50 | 2.09 | +0.296 (向上) |

> 关键观察: 正 pitch 使相机**前移 + 下降**, 自然缩短了到目标的距离,
> 因此实际需要的 pitch 角度小于从固定点计算的几何角. 经验缩放因子 0.7 可补偿此效应.

## 6. 公式适用条件与局限

### 适用

- 目标水平距离 >> 相机偏移量 (dist_xy >> 2m)
- 小角度偏差 (target_angle - cur_angle 较小)
- 用于 reward 的近似像素误差信号

### 局限

- 近距离目标 (< 3m) 时, 摇臂视差导致投影与真实像素不精确匹配
- 大 pitch 角时相机位置偏移大, 线性近似失效
- 不适用于精确的像素级目标检测, 仅用于 reward shaping
