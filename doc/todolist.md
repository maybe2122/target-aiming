很好，这个阶段 1 已经是一个**完整视觉控制 pipeline**了。我帮你拆成可以直接实现的工程 TODO（按实现顺序）。

你的原始目标：

```text
生成robot 和 primitive
在robot上附着相机
能够保存图像
并计算prim距离相机图像中心的误差
```

我把它细化成 **7 个步骤** 👇

---

# 阶段1：视觉误差计算系统（细化 TODO）

---

# Step 1 — 创建场景结构

目标：

```text
每个 env:
    gimbal(robot)
    cube(target)
    camera(挂在 gimbal 上)
```

需要做：

### 1.1 定义 robot (articulation)

```text
ArticulationCfg
    gimbal.usd
    yaw joint
    pitch joint
```

TODO

* [ ] 加入 ArticulationCfg
* [ ] 设置 prim_path = /World/envs/env_.*/Robot
* [ ] 设置 init pose
* [ ] 设置 actuator

---

### 1.2 定义 primitive target

```text
RigidObjectCfg
    cube
```

TODO

* [ ] prim_path = /World/envs/env_.*/Target
* [ ] cube size
* [ ] 初始位置
* [ ] 随机化位置（可选）

---

# Step 2 — 在 robot 上附着 camera

目标：

```text
camera 跟随 gimbal 运动
```

---

### 2.1 在 USD 中定义 camera mount

你有两种方式：

方式 A（推荐）

```text
Robot
 └── camera_link
      └── Camera
```

方式 B（代码 attach）

```python
prim_path="/World/envs/env_.*/Robot/camera"
```

---

TODO

* [ ] 创建 CameraCfg
* [ ] prim_path 指向 robot 子节点
* [ ] 设置分辨率
* [ ] 设置 RGB 输出

---

示例

```python
CameraCfg(
    prim_path="/World/envs/env_.*/Robot/Camera",
    height=256,
    width=256,
    data_types=["rgb"],
)
```

---

# Step 3 — 相机获取图像

目标：

```text
camera → tensor image
```

TODO

* [ ] scene 中加入 camera
* [ ] run loop 中 update camera
* [ ] 获取 camera.data.output["rgb"]

---

代码逻辑

```python
camera.update(dt)

rgb = camera.data.output["rgb"]
```

shape

```text
(num_envs, H, W, 3)
```

---

# Step 4 — 保存图像

目标：

```text
保存 debug 图像
```

TODO

* [ ] 创建 writer
* [ ] convert tensor → numpy
* [ ] 保存 png

最简单版本：

```python
image = rgb[0].cpu().numpy()
```

然后

```python
imageio.imwrite(...)
```

---

# Step 5 — 获取 target 在图像中的位置

这是最关键一步

你有三种方法：

---

## 方法 A（推荐）：semantic segmentation

给 cube 加标签：

```python
semantic_tags=[("class","target")]
```

相机输出：

```python
semantic_segmentation
```

TODO

* [ ] 开启 semantic_segmentation
* [ ] 找到 target 像素
* [ ] 计算中心

---

## 方法 B：instance segmentation

更稳

```text
instance_segmentation_fast
```

---

## 方法 C：直接用 3D 投影（最稳定）

```text
3D target pos
→ 投影到 image plane
```

推荐用于 RL

---

# Step 6 — 计算图像中心

图像中心：

```python
cx = width / 2
cy = height / 2
```

例如：

```text
256x256

center = (128,128)
```

---

# Step 7 — 计算误差

目标中心：

```text
target_u , target_v
```

误差：

```python
error_x = target_u - cx
error_y = target_v - cy
```

归一化：

```python
error_x /= cx
error_y /= cy
```

最终：

```text
[-1 , 1]
```

---

# 阶段1最终输出

你最终会得到：

```python
obs = [
    error_x,
    error_y
]
```

或者

```python
obs = rgb
```

---

# 阶段1完成后你就可以

```text
reward = - error^2
```

然后直接做 RL

---

# 阶段1完整流程图

```
gimbal joints
      ↓
camera pose
      ↓
RGB image
      ↓
find target pixel
      ↓
compute center error
      ↓
(error_x , error_y)
```

---

# 我建议你阶段1再拆更细

### Phase 1.1

生成：

```
robot
cube
camera
```

---

### Phase 1.2

保存图像

---

### Phase 1.3

获取 segmentation

---

### Phase 1.4

计算目标中心

---

### Phase 1.5

计算误差

---

如果你告诉我：

* 用 RGB 训练
  还是
* 用 error 训练

我可以帮你设计 observation space。
