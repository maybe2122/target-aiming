# EyeVLA Simulation Inference Bundle

更新时间：2026-04-13

这是一份面向仿真/RL 同事的单独交付目录，用来接入当前 EyeVLA 推理服务。

目录结构：

- `server/serve.py`
  交付入口；内部调用仓库主服务实现
- `server/run_python_vllm_server.sh`
  以 Python vLLM 启动 OpenAI-compatible 推理服务
- `server/run_hf_server.sh`
  以 HF merged 模型启动服务
- `server/run_openai_bridge.sh`
  以 OpenAI-compatible 后端桥接启动服务
- `server/gate_check.sh`
  对 `/predict` 做基本门禁检查
- `client/vla_client.py`
  Python 客户端
- `client/example_usage.py`
  最小调用示例

---

## 1. 交付边界

本包负责：

- 服务启动入口
- `/predict` 客户端调用
- 门禁与最小联调方式

仿真侧负责：

- episode 调度
- YOLO / GT / 事件窗口统计
- 在线 RL 测试
- reward 与终止条件

---

## 2. 启动服务

在仓库根目录执行：

### 推荐：Python vLLM + bridge

先启动 vLLM：

```bash
bash deliverables/sim_inference_bundle/server/run_python_vllm_server.sh \
  outputs/eyevla_v8_r2_merged_fp16 \
  8097
```

再启动统一 `/predict` bridge：

```bash
bash deliverables/sim_inference_bundle/server/run_openai_bridge.sh \
  http://127.0.0.1:8097/v1 \
  8000
```

### 备选：HF merged 后端

```bash
bash deliverables/sim_inference_bundle/server/run_hf_server.sh \
  outputs/eyevla_v8_merged_fp16_r2 \
  8000
```

### OpenAI-compatible 桥接后端

```bash
bash deliverables/sim_inference_bundle/server/run_openai_bridge.sh \
  http://127.0.0.1:8097/v1 \
  8000
```

健康检查：

```bash
curl http://127.0.0.1:8097/v1/models
curl http://127.0.0.1:8000/health
```

跨机器访问时，当前部署机对外地址是：

```bash
curl http://192.168.3.150:8000/health
```

---

## 3. 客户端最小示例

```python
from deliverables.sim_inference_bundle.client.vla_client import VLAClient

client = VLAClient("http://192.168.3.150:8000")

result = client.predict_path(
    "/path/to/frame.jpg",
    instruction="Track the vehicle in the scene.",
    predict_type="grounding_action",
)

print(result["action"])
```

如果仿真侧直接拿到 OpenCV 图像：

```python
result = client.predict_cv2(
    cv_image,
    instruction="Track the vehicle in the scene.",
    predict_type="grounding_action",
)
```

---

## 4. 接口说明

### `POST /predict`

请求：

```json
{
  "image": "<base64 encoded JPEG>",
  "instruction": "Track the vehicle in the scene.",
  "type": "grounding_action"
}
```

响应核心字段：

- `action.delta_pan`
- `action.delta_tilt`
- `action.delta_zoom`
- `latency_ms`

调试字段：

- `raw_text`
- `action_debug`

### `POST /debug/grounding`

仅用于 grounding 调试，不建议作为 RL 主调用口。

---

## 5. 门禁检查

```bash
bash deliverables/sim_inference_bundle/server/gate_check.sh \
  http://192.168.3.150:8000/predict \
  /path/to/sample.jpg \
  grounding_action
```

---

## 6. 备注

这份 bundle 是“仓库内单独交付目录”，默认依赖当前仓库根目录下的主实现：

- [scripts/serve.py](/home/sz/Project/agent_app/unsloth_fn/scripts/serve.py)
- [scripts/gate_action_tokens.py](/home/sz/Project/agent_app/unsloth_fn/scripts/gate_action_tokens.py)

更完整的部署说明见：

- [docs/inference-deployment-for-sim.md](/home/sz/Project/agent_app/unsloth_fn/docs/inference-deployment-for-sim.md)

当前在 `192.168.3.150` 的实际部署口径也是：

- Python vLLM: `127.0.0.1:8097`
- EyeVLA bridge: `0.0.0.0:8000`
- 跨机访问入口: `http://192.168.3.150:8000`
