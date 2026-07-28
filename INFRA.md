# DiT4DiT OpenOven 推理与 SIMPLE 评测

本文记录如何使用现有 OpenOven 权重启动 DiT4DiT 推理服务，并在 `/home/d013/桌面/SIMPLE` 中评测 `G1WholebodyOpenOvenTeleop-v0`。

---

## 1. 路径约定

| 项 | 路径 |
|----|------|
| DiT4DiT 仓库 | `/home/d013/桌面/DIT4DIT` |
| SIMPLE 仓库 | `/home/d013/桌面/SIMPLE` |
| OpenOven 权重目录 | `/home/d013/桌面/CKPT/DIT4DIT/OPENOVEN` |
| checkpoint | `/home/d013/桌面/CKPT/DIT4DIT/OPENOVEN/checkpoints/steps_40000_pytorch_model.pt` |
| checkpoint config | `/home/d013/桌面/CKPT/DIT4DIT/OPENOVEN/config.yaml` |
| 归一化统计 | `/home/d013/桌面/CKPT/DIT4DIT/OPENOVEN/dataset_statistics.json` |
| Cosmos backbone | `/home/d013/桌面/DIT4DIT/Cosmos-Predict2.5-2B` |
| SIMPLE server 脚本 | `deployment/model_server/server_policy_simple_g1.py` |

---

## 2. 模型与接口

OpenOven 权重对应任务：

```text
G1WholebodyOpenOvenTeleop-v0
```

核心维度：

| 项 | 值 |
|----|----|
| state | `32D` |
| action | `36D` |
| action chunk | `16` |
| 输入图像 | SIMPLE 原始 `360x640`，server 内部 resize 到 `224x224` |
| prompt | `move forward to the oven and open it` |

SIMPLE 侧走已有的 websocket + decoupled WBC 评测路径：

```text
policy = vlajepa_decoupled_wbc
server = ws://127.0.0.1:10090
```

虽然 policy 名仍叫 `vlajepa_decoupled_wbc`，但 server 端实际是 DiT4DiT。该路径复用 SIMPLE 已有的 32D state / 36D action 布局和 WBC 控制逻辑。

---

## 3. 推荐启动方式：24G 显存

24G 显卡不要直接全量加载 text encoder。推荐启动时先用 text encoder 编码固定 prompt，然后缓存 `prompt_embeds`，随后释放 text encoder。

启动 DiT4DiT 推理端：

```bash
cd /home/d013/桌面/DIT4DIT
conda activate dit4dit

python deployment/model_server/server_policy_simple_g1.py \
  --ckpt_path /home/d013/桌面/CKPT/DIT4DIT/OPENOVEN/checkpoints/steps_40000_pytorch_model.pt \
  --base_model /home/d013/桌面/DIT4DIT/Cosmos-Predict2.5-2B \
  --port 10090 \
  --cuda 0 \
  --cpu_offload_text_encoder \
  --cache_prompt_embeds \
  --cached_prompt "move forward to the oven and open it" \
  --use_bf16
```

启动成功标志：

```text
INFO:root:server running ...
INFO:websockets.server:server listening on 0.0.0.0:10090
```

### 3.1 显存 / 内存策略

使用 `--cpu_offload_text_encoder --cache_prompt_embeds` 后：

| 阶段 | CPU | GPU |
|------|-----|-----|
| 启动时 | text encoder 短暂用于 prompt 编码 | transformer / VAE / action model |
| 编码后 | text encoder 被释放 | cached prompt embeds + transformer / VAE / action model |
| 每次推理 | 不再跑 text encoder | 视频特征提取 + action diffusion |

参数量参考：

| 组件 | 参数量 |
|------|--------|
| Cosmos text_encoder | 约 `8.29B` |
| Cosmos transformer | 约 `2.06B` |
| Cosmos VAE | 约 `0.13B` |
| DiT4DiT action_model | 约 `0.16B` |
| checkpoint 总参数 | 约 `10.64B` |
| prompt cache 后每步主要参与推理 | 约 `2.35B` |

### 3.2 如果 bf16 报错

如果显卡或环境不支持 bf16，去掉：

```bash
--use_bf16
```

但 24G 显存下优先尝试带 `--use_bf16`。

---

## 4. SIMPLE 评测

另开一个终端，先跑 1 个 episode 冒烟测试：

```bash
cd /home/d013/桌面/SIMPLE

TASK=simple/G1WholebodyOpenOvenTeleop-v0 \
NUM_EPISODES=1 \
MAX_EPISODE_STEPS=1000 \
bash scripts/run_eval_vlajepa_clean.sh level-0 10090
```

冒烟测试通过后，跑三档完整评测：

```bash
cd /home/d013/桌面/SIMPLE

MAX_EPISODE_STEPS=1000 \
bash scripts/run_eval_vlajepa_all_levels.sh \
  simple/G1WholebodyOpenOvenTeleop-v0 \
  10090 \
  10
```

输出位置：

```text
/home/d013/桌面/SIMPLE/data/evals_decoupled_wbc/level-0/eval_stats.txt
/home/d013/桌面/SIMPLE/data/evals_decoupled_wbc/level-1/eval_stats.txt
/home/d013/桌面/SIMPLE/data/evals_decoupled_wbc/level-2/eval_stats.txt
```

---

## 5. 关键实现细节

### 5.1 图像 resize

SIMPLE 发来的 head camera 图像是 `360x640`。OpenOven 训练配置要求：

```yaml
datasets:
  vla_data:
    image_size: [224, 224]
```

因此 `server_policy_simple_g1.py` 会在推理前把输入图像 resize 到 `224x224`，再送入 Cosmos。

如果不 resize，会报：

```text
ValueError: `height` and `width` must be divisible by 16 but are 360 and 640.
```

### 5.2 state 归一化

SIMPLE 发来的 `32D state` 是原始值，server 会使用 checkpoint 的 `dataset_statistics.json` 做 `min_max` 归一化后再送入模型。

OpenOven state 布局：

```text
left_hand(7) + right_hand(7) + left_arm(7) + right_arm(7) + rpy(3) + height(1)
```

### 5.3 action 反归一化

模型输出是 normalized action，server 会转回 SIMPLE/WBC 需要的真实 action。

OpenOven action 布局：

```text
left_hand(7) + right_hand(7) + left_arm(7) + right_arm(7)
+ rpy(3) + height(1)
+ torso_vx(1) + torso_vy(1) + torso_vyaw(1) + target_yaw(1)
```

反归一化规则：

| 维度 | 字段 | 方式 |
|------|------|------|
| `0:32` | hand / arm / rpy / height | `min_max` |
| `32:36` | torso/nav | `mean_std` |

---

## 6. 常见问题

### 6.1 启动时 `已杀死`

通常是 CPU 内存峰值过高。server 已使用：

```python
torch.load(..., weights_only=True, mmap=True)
load_state_dict(..., assign=True)
```

用于降低加载 20G checkpoint 时的 CPU 内存峰值。若仍被杀，优先检查：

```bash
free -h
```

可考虑增加 swap 或关闭其它占内存进程。

### 6.2 safety_checker warning

看到下面 warning 可以忽略：

```text
Expected types for safety_checker ...
```

当前使用 dummy safety checker，目的是避免额外 guardrail 依赖，不影响 OpenOven 推理。

### 6.3 评测端连接不上

确认推理端已经监听：

```text
server listening on 0.0.0.0:10090
```

SIMPLE 侧端口必须一致：

```bash
bash scripts/run_eval_vlajepa_clean.sh level-0 10090
```

### 6.4 prompt cache 的限制

使用：

```bash
--cache_prompt_embeds
```

后，server 会始终使用 `--cached_prompt` 指定的 prompt。SIMPLE 每个 episode 传来的 instruction 不再参与 text encoder 编码。

这适用于 OpenOven 这类固定 prompt 评测；如果要评测多个不同 prompt 的任务，应关闭 prompt cache，或为不同 prompt 分别启动 server。

