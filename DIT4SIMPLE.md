# 使用 `datasets/simple` 微调 DiT4DiT

本文说明如何用仓库内 `datasets/simple/` 下的 G1 Wholebody LeRobot 数据微调 DiT4DiT。  
**已落地示例**：

- `G1WholebodyOpenOvenTeleop-v0`（开烤箱）→ `G1_OPEN_OVEN`
- `G1WholebodyOpenFaucetTeleop-v0`（开水龙头）→ `G1_OPEN_FAUCET`（与 Oven 同 modality / DataConfig；换任务时务必先补 `video_info`，见 [§4.5.1](#451-补全空的-video_info换任务几乎必做)）

---

## 目录

1. [数据概览](#1-数据概览)
2. [State / Action / 图像维度](#2-state--action--图像维度)
3. [归一化方式](#3-归一化方式)
4. [需要修改哪些文件](#4-需要修改哪些文件)
5. [超参文件重点参数](#5-超参文件重点参数)
6. [训练 vs 冻结](#6-训练-vs-冻结)
7. [如何开始训练](#7-如何开始训练)
8. [监控与产出](#8-监控与产出)
9. [常见问题](#9-常见问题)

---

## 1. 数据概览

### 1.1 根目录

```
datasets/simple/
├── G1WholebodyOpenOvenTeleop-v0/     # 开烤箱（本指南主示例）
├── G1WholebodyOpenTrashCanTeleop-v0/
├── G1WholebodyXMovePickTeleop-v0/
└── ...                               # 同系列其它任务
```

每个任务是一个 **LeRobot v2.1** 数据集，典型结构：

```
G1WholebodyOpenOvenTeleop-v0/
├── meta/
│   ├── info.json           # fps、特征 shape、路径模板（video_info 不能为空）
│   ├── modality.json       # state/action/video 切片映射（最重要）
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl
│   ├── stats.json / stats_gr00t.json / ...
│   └── ...
├── data/chunk-000/episode_XXXXXX.parquet
└── videos/chunk-000/egocentric/episode_XXXXXX.mp4
```

### 1.2 OpenOven 规模（参考）

| 项 | 值 |
|----|-----|
| episodes | 99 |
| frames | 34713 |
| fps | 50 |
| 原始相机 | `observation.images.egocentric`，分辨率 **360×640×3** |
| robot_type | `g1` |

> 注意：`meta/info.json` 里对应视频特征的 `video_info` 必须填全（含 `video.fps` 等），否则 dataloader 会 `KeyError: video.fps` / `KeyError: info`。  
> **补全步骤见 [§4.5.1](#451-补全空的-video_info换任务几乎必做)**（OpenFaucet 等多数任务原始为空，仿写配置前先做这一步）。

---

## 2. State / Action / 图像维度

维度由 `meta/modality.json` + `G1WholebodyTeleopDataConfig` 共同决定，并在 yaml 的 `action_dim` / `state_dim` 中声明。

### 2.1 State：32 维

| 字段 | 切片 | 维数 |
|------|------|------|
| `state.left_hand` | `[0:7]` | 7 |
| `state.right_hand` | `[7:14]` | 7 |
| `state.left_arm` | `[14:21]` | 7 |
| `state.right_arm` | `[21:28]` | 7 |
| `state.rpy` | `[28:31]` | 3 |
| `state.height` | `[31:32]` | 1 |
| **合计** | | **32** |

对应 yaml：`framework.action_model.state_dim: 32`、`datasets.vla_data.max_state_dim: 32`。

### 2.2 Action：36 维（chunk = 16）

| 字段 | 切片 | 维数 | 备注 |
|------|------|------|------|
| `action.left_hand` | `[0:7]` | 7 | absolute |
| `action.right_hand` | `[7:14]` | 7 | absolute |
| `action.left_arm` | `[14:21]` | 7 | absolute |
| `action.right_arm` | `[21:28]` | 7 | absolute |
| `action.rpy` | `[28:31]` | 3 | absolute |
| `action.height` | `[31:32]` | 1 | absolute |
| `action.torso_vx` | `[32:33]` | 1 | 速度类 |
| `action.torso_vy` | `[33:34]` | 1 | 速度类 |
| `action.torso_vyaw` | `[34:35]` | 1 | 速度类 |
| `action.target_yaw` | `[35:36]` | 1 | 航向 |
| **合计** | | **36** |

- 单步 action：**36D**
- 时间长度：**16**（`action_horizon: 16`，`future_action_window_size: 15`，`action_indices = range(16)`）
- 因此网络侧 action 张量形状约为 **`[B, 16, 36]`**

对应 yaml：`framework.action_model.action_dim: 36`、`datasets.vla_data.max_action_dim: 36`。

### 2.3 图像 / 视频

| 阶段 | 形状 / 说明 |
|------|-------------|
| 原始视频 | `360 × 640 × 3`，单视角 egocentric |
| modality 映射 | `video.rs_view` ← `observation.images.egocentric` |
| 送入模型前 resize | **`224 × 224`**（`image_size: [224, 224]`，`default_image_resolution: [3, 224, 224]`） |
| 时间采样 | `video_delta_indices: [0..16]` → 先取 **17** 帧 |
| 下采样 | `action_video_freq_ratio`（当前为 **1**） |

**当前配置下一次性进网络的帧数：**

```text
len(video_delta_indices) / action_video_freq_ratio
= 17 / 1 = 17 帧
```

若 `action_video_freq_ratio=2`，则为 `range(0,17,2)` → **9 帧**。

> 关系记忆：action chunk=16；视频 indices 取到 16（含 0）共 17 点；ratio=1 时 **17 帧图像进 Cosmos**，action 仍是 16 步。

---

## 3. 归一化方式

归一化写在 DataConfig 的 `StateActionTransform` 里，**不在 yaml 里逐字段声明**。  
OpenOven 使用的是 `g1_wholebody_teleop` → `G1WholebodyTeleopDataConfig`：

| 模态 | 字段 | 方式 |
|------|------|------|
| state 全部 | hand / arm / rpy / height | **`min_max`** |
| action 关节类 | hand / arm / rpy / height | **`min_max`** |
| action 速度/航向 | `torso_vx`, `torso_vy`, `torso_vyaw`, `target_yaw` | **`mean_std`** |

- **不使用** `StateActionSinCosTransform`（不用 SinCos 扩维）
- 统计量来自数据集 `meta/` 下的 stats；训练时会落到  
  `results/Checkpoints_G1_OPEN_OVEN/dit4dit_G1_OPEN_OVEN/dataset_statistics.json`

代码位置：

```text
DiT4DiT/dataloader/gr00t_lerobot/data_config.py
  class G1WholebodyTeleopDataConfig
```

---

## 4. 需要修改哪些文件

用 `datasets/simple` 里某个新任务微调时，通常改下面 **4 处**（OpenOven 已改好，可作模板）。

### 4.1 `data_config.py` — 注册机器人模态与归一化

**路径**：`DiT4DiT/dataloader/gr00t_lerobot/data_config.py`

需要：

1. 定义 `G1WholebodyTeleopDataConfig`（或你的新 Config）  
   - `video_keys` / `state_keys` / `action_keys`  
   - `action_indices = list(range(16))`（chunk 长度）  
   - `transform()` 里的 `normalization_modes`
2. 在 `ROBOT_TYPE_CONFIG_MAP` 注册，例如：

```python
"g1_wholebody_teleop": G1WholebodyTeleopDataConfig(),
```

### 4.2 `mixtures.py` — 把数据集挂到 data_mix 名

**路径**：`DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`

```python
"G1_OPEN_OVEN": [
    ("G1WholebodyOpenOvenTeleop-v0", 1.0, "g1_wholebody_teleop"),
],
```

含义：`(数据集文件夹名, 采样权重, robot_type 名)`。  
文件夹名必须等于 `datasets/simple/` 下的目录名。

换任务示例：

```python
"G1_OPEN_TRASH": [
    ("G1WholebodyOpenTrashCanTeleop-v0", 1.0, "g1_wholebody_teleop"),
],
```

若 `modality.json` 布局与 OpenOven 一致，可复用同一个 `g1_wholebody_teleop` Config。

### 4.3 训练 yaml — 超参与数据路径

**路径**：`DiT4DiT/config/G1_OPEN_OVEN/dit4dit_G1_OPEN_OVEN.yaml`

重点对齐：

- `action_dim: 36` / `state_dim: 32`
- `data_root_dir: ./datasets/simple`
- `data_mix: G1_OPEN_OVEN`
- `video_delta_indices` / `action_video_freq_ratio`
- `freeze_modules` / `max_train_steps` / `save_interval` / 学习率

### 4.4 启动脚本

**路径**：`examples/G1_OPEN_OVEN/train_files/run_G1_OPEN_OVEN.sh`

用 CLI 覆盖 yaml 中的 BS、步数、冻结列表、`action_video_freq_ratio` 等。  
监控脚本（可选）：`examples/G1_OPEN_OVEN/train_files/watch_train.sh`。

### 4.5 数据侧（必要时）

| 文件 | 何时改 |
|------|--------|
| `meta/modality.json` | 字段切片、视频 key 与 DataConfig 不一致时 |
| `meta/info.json` | **绝大多数 `simple` 任务必做**：补全空的 `video_info`；确认 `video_path` / fps |
| parquet / mp4 | 原始数据错误时 |

一般 **不必改** 训练主代码 `DiT4DiT/training/train.py`。

#### 4.5.1 补全空的 `video_info`（换任务几乎必做）

`datasets/simple` 下很多任务的 `meta/info.json` 里，视频特征的 `video_info` 是 **空字典 `{}`**。  
Dataloader（`DiT4DiT/dataloader/gr00t_lerobot/datasets.py` → `_get_metadata`）会先读：

```text
features[<video_key>].video_info["video.fps"]
```

拿不到就回退读 `info["video.channels"]` / `info["video.fps"]`。两边都没有时，启动会直接挂掉，典型报错：

```text
KeyError: 'video.fps'
# 随后 except 分支再报：
KeyError: 'info'
```

**现象对照**（扫描结果，仅供参考；以你本地 `info.json` 为准）：

| 状态 | 示例 |
|------|------|
| 已填好 | `G1WholebodyOpenOvenTeleop-v0`、`G1WholebodyBendPickMixed-v0`、`G1WholebodyBendPickMP-v0`、`G1WholebodyTabletopGraspMP-v0`、`G1WholebodyXMoveBendPickMP-v0` |
| 空 `video_info`（需补） | `G1WholebodyOpenFaucetTeleop-v0`、`G1WholebodyOpenTrashCanTeleop-v0`、`G1WholebodyCloseDoorTeleop-v0`、以及多数 `*Teleop-v0` |

**补全步骤（以 OpenFaucet 为例）**：

1. **先备份**

```bash
cp datasets/simple/G1WholebodyOpenFaucetTeleop-v0/meta/info.json \
   datasets/simple/G1WholebodyOpenFaucetTeleop-v0/meta/info.json.bak
```

2. **从真实 mp4 读出参数**（环境里常没有 `ffprobe`，用 PyAV）：

```bash
# conda activate dit4dit 后
python - <<'EOF'
import av, glob
f = sorted(glob.glob(
    "datasets/simple/G1WholebodyOpenFaucetTeleop-v0/videos/chunk-*/egocentric/*.mp4"
))[0]
c = av.open(f); s = c.streams.video[0]
print("codec=", s.codec_context.name,
      "fps=", float(s.average_rate),
      "WxH=", s.codec_context.width, s.codec_context.height,
      "pix_fmt=", s.codec_context.pix_fmt)
c.close()
EOF
```

本系列典型结果：`h264` / `50.0` / `640×360` / `yuv420p`。也可直接对照顶层 `info.json` 的 `"fps": 50` 与 `shape: [360, 640, 3]`。

3. **把空的 `video_info` 改成**（键名必须带 `video.` 前缀）：

```json
"observation.images.egocentric": {
  "dtype": "video",
  "shape": [360, 640, 3],
  "names": ["height", "width", "channel"],
  "video_info": {
    "video.fps": 50.0,
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": false,
    "video.channels": 3
  }
}
```

4. **快速自检**（不必起满 8 卡）：

```bash
PYTHONPATH=$(pwd) python - <<'EOF'
from omegaconf import OmegaConf
from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset
cfg = OmegaConf.load(
    "DiT4DiT/config/G1_OPEN_FAUCET/dit4dit_G1_OPEN_FAUCET.yaml"
).datasets.vla_data
ds = get_vla_dataset(data_cfg=cfg)
print("OK", len(ds), ds[0]["action"].shape, ds[0]["state"].shape)
EOF
```

成功时会打印总步数（应等于 `info.json` 的 `total_frames`），并可能写出 `meta/stats_gr00t.json`、`meta/steps_data_index.pkl`（缓存，下次直接复用）。

> OpenFaucet 已按上述步骤补过；换到其它空 `video_info` 任务时，**先做这一步再起 DLC**，否则会在加载数据阶段就失败。

---

## 5. 超参文件重点参数

配置文件：`DiT4DiT/config/G1_OPEN_OVEN/dit4dit_G1_OPEN_OVEN.yaml`  
（启动脚本里的同名 CLI 会覆盖 yaml。）

### 5.1 框架 / 动作头（必对）

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `framework.cosmos25.base_model` | `./Cosmos-Predict2.5-2B` | 基座权重目录 |
| `framework.cosmos25.training` | `joint` | action + future video 联合训练 |
| `framework.cosmos25.extract_layer` | `17` | 从 Cosmos transformer 抽特征的层 |
| `framework.action_model.action_dim` | `36` | 必须与数据一致 |
| `framework.action_model.state_dim` | `32` | 必须与数据一致 |
| `framework.action_model.action_horizon` | `16` | action chunk 长度 |
| `framework.action_model.future_action_window_size` | `15` | 与 horizon 配套（当前+未来共 16） |

### 5.2 数据（显存与时间相关）

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `datasets.vla_data.data_root_dir` | `./datasets/simple` | 数据根 |
| `datasets.vla_data.data_mix` | `G1_OPEN_OVEN` | mixtures 里的名字 |
| `datasets.vla_data.per_device_batch_size` | `8`（脚本可覆盖） | **单卡 BS**；OOM 时优先降这个 |
| `datasets.vla_data.image_size` | `[224, 224]` | 输入分辨率 |
| `datasets.vla_data.video_delta_indices` | `[0..16]` | 17 个视频时间点 |
| `datasets.vla_data.action_video_freq_ratio` | `1` | `1`→17 帧；`2`→9 帧（省显存） |
| `datasets.vla_data.max_state_dim` / `max_action_dim` | `32` / `36` | padding 上限 |

### 5.3 训练调度

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `trainer.max_train_steps` | `100000` | 总步数 |
| `trainer.save_interval` | `10000` | 每 1 万步存一次 |
| `trainer.num_warmup_steps` | `2000` | warmup |
| `trainer.learning_rate.backbone_interface` | `1e-5` | Cosmos transformer 等可训部分 |
| `trainer.learning_rate.action_model` | `1e-4` | Action DiT |
| `trainer.freeze_modules` | 见下一节 | 逗号分隔的模块路径 |
| `trainer.logging_frequency` | `10` | 每 10 step 打 loss |
| `trainer.enable_gradient_checkpointing` | `true` | 省显存 |

### 5.4 启动脚本里常改的项

`examples/G1_OPEN_OVEN/train_files/run_G1_OPEN_OVEN.sh`：

| 项 | 当前 | 说明 |
|----|------|------|
| 第 1 参数 BS | 默认 `8` | `bash run_....sh 4` → 单卡 BS=4 |
| `--num_processes` | `8` | 固定 8 卡 |
| `action_video_freq_ratio` | `1` | 与 yaml 对齐 |
| `max_train_steps` / `save_interval` | `100000` / `10000` | 10 万步、每 1 万存盘 |

---

## 6. 训练 vs 冻结

当前冻结列表（yaml + 脚本一致）：

```text
backbone_interface.extractor.text_encoder
backbone_interface.extractor.vae
```

| 模块 | 状态 | 说明 |
|------|------|------|
| Cosmos **text_encoder** | **冻结** | 文本编码 |
| Cosmos **VAE** | **冻结** | 视频编解码 |
| Cosmos **transformer**（backbone） | **训练** | 在 `backbone_interface` 学习率下更新 |
| **action_model**（Action DiT） | **训练** | 动作扩散头，学习率通常更高 |

日志里典型规模（OpenOven 启动时打印过）：

```text
Total parameters:     ~10641M
Trainable parameters: ~2222M   (~20.9%)
Frozen parameters:    ~8419M
```

若要把整棵 Cosmos backbone 也冻住、只训 action 头，需把 `freeze_modules` 扩成覆盖 `backbone_interface`（或按项目里其它实验写法调整）；本 OpenOven 配置是 **joint：训 transformer + action_model，冻 text_encoder + vae**。

---

## 7. 如何开始训练

### 7.1 前置检查

1. 环境：`conda activate dit4dit`（或你的等价环境）
2. 基座模型目录存在：`./Cosmos-Predict2.5-2B`
3. 数据可读：`./datasets/simple/G1WholebodyOpenOvenTeleop-v0/`
4. `info.json` 的 `video_info` 已填全
5. GPU：推荐 **8× 大显存卡**；`ratio=1`（17 帧）时单卡 BS 建议从 **4 或更小** 试起

### 7.2 DLC / 本机启动命令

仓库根目录执行：

```bash
. /cpfs_infra/shared/xiaoxinyu/opt/miniconda3/etc/profile.d/conda.sh && \
conda activate dit4dit && \
cd /cpfs_infra/shared/xiaoxinyu/DiT4DiT && \
bash examples/G1_OPEN_OVEN/train_files/run_G1_OPEN_OVEN.sh 4
```

- 唯一 CLI 参数：**per-device batch size**（上例为 `4`）
- 默认不传则 BS=`8`（显存紧时不要用）
- 卡数写死为 **8**（`--num_processes 8`）

### 7.3 换 `simple` 里另一个任务（最短路径）

假设任务与 OpenOven 的 `modality.json` 布局相同：

1. **先检查并补全** `meta/info.json` 的 `video_info`（见 [§4.5.1](#451-补全空的-video_info换任务几乎必做)；多数任务默认是空的）
2. 在 `mixtures.py` 增加新 mix（或改 `G1_OPEN_OVEN` 指向新文件夹名）
3. 仿写 yaml / 启动脚本：改 `data_mix`、`run_id`、`run_root_dir`（可参考已落地的 `G1_OPEN_FAUCET`）
4. 确认 `action_dim`/`state_dim` 仍为 36/32
5. 重新启动

若布局不同：先改 `modality.json` 与 `G1WholebodyTeleopDataConfig`（或新建 Config），再注册到 `ROBOT_TYPE_CONFIG_MAP`。

---

## 8. 监控与产出

### 8.1 日志

- 训练日志：`log/dit4dit_G1_OPEN_OVEN_<时间戳>/train.log`
- 成功标志：出现 `Step 10, Loss: {...}` 且文件持续更新
- 一键监控（DSW / 挂同一 CPFS 的机器）：

```bash
cd /cpfs_infra/shared/xiaoxinyu/DiT4DiT
bash examples/G1_OPEN_OVEN/train_files/watch_train.sh
```

> DLC 控制台显示 Running ≠ 一定在出 Step。以 `train.log` / wandb 的 Step 为准。  
> 配置打完后约 1 分钟内应有 Step；长时间停在 `0%` 优先查 OOM / CUDA / NCCL。

### 8.2 权重

```text
results/Checkpoints_G1_OPEN_OVEN/dit4dit_G1_OPEN_OVEN/checkpoints/
  steps_10000_pytorch_model.pt
  steps_20000_pytorch_model.pt
  ...
```

另有 `dataset_statistics.json`、wandb 目录等同目录产物。

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| `KeyError: 'video.fps'`，随后 `KeyError: 'info'` | **空 `video_info`**。按 [§4.5.1](#451-补全空的-video_info换任务几乎必做) 从 mp4 读参数并写入 `meta/info.json`（与配置仿写无关） |
| `WANDB_API_KEY: Please export ...` | DLC 无该环境变量。在启动脚本里 `export WANDB_API_KEY=...`，或在 DLC 提交页填环境变量（push 前勿把明文 key 留在会进 git 的 `examples/`） |
| CUDA OOM | 降 `per_device_batch_size`；或把 `action_video_freq_ratio` 改回 `2`（9 帧） |
| 日志被 torchvision 弃用警告刷屏 | 脚本已设 `PYTHONWARNINGS=...`；可忽略 |
| `CUDA is not available` 且无 Step | 检查 DLC 任务是否真分配到 GPU |
| action/state 维数对不上 | 对齐 `modality.json`、DataConfig、`action_dim`/`state_dim` |
| `Robot type ... not found in ROBOT_TYPE_TO_EMBODIMENT_TAG` | 警告可忽略，会回退 `EmbodimentTag.NEW_EMBODIMENT` |

---

## 附录：已落地任务关键文件一览

### OpenOven

| 角色 | 路径 |
|------|------|
| 数据 | `datasets/simple/G1WholebodyOpenOvenTeleop-v0/` |
| DataConfig | `DiT4DiT/dataloader/gr00t_lerobot/data_config.py` → `G1WholebodyTeleopDataConfig` |
| Mix 注册 | `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py` → `G1_OPEN_OVEN` |
| 超参 | `DiT4DiT/config/G1_OPEN_OVEN/dit4dit_G1_OPEN_OVEN.yaml` |
| 启动 | `examples/G1_OPEN_OVEN/train_files/run_G1_OPEN_OVEN.sh` |
| 监控 | `examples/G1_OPEN_OVEN/train_files/watch_train.sh` |

### OpenFaucet

| 角色 | 路径 |
|------|------|
| 数据 | `datasets/simple/G1WholebodyOpenFaucetTeleop-v0/`（`info.json` 的 `video_info` 已补全） |
| DataConfig | 复用 `g1_wholebody_teleop`（与 Oven 相同） |
| Mix 注册 | `mixtures.py` → `G1_OPEN_FAUCET` |
| 超参 | `DiT4DiT/config/G1_OPEN_FAUCET/dit4dit_G1_OPEN_FAUCET.yaml` |
| 启动 | `examples/G1_OPEN_FAUCET/train_files/run_G1_OPEN_FAUCET.sh` |
| 监控 | `examples/G1_OPEN_FAUCET/train_files/watch_train.sh` |

通用微调指南：`FINETUNE_GUIDE.md`（更通用的六步流程）。

---

*文档与当前仓库配置同步：action chunk=16，`action_video_freq_ratio=1`（17 帧），冻 text_encoder+vae，训 transformer+action_model，总步数 10 万、每 1 万存盘。换任务时先补 `video_info`（§4.5.1）。*
