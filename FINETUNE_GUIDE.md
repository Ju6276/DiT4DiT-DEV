# DiT4DiT 自定义数据集微调指南

本文档详细说明如何使用你自己的数据集微调 DiT4DiT 模型，共分为 **六个步骤**。

---

## 目录

1. [环境准备](#第一步环境准备)
2. [数据准备](#第二步数据准备最关键)
3. [代码注册](#第三步代码注册需修改-2-个文件)
4. [训练配置](#第四步编写训练配置)
5. [启动训练](#第五步启动训练)
6. [评估与部署](#第六步评估与部署)

---

## 第一步：环境准备

### 1.1 基本要求

- Python >= 3.10
- CUDA >= 12.4
- 推荐 8 张以上 A100/H100 GPU

### 1.2 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

> `requirements.txt` 中的关键依赖：torch==2.7.0, deepspeed==0.18.4, accelerate==1.12.0, transformers==4.57.6, omegaconf==2.3.0。
> 其中 `diffusers` 依赖一个特定的 git commit，见 requirements.txt 第 19 行。

### 1.3 下载 Cosmos-Predict 2.5 基座模型

```bash
# 从 HuggingFace 下载，指定 revision
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
    --revision diffusers/base/post-trained \
    --local-dir ./Cosmos-Predict2.5-2B
```

下载完成后目录结构应包含 `text_encoder/`、`transformer/`、`vae/`、`scheduler/`、`tokenizer/` 等子目录。

---

## 第二步：数据准备（最关键）

DiT4DiT 的 dataloader **只支持 LeRobot v2.0 格式**（见 `DiT4DiT/dataloader/lerobot_datasets.py`）。你需要将原始数据转换为以下磁盘结构。

### 2.1 目标目录结构

```
data_root_dir/
└── your_dataset_name/           # 文件夹名会在 mixtures.py 中引用
    ├── meta/
    │   ├── info.json            # 数据集元信息
    │   ├── modality.json        # 模态映射（最关键）
    │   ├── episodes.jsonl       # 每个 episode 的信息
    │   ├── episodes_stats.jsonl
    │   ├── tasks.jsonl          # 任务语言描述
    │   ├── stats.json           # 归一化统计量
    │   └── relative_stats.json
    ├── norm_stats.json          # 供 StateActionTransform 使用
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            └── {video_key}/     # 如 "left"、"ego_view" 等
                ├── episode_000000.mp4
                ├── episode_000001.mp4
                └── ...
```

### 2.2 关键元数据文件说明

#### `meta/info.json`

定义数据集的特征结构。必须包含以下字段：

```json
{
    "codebase_version": "v2.1",
    "robot_type": "your_robot",
    "total_episodes": 100,
    "total_frames": 50000,
    "total_tasks": 1,
    "total_videos": 0,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 20,
    "splits": {
        "train": "0:100"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.ego_view": {
            "dtype": "image",
            "shape": [720, 1280, 3],
            "names": ["height", "width", "channel"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": ["joint_0", "joint_1", "..."]
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": ["joint_0", "joint_1", "..."]
        },
        "frame_index": {"dtype": "int64", "shape": [1], "names": null},
        "episode_index": {"dtype": "int64", "shape": [1], "names": null},
        "index": {"dtype": "int64", "shape": [1], "names": null},
        "task_index": {"dtype": "int64", "shape": [1], "names": null}
    }
}
```

#### `meta/modality.json`

**这是连接你的数据与 DataConfig 的桥梁。** 它定义了 `state`、`action`、`video`、`annotation` 四种模态如何从 parquet 的原始列中切分/映射出来。

```json
{
    "state": {
        "left_arm": {"start": 0, "end": 7},
        "right_arm": {"start": 7, "end": 14}
    },
    "action": {
        "left_arm": {"start": 0, "end": 7},
        "right_arm": {"start": 7, "end": 14}
    },
    "video": {
        "ego_view": {"original_key": "observation.images.ego_view"}
    },
    "annotation": {
        "human.task_description": {"original_key": "task_index"}
    }
}
```

- `state` / `action` 中的 key（如 `"left_arm"`）对应 DataConfig 里的 `state.left_arm` / `action.left_arm`
- `video` 中的 key（如 `"ego_view"`）对应 DataConfig 里的 `video.ego_view`
- `annotation` 中的 key（如 `"human.task_description"`）对应 DataConfig 里的 `annotation.human.task_description`

#### `meta/tasks.jsonl`

每行一个 JSON 对象，定义 `task_index` 与语言描述的映射：

```jsonl
{"task_index": 0, "task": "Pick up the red cup and place it on the plate."}
{"task_index": 1, "task": "Open the drawer and put the bottle inside."}
```

#### `meta/episodes.jsonl`

每行一个 JSON 对象，描述每个 episode 的元信息：

```jsonl
{"episode_index": 0, "tasks": ["Pick up the red cup and place it on the plate."], "length": 500}
{"episode_index": 1, "tasks": ["Pick up the red cup and place it on the plate."], "length": 430}
```

#### `meta/stats.json`

每个特征的归一化统计量（mean, std, min, max, q01, q99）。用于训练时 `StateActionTransform` 对 state/action 做归一化。需要你提前遍历整个数据集计算。

#### `norm_stats.json`

放在数据集根目录下（与 `meta/` 同级），结构示例：

```json
{
    "norm_stats": {
        "actions": {
            "mean": [0.0, 0.0, ...],
            "std": [1.0, 1.0, ...],
            "q01": [-0.5, -0.5, ...],
            "q99": [0.5, 0.5, ...],
            "min": [-1.0, -1.0, ...],
            "max": [1.0, 1.0, ...]
        },
        "states": {
            "mean": [...],
            "std": [...],
            "q01": [...],
            "q99": [...],
            "min": [...],
            "max": [...]
        }
    }
}
```

### 2.3 Parquet 文件内容

每个 `episode_XXXXXX.parquet` 存储一个 episode 的逐帧数据，列包含：
- `observation.state`：float32 数组
- `action`：float32 数组
- `frame_index`、`episode_index`、`index`、`task_index`：int64
- 视频帧不存在 parquet 中，而是以 MP4 文件存储在 `videos/` 目录下

### 2.4 视频要求

- 格式：MP4
- 分辨率：任意（训练时会被自动 resize 到 **224x224**）
- 视频后端：`torchvision_av`（默认）或 `decord`
- 帧 0 作为 Cosmos 的 conditioning frame，后续帧用于 future video loss

### 2.5 数据转换参考

如果你的原始数据是 HDF5、ROS bag 等格式，需自行编写转换脚本。可参考：
- `examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py`

---

## 第三步：代码注册（需修改 2 个文件）

### 3.1 在 `data_config.py` 中新增 DataConfig 类并注册 robot_type

编辑文件 `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`，参考现有类（如 `UnitreeG1AlohaOnlyArmsDataConfig`）新增你的配置。

需要定义以下内容：

```python
class MyRobotDataConfig(BaseDataConfig):
    # 视频流 key — 必须与 modality.json 中 video 部分的 key 对应
    # 格式："video.{modality.json 中 video 下的 key}"
    video_keys = ["video.ego_view"]

    # 状态 key — 必须与 modality.json 中 state 部分的 key 对应
    # 格式："state.{modality.json 中 state 下的 key}"
    state_keys = [
        "state.left_arm",
        "state.right_arm",
    ]

    # 动作 key — 必须与 modality.json 中 action 部分的 key 对应
    # 格式："action.{modality.json 中 action 下的 key}"
    action_keys = [
        "action.left_arm",
        "action.right_arm",
    ]

    # 语言指令 key — 必须与 modality.json 中 annotation 部分的 key 对应
    # 格式："annotation.{modality.json 中 annotation 下的 key}"
    language_keys = ["annotation.human.task_description"]

    # 观测帧索引，通常为 [0]
    observation_indices = [0]

    # 动作 chunk 长度，list(range(N)) 表示预测未来 N 步动作
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # state 转 tensor 并做 sin/cos 编码（适用于关节角度）
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action 转 tensor 并做 min_max 归一化
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
```

**归一化方式说明**（`normalization_modes` 可选值）：

| 方式 | 适用场景 |
|------|----------|
| `min_max` | 连续值（关节角度、末端位姿等） |
| `binary` | 二值量（夹爪开合） |
| `q99` | 范围较大的连续值，用 1%/99% 分位数裁剪 |

然后在同一文件底部的 `ROBOT_TYPE_CONFIG_MAP` 中注册：

```python
ROBOT_TYPE_CONFIG_MAP = {
    # ... 已有条目 ...
    "my_robot": MyRobotDataConfig(),   # <-- 新增
}
```

### 3.2 在 `mixtures.py` 中注册 Dataset Mixture

编辑文件 `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`，在 `DATASET_NAMED_MIXTURES` 字典中新增条目：

```python
DATASET_NAMED_MIXTURES = {
    # ... 已有条目 ...

    "my_dataset_mix": [
        # (数据集文件夹名, 采样权重, robot_type名)
        ("your_dataset_folder_name", 1.0, "my_robot"),
    ],

    # 如果有多个数据集混合训练：
    "my_multi_dataset_mix": [
        ("dataset_task_A", 1.0, "my_robot"),
        ("dataset_task_B", 0.5, "my_robot"),   # 权重 0.5 表示采样概率更低
    ],
}
```

**注意**：元组第一个元素 `"your_dataset_folder_name"` 必须与 `data_root_dir` 下的实际文件夹名完全一致。

### 3.3 关键对应关系速查

```
modality.json 中的 key          DataConfig 中的引用方式
─────────────────────────       ────────────────────────
state.left_arm                  "state.left_arm"     (state_keys)
action.left_arm                 "action.left_arm"    (action_keys)
video.ego_view                  "video.ego_view"     (video_keys)
annotation.human.task_description  "annotation.human.task_description" (language_keys)
```

---

## 第四步：编写训练配置

### 4.1 创建 YAML 配置文件

复制 `DiT4DiT/config/libero/dit4dit_libero.yaml` 作为模板，在 `DiT4DiT/config/` 下创建你的配置文件。

需要修改的关键字段：

```yaml
framework:
  cosmos25:
    base_model: ./Cosmos-Predict2.5-2B       # Cosmos 权重本地路径

  action_model:
    action_dim: 14                            # 你的 action 总维度
    state_dim: 28                             # 你的 state 总维度（sin/cos 编码后会翻倍）
    future_action_window_size: 15             # = len(action_indices) - 1
    action_horizon: 16                        # = len(action_indices)

datasets:
  vla_data:
    data_root_dir: /path/to/your/data_root    # 数据集根目录
    data_mix: my_dataset_mix                  # mixtures.py 中注册的名称
    max_state_dim: 28                         # >= state_dim
    max_action_dim: 14                        # >= action_dim
    video_delta_indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    action_video_freq_ratio: 2                # 视频帧降采样率
    per_device_batch_size: 4                  # 根据显存调整

trainer:
  max_train_steps: 100000                     # 根据数据量调整
  save_interval: 5000
  freeze_modules: ''                          # 空字符串 = 全部可训练
  learning_rate:
    backbone_interface: 1.0e-05               # Cosmos backbone 学习率
    action_model: 1.0e-04                     # Action DiT 学习率
```

### 4.2 冻结策略建议

| 策略 | `freeze_modules` 值 | 适用场景 |
|------|---------------------|----------|
| 冻结文本编码器和VAE | `backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae` | 推荐起步，节省显存 |
| 全部可训练 | `''`（空字符串） | 数据充足时效果更好，显存需求更大 |

### 4.3 维度计算注意事项

- 如果 `transform()` 中使用了 `StateActionSinCosTransform`，state_dim = 原始关节维度 x 2
- `future_action_window_size` = `action_horizon` - 1
- `max_state_dim` 和 `max_action_dim` 必须 >= 实际维度（用于 padding）

### 4.4 加载已有 DiT4DiT Checkpoint（可选）

如果不是从 Cosmos 原始权重开始，而是从已有的 DiT4DiT checkpoint 继续训练：

```bash
--trainer.pretrained_checkpoint /path/to/checkpoint.pt \
--trainer.reload_modules "backbone_interface,action_model"
```

---

## 第五步：启动训练

### 5.1 编写启动脚本

参考 `examples/LIBERO/train_files/run_libero.sh`，创建你的训练脚本：

```bash
#!/bin/bash
export WANDB_API_KEY=your_wandb_api_key
export PYTHONPATH=$(pwd)
export WANDB_MODE=online

# ---- 配置区 ----
base_model=./Cosmos-Predict2.5-2B
data_root_dir=/path/to/your/data_root
data_mix=my_dataset_mix
freeze_module_list="backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"

run_root_dir=./results/my_experiment
run_id=run_001

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# ---- 启动训练 ----
accelerate launch \
  --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  DiT4DiT/training/train.py \
  --config_yaml DiT4DiT/config/my_config/dit4dit_my_robot.yaml \
  --framework.cosmos25.base_model ${base_model} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --trainer.learning_rate.backbone_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.num_warmup_steps 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project my_project \
  --wandb_entity my_entity
```

### 5.2 关键参数说明

| 参数 | 说明 |
|------|------|
| `--num_processes` | GPU 数量 |
| `--config_file` | DeepSpeed 配置，推荐 `deepspeed_zero2.yaml` |
| `--per_device_batch_size` | 单卡 batch size，8 卡 A100 建议 4-16 |
| `--max_train_steps` | 总训练步数 |
| `--save_interval` | 每隔多少步保存 checkpoint |
| `--action_video_freq_ratio` | 视频帧降采样（2 = 每隔 1 帧取 1 帧给 Cosmos） |

### 5.3 监控训练

- 通过 **Weights & Biases** 监控 `action_loss` 和 `future_video_loss`
- Checkpoint 保存在 `{run_root_dir}/{run_id}/checkpoints/steps_XXXXX/`

### 5.4 断点续训

设置以下参数即可从最近的 checkpoint 恢复训练：

```bash
--trainer.is_resume true
```

---

## 第六步：评估与部署

### 6.1 加载 Checkpoint

Checkpoint 保存路径：`{run_root_dir}/{run_id}/checkpoints/steps_XXXXX/`

加载时需确保 checkpoint 目录下的 `config.yaml` 中 `base_model` 路径指向本地 Cosmos 权重。

### 6.2 部署推理

项目提供了 WebSocket policy server，位于 `deployment/model_server/`，可用于实时推理。

各任务的评估脚本位于 `examples/` 下对应任务的 `eval_files/` 目录中。

---

## 常见问题

### Q1：训练时报 key 找不到

检查 `modality.json` 中的 key 是否与 `data_config.py` 中 DataConfig 的 `video_keys`、`state_keys`、`action_keys`、`language_keys` 严格对应。映射规则：

```
modality.json:  {"state": {"left_arm": {...}}}
DataConfig:     state_keys = ["state.left_arm"]
```

### Q2：action_dim / state_dim 不匹配

- YAML 中的 `action_dim` 必须等于你所有 `action_keys` 维度之和
- 如果使用了 `StateActionSinCosTransform`，`state_dim` = 原始维度 x 2
- `max_action_dim` >= `action_dim`，`max_state_dim` >= `state_dim`

### Q3：显存不足 (OOM)

- 减小 `per_device_batch_size`（如 2 或 1）
- 增加 `gradient_accumulation_steps` 以保持等效 batch size
- 冻结更多模块（text_encoder + vae）
- 使用 DeepSpeed ZeRO-3（`deepspeed_zero3.yaml`）

### Q4：norm_stats.json 如何生成

需要遍历整个数据集，计算 action 和 state 的 mean、std、min、max、q01、q99 统计量，然后写入 JSON。维度需要 pad 到 `max_action_dim` / `max_state_dim`（不足部分填 0）。
