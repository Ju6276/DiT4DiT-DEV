# G1WholebodyOpenOvenTeleop-v0 训练配置方案（草稿）

> 状态：草稿，后续会继续修改。  
> 目标数据集：`datasets/simple/G1WholebodyOpenOvenTeleop-v0`  
> 原则：数据已是 LeRobot v2.1，一般无需再做格式转换；仓库内尚未注册训练入口，需按「DataConfig → mixtures → YAML → 启动脚本」配齐。  
> 参考：乒乓球配置仅作 **YAML/启动脚本的文件结构模板**；modality、归一化、chunk、时序超参以 wholebody / Handover 为准，见正文。  
> 通用流程见 `FINETUNE_GUIDE.md`。

---

## 1. 数据集现状

| 项 | 值 |
|----|-----|
| 路径 | `datasets/simple/G1WholebodyOpenOvenTeleop-v0` |
| 格式 | LeRobot v2.1（`meta/` + `data/` + `videos/`） |
| 规模 | 99 episodes，34713 帧 |
| FPS | **50** |
| 任务语 | `"move forward to the oven and open it"` |
| 相机 | `video.rs_view` → `observation.images.egocentric`（360×640，**仅此单视角**） |
| State | **32 维**：left/right hand(7+7) + left/right arm(7+7) + rpy(3) + height(1) |
| Action | **36 维**：上述 32 + `torso_vx/vy/vyaw` + `target_yaw` |

### 1.1 modality.json 切片（训练必须对齐）

**State（32）**

| key | start:end | 维 |
|-----|-----------|----|
| `left_hand` | 0:7 | 7 |
| `right_hand` | 7:14 | 7 |
| `left_arm` | 14:21 | 7 |
| `right_arm` | 21:28 | 7 |
| `rpy` | 28:31 | 3 |
| `height` | 31:32 | 1 |

**Action（36）** = state 切片 +

| key | start:end | 维 | absolute |
|-----|-----------|----|----------|
| `torso_vx` | 32:33 | 1 | false |
| `torso_vy` | 33:34 | 1 | false |
| `torso_vyaw` | 34:35 | 1 | false |
| `target_yaw` | 35:36 | 1 | true |

**Video / Language**

- `video.rs_view` ← `observation.images.egocentric`
- `annotation.human.task_description` ← `task_index`

### 1.2 统计量

- 已有：`meta/stats.json`（states=32，action=36）
- 缺失：`meta/stats_gr00t.json`（首次 dataloader 加载会自动计算并写入，一般不必手做）
- 文档里提到的根目录 `norm_stats.json`：当前 loader 更依赖 `stats_gr00t` / 自动统计

### 1.3 同系列数据集

`datasets/simple/` 下多数 `G1Wholebody*Teleop-v0`（如 CloseDoor / OpenFaucet / PushOfficeChair）**共用同一套 modality**。  
后续可共享同一个 `robot_type`，在 mixtures 里做多任务混合。

---

## 2. 训练链路（必须对齐的三层）

```
YAML.data_mix
  → mixtures.py（文件夹名 + robot_type）
    → data_config.py（keys 对齐 modality.json）
      → datasets/simple/G1WholebodyOpenOvenTeleop-v0/
```

最终路径：`{data_root_dir}/{mixtures 里的文件夹名}`

推荐：

- `data_root_dir = ./datasets/simple`
- mixture 文件夹名 = `G1WholebodyOpenOvenTeleop-v0`

---

## 3. 为什么不能直接复用现有 G1 配置

| 配置 | 问题 |
|------|------|
| `G1PingpangqiuDataConfig` | 单臂 + torso/teleop_navigate，action=15 / state=27→54，相机 `video.left` |
| `UnitreeG1DataConfig` | 只有四肢，**缺** rpy / height / 底盘四维；虽有 `video.rs_view` 仍不够 |

必须新建 **wholebody teleop** 专用 DataConfig。

---

## 4. 模型模块与冻结 / 训练策略（本数据集）

`training: joint` 时，顶层只有两大块：`backbone_interface`（Cosmos 视频+语言）与 `action_model`（动作 DiT）。  
`freeze_modules` 用逗号分隔的**相对路径**（相对 `DiT4DiT` 根），例如 `backbone_interface.extractor.vae`。

### 4.1 模块树（可冻结路径）

```
DiT4DiT
├── backbone_interface                    # _Cosmos25_Interface
│   └── extractor                         # Cosmos25FeatureExtractor
│       ├── text_encoder                  # 语言编码（任务指令 → token 特征）
│       ├── tokenizer                     # 一般无可训参数 / 不单独训
│       ├── transformer                   # Cosmos 视频 DiT（提特征 + future video FM）
│       ├── vae                           # 视频编解码到 latent
│       ├── scheduler                     # 采样调度，通常无梯度
│       └── video_processor               # 预处理
└── action_model                          # FlowmatchingActionHead（必须适配 36D / chunk=16）
    ├── model                             # 交叉注意力 DiT 主干
    ├── state_encoder                     # state → 嵌入（MLP，输入维=state_dim=32）
    ├── action_encoder                    # 噪声动作轨迹编码
    ├── action_decoder                    # 隐状态 → action_dim=36
    └── position_embedding                # 可选位置嵌入
```

学习率分组（`trainer.learning_rate`）也是按**顶层路径**配的：

| 键 | 作用对象 |
|----|----------|
| `backbone_interface` | Cosmos 整棵（其中未冻结的子模块） |
| `action_model` | 整个动作头 |
| `base` | 其余未点名参数 |

### 4.2 各模块该不该训（针对 OpenOven / wholebody teleop）

| 模块路径 | 建议 | 原因 |
|----------|------|------|
| `backbone_interface.extractor.transformer` | **训练（已定）** | `joint` 下 future video + 给动作头提特征；第一人称全身域适应 |
| `backbone_interface.extractor.text_encoder` | **冻结（已定）** | 任务句单一、短 |
| `backbone_interface.extractor.vae` | **冻结（已定）** | 通用编解码；省显存、稳住 latent |
| `tokenizer` / `scheduler` / `video_processor` | 不单独管 | 无关键可训参数或不参与反传 |
| `action_model`（整棵） | **训练（已定）** | 新 embodiment：36D、chunk=16 |

### 4.3 已确认方案：只训 transformer + action_model

**用户定稿：只训练 `transformer` 与 `action_model`；冻结 `text_encoder` 与 `vae`。**

```text
freeze_modules:
  backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae
```

| 部分 | 状态 | LR |
|------|------|-----|
| `backbone_interface.extractor.text_encoder` | **冻结** | — |
| `backbone_interface.extractor.vae` | **冻结** | — |
| `backbone_interface.extractor.transformer` | **训练** | `backbone_interface: 1e-5` |
| `action_model`（整棵：model / state_encoder / action_encoder / action_decoder / …） | **训练** | `action_model: 1e-4` |

说明：

- 未写入 `freeze_modules` 的 backbone 子模块（主要是 **transformer**）可训；LR 通过 `learning_rate.backbone_interface` 作用在 backbone 下未冻结参数上。
- chunk=16 + `action_video_freq_ratio=2`（约 9 帧进 Cosmos）显存仍可能紧 → **优先降 batch**，不要为此解冻 VAE，也不要默认改成只训 action。
- 备选（仅显存实在不够时）：`freeze_modules: backbone_interface`（整棵 Cosmos 冻，只训 action_model）——弱化视频域适应，非首选。

### 4.4 明确不训的模块

| 模块 | 原因（简要） |
|------|----------------|
| text_encoder | 任务句短且单一，微调收益小 |
| vae | 通用像素↔latent；动它费显存且易拧歪预训练 latent，小数据不划算 |

---

## 5. 落地步骤（计划）

### 步骤 0：环境与基座（一次性）

- Conda：`dit4dit`
- 基座：`./Cosmos-Predict2.5-2B`
- 安装见 `FINETUNE_GUIDE.md` / README
- 数据侧：无需再转换；可选检查 `videos/.../egocentric/*.mp4` 能否解码

### 步骤 1：新建 DataConfig（`DiT4DiT/dataloader/gr00t_lerobot/data_config.py`）

建议类名：`G1WholebodyTeleopDataConfig`  
建议注册名：`g1_wholebody_teleop`

```python
video_keys = ["video.rs_view"]
state_keys = [
    "state.left_hand", "state.right_hand",
    "state.left_arm", "state.right_arm",
    "state.rpy", "state.height",
]
action_keys = [
    "action.left_hand", "action.right_hand",
    "action.left_arm", "action.right_arm",
    "action.rpy", "action.height",
    "action.torso_vx", "action.torso_vy",
    "action.torso_vyaw", "action.target_yaw",
]
language_keys = ["annotation.human.task_description"]
observation_indices = [0]
action_indices = list(range(16))  # action chunk=16，与 YAML action_horizon=16 一致
```

**Transform（已确认采纳：对齐 G1HandoverDataConfig；经 OpenOven + OpenTrashCan 分布核验）**

```python
def transform(self):
    transforms = [
        StateActionToTensor(apply_to=self.state_keys),
        StateActionTransform(
            apply_to=self.state_keys,
            normalization_modes={
                "state.left_hand": "min_max",
                "state.right_hand": "min_max",
                "state.left_arm": "min_max",
                "state.right_arm": "min_max",
                "state.rpy": "min_max",
                "state.height": "min_max",
            },
        ),
        StateActionToTensor(apply_to=self.action_keys),
        StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes={
                "action.left_hand": "min_max",
                "action.right_hand": "min_max",
                "action.left_arm": "min_max",
                "action.right_arm": "min_max",
                "action.rpy": "min_max",
                "action.height": "min_max",
                "action.torso_vx": "mean_std",
                "action.torso_vy": "mean_std",
                "action.torso_vyaw": "mean_std",
                "action.target_yaw": "mean_std",
            },
        ),
    ]
    return ComposedModalityTransform(transforms=transforms)
```

要点：

- State：**不用 SinCos**；全部 `min_max` → YAML `state_dim=32`（不是 64）
- Action：关节/姿态 `min_max`；底盘速度与 yaw（`torso_vx/vy/vyaw`、`target_yaw`）用 `mean_std`
- 核验结论见 §6.2（OpenOven）与 §6.5（OpenTrashCan）：**整体正确，可作为统一 recipe**

注册：

```python
ROBOT_TYPE_CONFIG_MAP["g1_wholebody_teleop"] = G1WholebodyTeleopDataConfig()
```

### 步骤 2：注册 Mixture（`DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`）

单任务（**已定**）：

```python
"G1_OPEN_OVEN": [
    ("G1WholebodyOpenOvenTeleop-v0", 1.0, "g1_wholebody_teleop"),
],
```

多任务暂不启用（同一 `robot_type` 以后可加）。

### 步骤 3：新建 YAML

建议路径：`DiT4DiT/config/G1_OPEN_OVEN/dit4dit_G1_OPEN_OVEN.yaml`  
模板：`DiT4DiT/config/G1_PINGPANGQIU/dit4dit_G1_PINGPANGQIU.yaml`

**必须改（相对默认模板 / 本数据集定稿）**

| 字段 | 建议值 | 依据 |
|------|--------|------|
| `run_id` | `dit4dit_G1_OPEN_OVEN` | 实验隔离 |
| `framework.action_model.action_dim` | **36** | modality action 总维 |
| `framework.action_model.state_dim` | **32** | state 用 min_max，不用 SinCos |
| `framework.action_model.action_horizon` | **16** | **本任务 action chunk=16** |
| `framework.action_model.future_action_window_size` | **15** | = action_horizon - 1 |
| `datasets.vla_data.data_root_dir` | `./datasets/simple` | 数据父目录 |
| `datasets.vla_data.data_mix` | `G1_OPEN_OVEN` | mixtures 注册名 |
| `datasets.vla_data.max_action_dim` | **36** | ≥ action_dim |
| `datasets.vla_data.max_state_dim` | **32** | ≥ state_dim（或更大 pad） |
| `datasets.vla_data.video_delta_indices` | **`[0..16]`** | 与 chunk=16 对齐 |
| `datasets.vla_data.action_video_freq_ratio` | **2** | 隔帧进 Cosmos；`[0..16]` → **9 帧** |
| DataConfig `action_indices` | `list(range(16))` | 必须与 action_horizon 同步 |
| `datasets.vla_data.per_device_batch_size` | **32** | 用户定稿（2 卡 → 全局 BS=64） |
| `trainer.max_train_steps` | **40000** | 用户定稿 |
| `trainer.save_interval` | **10000** | 用户定稿 |
| `trackers` | `[jsonl, wandb]` | **开启 WANDB** |

**本数据集时序（50fps）**

| 项 | 定值 | 说明 |
|----|------|------|
| 原始 FPS | **50** | `info.json` |
| action chunk | **16 步 ≈ 0.32s**（50fps） | 已定 |
| `action_video_freq_ratio` | **2（写死）** | `[0..16]` 再隔帧 → Cosmos **9 帧**（有效约 25fps）。只降视频、不改 action |

**框架 / 训练起步值（来自 DiT4DiT+Cosmos 惯例，不是「因为乒乓球」）**

这些与「是否乒乓球数据」无关；换任何自定义数据也常从这里起步，可按显存与收敛再改：

| 项 | 起步值 | 真正原因 |
|----|--------|----------|
| `image_size` / resize | `[224,224]` | Cosmos/VLA 训练管线固定 resize 到 224，与源分辨率 360×640 无关 |
| `video_backend` | `torchvision_av` | 仓库默认解码后端 |
| `training` | `joint` | 视频 flow matching + 动作联合训练（本框架主设定） |
| `extract_layer` | `17` | Cosmos 特征层惯例 |
| `action_model_type` | `DiT-B` | 动作头容量起步选择 |
| `freeze_modules` | `backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae` | **已定**：只训 transformer + action_model |
| LR | backbone `1e-5`，action `1e-4` | 配合上表 |

### 步骤 4：启动脚本

复制：`examples/G1_PINGPANGQIU/train_files/run_G1_PINGPANGQIU.sh`  
建议目标：`examples/G1_OPEN_OVEN/train_files/run_G1_OPEN_OVEN.sh`

脚本内改：

```bash
data_root_dir=./datasets/simple
data_mix=G1_OPEN_OVEN
run_root_dir=./results/Checkpoints_G1_OPEN_OVEN
run_id=dit4dit_G1_OPEN_OVEN
# --num_processes 2
# --datasets.vla_data.per_device_batch_size 32
# --trainer.max_train_steps 40000
# --trainer.save_interval 10000
# --trackers jsonl wandb  （开启 WANDB）
```

CLI 覆盖建议与现有脚本一致：`flow_matching.time_distribution=uniform`，`high_sigma_*=null` 等。

### 步骤 5：开训前自检

1. `./Cosmos-Predict2.5-2B` 存在
2. `modality.json` keys ↔ DataConfig keys 一一对应
3. `action_dim=36`、`state_dim=32`、`action_horizon=16`、`future_action_window_size=15`、`action_indices=range(16)`、`action_video_freq_ratio=2`、`max_*` 一致
4. `data_root_dir` + mixture 文件夹名能拼出真实路径
5. 首次跑会生成 `meta/stats_gr00t.json`（正常）
6. `info.json` 的 `video_info: {}` 为空多数可忽略；视频元数据报错时再补
7. `video_delta_indices=[0..16]`，且 **`action_video_freq_ratio=2`**

---

## 6. 自我检查：与 G1HandoverDataConfig 对齐

对照另一项目的 `G1HandoverDataConfig` 与本数据集 `modality.json` / `stats.json`。

### 6.1 state/action key 顺序 —— **应保持一致**

OpenOven 的 modality 切片顺序与 Handover 定义完全一致：

- State 32D：`left_hand → right_hand → left_arm → right_arm → rpy → height`
- Action 36D：同上 + `torso_vx → torso_vy → torso_vyaw → target_yaw`
- Video：`video.rs_view`；Language：`annotation.human.task_description`

结论：**keys 与顺序应直接复用 Handover，不要改成 PINGPANGQIU 那套。**

### 6.2 归一化 —— **大体应一致，并修正初稿里的 SinCos**

| 模态 | Handover | OpenOven 数据证据 | 建议 |
|------|----------|-------------------|------|
| state.*（关节/rpy/height） | `min_max` | 有界绝对量；height∈[0.42,0.74] | **跟 Handover：`min_max`** |
| action hand/arm/rpy/height | `min_max` | 有界绝对关节/姿态 | **跟 Handover：`min_max`** |
| action.torso_vx/vy/vyaw | `mean_std` | `absolute=false`，近 0 对称；vyaw 有长尾（min/max 远大于 q01/q99） | **跟 Handover：`mean_std`** |
| action.target_yaw | `mean_std` | absolute=true，但均值≈0、std 小 | **跟 Handover：`mean_std`**（合理） |

与本仓库 PINGPANGQIU 习惯的差异：

- PINGPANGQIU：state 用 **SinCos** → `state_dim=2×raw`
- Handover / 本方案：state 用 **min_max** → `state_dim=32`（原始维）

DiT4DiT 的 `StateActionTransform` **已支持** `min_max` / `mean_std` / `binary` / `q99`，技术上可直接照搬 Handover 的 transform。

### 6.3 OpenOven 特有注意点（不完全等于 Handover）

1. **`action.left_hand` 全为 0**（min=max=mean=std=0）  
   - 本任务左手动作标签是死维；`min_max` 会遇到 0/0，实现里通常要靠 eps 或 mask。  
   - 仍建议保留 7 维以与 Handover / 同系列 modality **布局对齐**，不要删维。  
   - 训练时靠 padding mask / 常数维自然无梯度贡献即可。

2. **`state.left_hand` 变化极小**（多数维接近 0），`min_max` 可用但数值会很敏感。

3. **`action.right_hand`** 有明显开合范围（约到 1.5），**不是** binary；用 `min_max` 正确，不要改成 binary。

4. **`torso_vx`** 的 min/max 恰为 ±0.5 且 q01/q99 顶满边界，`mean_std` 与 `min_max` 差异不大；**`torso_vyaw`** 更值得用 `mean_std`（抑制极值）。

### 6.4 结论（给后续改配置用）

- **应该与 Handover 保持一致**（key 顺序 + 归一化分工）。  
- **不应**沿用初稿/PINGPANGQIU 的「state SinCos + action 全 min_max」。  
- YAML：`action_dim=36`，`state_dim=32`（若坚持 SinCos 才是 64，当前不推荐）。  
- **action chunk = 16**（已定）：`action_horizon=16`，`future_action_window_size=15`，`action_indices=list(range(16))`，`video_delta_indices=[0..16]`。  
- **`action_video_freq_ratio=2`（写死）**：Cosmos 收 9 帧。

### 6.5 二次核验：`G1WholebodyOpenTrashCanTeleop-v0` 数据分布

数据集：99 ep / 32285 帧 / 50fps；`modality.json` 与 OpenOven **完全相同**。  
任务语：`"move forward to the trash can and open it"`。

对 Handover 式 `transform()` 的逐项结论：

| key | 分布要点 | Handover 模式 | 是否合理 |
|-----|----------|---------------|----------|
| state/action 四肢 + rpy | 有界绝对关节/姿态，有真实变化 | `min_max` | **合理** |
| state/action `height` | **恒为 0.74**（min=max，std≈0） | `min_max` | 模式概念对，但本任务是**死维**；实现里 min==max → 归一化为 0，不崩 |
| action `torso_vx` | 相对速度，mean≈0.05，std≈0.14，范围约 [-0.07, 0.5] | `mean_std` | **合理** |
| action `torso_vy` | 近零、**长尾重**（tail≈1.58） | `mean_std` | **更应 mean_std**（比 min_max 更合适） |
| action `torso_vyaw` | **全 0** | `mean_std` | 死维；std==0 时实现回退为原值，不崩 |
| action `target_yaw` | **全 0**（虽 absolute=true） | `mean_std` | 死维；模式选择对本集会失效 |

与 OpenOven 对照（说明「按任务死维不同，共享 recipe 仍必要」）：

| 维 | OpenOven | OpenTrashCan |
|----|----------|--------------|
| `action.left_hand` | 全 0 | 有变化 → `min_max` 有意义 |
| `height` | 有变化 [0.42, 0.74] | 恒 0.74 |
| `torso_vyaw` / `target_yaw` | 有变化 | 全 0 |

**总判（写入定论）：**

> 可以按上述 Handover 式 `transform()` 使用。OpenTrashCan 分布支持该方案（尤其 `torso_vy` 长尾支持 `mean_std`）。  
> 这是 **G1 wholebody teleop 统一 recipe**，不是「每个维在本任务都有信息」；死维靠实现兜底，混训时不要按单任务改 mode。

1. 作为统一 recipe，这套归一化在 TrashCan 上**整体正确**，尤其 `torso_vy` 长尾支持 `mean_std`。  
2. 并非「每个维在本任务上都有信息」：height / torso_vyaw / target_yaw 对本集会退化；这是数据特性，不是模式选错。DiT4DiT 的 `min_max`/`mean_std` 对零 span / 零 std 有兜底。  
3. 若只训 TrashCan，改不改这些死维的 mode **几乎无影响**；若与 OpenOven 等混训，**必须保持 Handover 这套统一布局与归一化**。  
4. 次要瑕疵：`state.left_hand` 极值尾很重、主体贴零，`min_max` 会压缩主体；可考虑日后对 hand 试 `q99`，但不足以否定当前方案。

---

## 7. 待讨论 / 后续可能改动的点

1. ~~单任务 vs 多任务~~：**已定只训** `G1WholebodyOpenOvenTeleop-v0`
2. ~~时间窗 / action chunk / `action_video_freq_ratio`~~：已定 `action_horizon=16`，**`action_video_freq_ratio=2`（写死）**
3. ~~手部归一化~~：已确认用 `min_max`（非 binary）
4. ~~冻结策略~~：**已定** — 只训 `transformer` + `action_model`；冻 `text_encoder` + `vae`
5. ~~命名~~：**已定任务名** — mix/`run_id`/`config` 目录用 `G1_OPEN_OVEN`；`robot_type` 仍为可复用的 `g1_wholebody_teleop`
6. ~~训练资源~~：**已定** — **2 卡**，`per_device_batch_size=32`，`max_train_steps=40000`，`save_interval=10000`，**开启 WANDB**
7. ~~视角~~：**已定单视角** — 仅 `video.rs_view`（`observation.images.egocentric`），无腕部/多相机
8. ~~`info.json` video_info 空~~：已补 `video.fps=50` 等（否则 dataloader `KeyError: video.fps` / `info`）
9. **死维 / hand q99**：保持现状；布局不删维

> 显存提示：chunk=16 + ratio=2（约 9 帧）后显存应明显缓解；若仍 OOM 再降 BS。

---

## 8. 推荐落地顺序

1. ~~确认数据范围 / 命名 / 卡数 batch~~（已定，见 §7）
2. ~~写 DataConfig + mixtures + YAML + `run_*.sh`~~（已落地）
3. Smoke（少 step）验证 dataloader 与维度
4. 全量训练 40000 step；WANDB 记录

**启动：**

```bash
bash examples/G1_OPEN_OVEN/train_files/run_G1_OPEN_OVEN.sh
# 或覆盖 per-device batch：bash examples/G1_OPEN_OVEN/train_files/run_G1_OPEN_OVEN.sh 16
```

---

## 9. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-07-25 | 初稿：基于 OpenOven 元数据与仓库现有 G1 训练链路整理完整方案，尚未改代码 |
| 2026-07-25 | 对照另一项目 G1HandoverDataConfig 做自我检查：keys 应一致；归一化改跟 Handover（弃 SinCos）；state_dim 改为 32；记录 left_hand action 全 0 |
| 2026-07-25 | 用 OpenTrashCan 分布二次核验 Handover transform：整体正确；height/vyaw/target_yaw 本任务死维；torso_vy 长尾支持 mean_std |
| 2026-07-25 | 将已确认采纳的完整 `transform()` 代码与 TrashCan 总判定论显式写入 simple.md |
| 2026-07-25 | **action chunk 定为 32**（原误写 16）：同步改 `action_indices` / `action_horizon=32` / `future_action_window_size=31` / `video_delta_indices=[0..32]`，并改对照表与待讨论项 |
| 2026-07-25 | 删除「可先保持与 PINGPANGQIU 一致」：乒乓球只作 YAML/脚本结构模板；时序与超参按 50fps wholebody 重写 |
| 2026-07-25 | **`action_video_freq_ratio` 写死为 1**（视频不降采样；33 帧进 Cosmos） |
| 2026-07-25 | 删除「与 G1_PINGPANGQIU 对照」整节 |
| 2026-07-25 | 新增 §4 模型模块树与本数据集冻结/训练档位 |
| 2026-07-25 | **冻结策略用户定稿**：只训 transformer + action_model；冻 text_encoder + vae |
| 2026-07-25 | **开训定稿**：单任务 OpenOven；任务名 G1_OPEN_OVEN；2 卡 BS=32；40k step / save 10k；WANDB；确认单视角 rs_view |
| 2026-07-25 | **代码落地**：`G1WholebodyTeleopDataConfig` + mixture `G1_OPEN_OVEN` + YAML + `examples/G1_OPEN_OVEN/train_files/run_G1_OPEN_OVEN.sh` |
| 2026-07-25 | **action chunk 改为 30**：`action_horizon=30` / `future_action_window_size=29` / `action_indices=range(30)` / `video_delta_indices=[0..30]`（31 帧） |
| 2026-07-25 | **action chunk→16，`action_video_freq_ratio→2`**：Cosmos 约 9 帧；缓解 OOM |
