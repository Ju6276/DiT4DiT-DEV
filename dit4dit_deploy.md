# DiT4DiT 真实机器人部署指南（通用版）

本指南适用于在任意服务器/工作站上部署 DiT4DiT 推理服务（仅真机，无仿真）。所有命令可直接复制执行，请根据实际路径调整相关变量。

---

## 0. 变量定义（请根据实际情况修改）

```bash
# 工作目录（DiT4DiT 源码目录）
WORKDIR=~/DiT4DiT
# 模型主干（Cosmos-Predict2.5-2B）存放目录
BACKBONE_DIR=~/models/Cosmos-Predict2.5-2B-diffusers-base-post-trained
# 你的 checkpoint 目录（包含 config.yaml 和权重文件）
CKPT_DIR=~/my_checkpoints/your_model_dir
# Python 环境名
ENV_NAME=dit4dit
# CUDA 可见设备
CUDA_DEV=0
```

---

## 1. 安装部署环境

```bash
git clone https://github.com/Mondo-Robotics/DiT4DiT.git "$WORKDIR"
cd "$WORKDIR"

conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

---

## 2. 下载模型权重

### 2.1 Hugging Face 登录

```bash
conda activate $ENV_NAME
hf auth login
```

### 2.2 下载 Cosmos-Predict2.5-2B backbone

```bash
hf download nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --local-dir "$BACKBONE_DIR"
```

### 2.3 准备你的 checkpoint

- 如果是自己训练的模型，将 config.yaml 和权重文件放到 $CKPT_DIR 下。
- 若用官方权重，可用如下命令：

```bash
hf download mondo-robotics/dit4dit-model \
  --include "dit4dit_robocasa_gr1/*" \
  --local-dir "$WORKDIR/models/dit4dit-model"
```

---

## 3. 修改 checkpoint 配置

编辑 $CKPT_DIR/config.yaml，找到 base_model 那行，改为：

```yaml
base_model: <你的 $BACKBONE_DIR 路径>
```

---

## 4. 启动推理服务

```bash
cd "$WORKDIR"
deactivate 2>/dev/null; true
conda activate $ENV_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python \
  deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT_DIR/final_model/pytorch_model.pt" \
  --port 6398
```

- 若权重文件名或路径不同，请自行调整。
- 启动后出现 `server running ...` 即部署成功。
- RTX 系列 GPU 不加 `--use_bf16`。
- 端口可根据实际需求调整。

---
