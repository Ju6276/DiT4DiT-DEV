# DiT4DiT RoboCasa-GR1 推理 Pipeline

> 本文件只记录可直接复制执行的命令，无需改动代码。

---

## 一、安装 dit4dit 环境（服务端）

```bash
cd /home/d024/DiT4DiT

conda create -n dit4dit python=3.10 -y
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate dit4dit

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

> `requirements.txt` 相比仓库原版有三处修改（已保存在文件中）：
> - `sympy==1.13.1` → `sympy==1.13.3`
> - `triton==3.2.0` → `triton==3.3.0`
> - 删除所有 `nvidia-cuda-*` / `nvidia-cudnn-*` 等固定版本行

---

## 二、安装 robocasa 环境（仿真客户端）

```bash
conda create -n robocasa python=3.10 -y
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate robocasa

pip install robosuite==1.5.1
pip install robocasa-gr1-tabletop-tasks==0.2.0
pip install mujoco==3.2.6
pip install gymnasium tyro
```

下载仿真资产（只需执行一次，约几 GB）：

```bash
cd /home/d024/miniconda3/envs/robocasa/lib/python3.10/site-packages/robocasa
python scripts/download_tabletop_assets.py -y
```

---

## 三、下载模型权重

### 3.1 Hugging Face 登录

```bash
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate dit4dit
hf auth login
```

### 3.2 Cosmos-Predict2.5-2B backbone（diffusers 格式）

```bash
hf download nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --local-dir /home/d024/models/Cosmos-Predict2.5-2B-diffusers-base-post-trained
```

### 3.3 DiT4DiT RoboCasa-GR1 checkpoint

```bash
hf download mondo-robotics/dit4dit-model \
  --include "dit4dit_robocasa_gr1/*" \
  --local-dir /home/d024/DiT4DiT/models/dit4dit-model
```

---

## 四、修改 checkpoint 配置

编辑：`/home/d024/DiT4DiT/models/dit4dit-model/dit4dit_robocasa_gr1/config.yaml`

找到 `base_model` 那行，改为：

```yaml
base_model: /home/d024/models/Cosmos-Predict2.5-2B-diffusers-base-post-trained
```

（已在本机完成）

---

## 五、启动推理（两个终端）

### 终端 A：启动策略服务（dit4dit 环境）

```bash
cd /home/d024/DiT4DiT
deactivate 2>/dev/null; true
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate dit4dit

CUDA_VISIBLE_DEVICES=0 /home/d024/miniconda3/envs/dit4dit/bin/python \
  deployment/model_server/server_policy.py \
  --ckpt_path /home/d024/DiT4DiT/models/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt \
  --port 6398
```

> 等待出现 `server running ...` 字样后再开终端 B。
> RTX 系列 GPU 不加 `--use_bf16`。

### 终端 B：启动仿真评测（robocasa 环境）

```bash
cd /home/d024/DiT4DiT
deactivate 2>/dev/null; true
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate robocasa
export PYTHONPATH=/home/d024/DiT4DiT:$PYTHONPATH

/home/d024/miniconda3/envs/robocasa/bin/python \
  examples/Robocasa_tabletop/eval_files/simulation_env.py \
  --args.env_name "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env" \
  --args.port 6398 \
  --args.pretrained_path /home/d024/DiT4DiT/models/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt \
  --args.n_episodes 1 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12
```

> smoke test 用 `--args.n_episodes 1`，正式评测改为 `50`。

---

## 六、批量评测（全部 24 个任务）

先修改脚本顶部两个变量：

```bash
# examples/Robocasa_tabletop/eval_files/batch_eval_args.sh
MODEL_PYTHON=/home/d024/miniconda3/envs/dit4dit/bin/python
ROBOCASA_PYTHON=/home/d024/miniconda3/envs/robocasa/bin/python
```

然后执行：

```bash
cd /home/d024/DiT4DiT
export PYTHONPATH=/home/d024/DiT4DiT:$PYTHONPATH

bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh \
  /home/d024/DiT4DiT/models/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt \
  1 \
  720 \
  12 \
  "0"
```

> 单卡只写 `"0"`，多卡写 `"0,1,2,3"` 等。

---

## 七、查看结果

```bash
python utils/calculate_robocasa_success_rate.py \
  --log_dir /home/d024/DiT4DiT/models/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt.log/
```