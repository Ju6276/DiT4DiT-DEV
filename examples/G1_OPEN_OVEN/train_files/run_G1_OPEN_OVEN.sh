#!/bin/bash
source /cpfs_infra/shared/xiaoxinyu/opt/miniconda3/etc/profile.d/conda.sh
conda activate dit4dit
export PYTHONPATH=$(pwd)

# 抑制 torchvision 视频解码弃用警告刷屏（不影响训练）
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"

batch_size=${1:-8}

Framework_name=DiT4DiT
base_model=./Cosmos-Predict2.5-2B
freeze_module_list="backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"
DIT_TYPE="DiT-B"
data_root_dir=./datasets/simple
data_mix=G1_OPEN_OVEN

run_root_dir=./results/Checkpoints_G1_OPEN_OVEN
run_id=dit4dit_G1_OPEN_OVEN

: "${WANDB_API_KEY:?Please export WANDB_API_KEY before starting training}"
export WANDB_MODE=online

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

echo "=============================================="
echo " Launch: 8 GPUs | per_device_batch_size=${batch_size} | total_bs=$((batch_size * 8))"
echo " Monitor (DSW/本地看 CPFS):"
echo "   bash examples/G1_OPEN_OVEN/train_files/watch_train.sh"
echo " 成功标志: train.log 出现 Step 10/20/... 且文件持续更新"
echo "=============================================="

accelerate launch \
  --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  DiT4DiT/training/train.py \
  --config_yaml ./DiT4DiT/config/G1_OPEN_OVEN/dit4dit_G1_OPEN_OVEN.yaml \
  --framework.name ${Framework_name} \
  --framework.cosmos25.base_model ${base_model} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --datasets.vla_data.action_video_freq_ratio 1 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.backbone_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.num_warmup_steps 2000 \
  --framework.cosmos25.extract_layer 17 \
  --framework.cosmos25.flow_matching.time_distribution uniform \
  --framework.cosmos25.flow_matching.high_sigma_ratio null \
  --framework.cosmos25.flow_matching.high_sigma_min null \
  --framework.cosmos25.conditional_frame_timestep 0.0001 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --trackers jsonl wandb \
  --wandb_project DiT4DiT \
  --wandb_entity xinyu-xiao-kinetix-ai
