#!/bin/bash
source /cpfs_infra/shared/xiaoxinyu/opt/miniconda3/etc/profile.d/conda.sh
conda activate dit4dit
export PYTHONPATH=$(pwd)

batch_size=${1:-4}

Framework_name=DiT4DiT
base_model=./Cosmos-Predict2.5-2B
freeze_module_list="backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"
DIT_TYPE="DiT-B"
data_root_dir=./datasets
data_mix=G1_PINGPANGQIU

run_root_dir=./results/Checkpoints_G1_PINGPANGQIU
run_id=dit4dit_G1_PINGPANGQIU

: "${WANDB_API_KEY:?Please export WANDB_API_KEY before starting training}"
export WANDB_MODE=online

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 6 \
  DiT4DiT/training/train.py \
  --config_yaml ./DiT4DiT/config/G1_PINGPANGQIU/dit4dit_G1_PINGPANGQIU.yaml \
  --framework.name ${Framework_name} \
  --framework.cosmos25.base_model ${base_model} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.backbone_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.num_warmup_steps 5000 \
  --framework.cosmos25.extract_layer 17 \
  --framework.cosmos25.flow_matching.time_distribution uniform \
  --framework.cosmos25.flow_matching.high_sigma_ratio null \
  --framework.cosmos25.flow_matching.high_sigma_min null \
  --trainer.framework.cosmos25.conditional_frame_timestep 0.0001 \
  --datasets.vla_data.action_video_freq_ratio 2 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --trackers jsonl \
  --wandb_project DiT4DiT \
  --wandb_entity xinyu-xiao-kinetix-ai
