# DiT4DiT on HumanoidArena SONIC40

This branch adapts the official `Mondo-Robotics/DiT4DiT` model to the same
per-task HumanoidArena benchmark used by GR00T N1.7, Psi0, pi0.5, and VLA-JEPA.
HumanoidArena is used only for evaluation; training stays in this repository.

## Frozen interface

Each inference call receives one front RGB image `[480,640,3]` (HWC,
`uint8`), one state vector `[64]` (`float32`), and a canonical English prompt.
The server validates this external image and applies DiT4DiT's training-time
224×224 resize internally.
The state order is:

```text
[0:6]    heading-canonical root rotation, row-layout rotation 6D
[6:35]   canonical Unitree G1 joint positions, 29D
[35:64]  canonical Unitree G1 joint velocities, 29D
```

The model returns a `[30,40]` `semantic_v3` action chunk. Every step is:

```text
[0:2]    reference-root base-local XY delta
[2:3]    reference-root Z
[3:9]    reference-root row-layout rotation 6D
[9:38]   canonical Unitree G1 joint reference positions, 29D
[38:40]  left and right binary hand commands
```

The first 38 action dimensions use DiT4DiT's native continuous min-max
normalization. The final two hand dimensions remain binary during training and
are thresholded back to exact `0/1` values by the server. These 40 values are
semantic reference-pose fields, not SONIC latent tokens.

## Data

Use the seven shared, converted LeRobot v2.1 datasets. Do not use TWIST2 data
and do not reconvert data separately in this repository:

```text
${DATA_ROOT}/humanoidarena_sonic_v31_opendoor
${DATA_ROOT}/humanoidarena_sonic_v31_double_desk
${DATA_ROOT}/humanoidarena_sonic_v31_football
${DATA_ROOT}/humanoidarena_sonic_v31_pp_box
${DATA_ROOT}/humanoidarena_sonic_v31_boxing
${DATA_ROOT}/humanoidarena_sonic_v31_sit_sofa
${DATA_ROOT}/humanoidarena_sonic_v31_vision_navi
```

Validate any task before training:

```bash
python examples/HumanoidArena/validate_dataset.py \
  "${DATA_ROOT}/humanoidarena_sonic_v31_opendoor"
```

## Train one policy per task

Install DiT4DiT exactly as described in the upstream README and provide the
official diffusers-format Cosmos-Predict2.5-2B base model. The benchmark keeps
DiT4DiT's native AdamW, bf16, Accelerate, and DeepSpeed ZeRO-2 setup. It also
matches the official downstream-task scripts by freezing the Cosmos text
encoder and VAE while training the video transformer/interface and ActionDiT.

For the formal configuration (eight A100 GPUs, one sample per GPU, global batch
8, 100,000 optimizer steps, W&B online):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export GLOBAL_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export DATA_ROOT=/path/to/shared/data_v2
export BASE_MODEL=/path/to/Cosmos-Predict2.5-2B-diffusers-base-post-trained
export OUTPUT_ROOT=/path/to/checkpoints/humanoidarena
export WANDB_MODE=online
export WANDB_ENTITY=your-wandb-entity
export WANDB_PROJECT=HumanoidArena

bash scripts/dit4dit_humanoidarena_sonic40.sh opendoor
```

Train the other six policies sequentially:

```bash
for task in double_desk football pp_box boxing sit_sofa vision_navi; do
  bash scripts/dit4dit_humanoidarena_sonic40.sh "${task}"
done
```

The wrapper verifies that
`global_batch = processes * per_device_batch * gradient_accumulation`. For the
formal command this is `8 = 8 * 1 * 1`. Each task receives its own run ID and
checkpoint directory.

## Serve a checkpoint

Pass a checkpoint `.pt` file, not merely the run directory. Both an
intermediate `checkpoints/steps_*_pytorch_model.pt` file and
`final_model/pytorch_model.pt` have the required run-level `config.yaml` and
`dataset_statistics.json` parents.

```bash
python examples/HumanoidArena/serve_humanoidarena.py \
  --policy-path /path/to/run/final_model/pytorch_model.pt \
  --device cuda:0 --host 127.0.0.1 --port 8000
```

The server implements `POST /infer` and `POST /reset`, accepts the standard
HumanoidArena payload, and returns `{"action_chunk": ...}`.

## Evaluate in HumanoidArena

Use HumanoidArena's existing `sonic_pi05` task wrapper with:

```bash
export SERVER_SCRIPT=/path/to/DiT4DiT/examples/HumanoidArena/serve_humanoidarena.py
export SERVER_PYTHON=/path/to/dit4dit/environment/bin/python
export MODEL_PATHS_CSV=/path/to/run/final_model/pytorch_model.pt
export SONIC_VLA_ACTION_FORMAT=semantic_v3
```

First run one headless episode with recording disabled. Only after that smoke
contains no `process_error` or `worker_error`, run the formal base evaluation:
three seeds (`0,1,2`) times 20 repeats, for exactly 60 episodes, with recording
enabled. Use a fresh results directory for the formal run.

## Single-4090 diagnostic

This command is only a real one-step compatibility/memory test. It deliberately
uses global batch 1 and must not be reported as a benchmark training run. On a
48 GB 4090D, the official frozen-module recipe reaches forward and backward but
still runs out of memory when the first AdamW step allocates its moment state:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 GLOBAL_BATCH_SIZE=1 \
MAX_STEPS=1 SAVE_INTERVAL=1 WANDB_MODE=disabled \
DATA_ROOT=/path/to/shared/data_v2 \
BASE_MODEL=/path/to/Cosmos-Predict2.5-2B-diffusers-base-post-trained \
bash scripts/dit4dit_humanoidarena_sonic40.sh opendoor
```
