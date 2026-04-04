#!/usr/bin/env python3
"""LADD experiment runner — the ONLY file the research agent modifies.

Hyperparameters are constants at the top. This script generates a bash script
that runs train_ladd.py, extracts the checkpoint, and evaluates — then execs it.
The Python process is fully replaced, so no GPU memory is wasted.

Usage:
    python research/experiment.py
"""

import os
import shutil
import sys
import tempfile

# ---------------------------------------------------------------------------
# HYPERPARAMETERS — the research agent modifies ONLY this section
# ---------------------------------------------------------------------------

# Model & data (fixed for comparability — do not change)
MODEL_PATH = "models/Z-Image"
TRAIN_DATA = "data/debug/metadata.json"
IMAGE_SIZE = 512
SEED = 42

# Training budget (fixed for comparability — do not change)
MAX_TRAIN_STEPS = 500
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1

# === TUNABLE HYPERPARAMETERS START ===

# Learning rates
STUDENT_LR = 1e-5
DISC_LR = 1e-4

# LR schedule
LR_WARMUP_STEPS = 50

# LADD dynamics
GEN_UPDATE_INTERVAL = 5           # D steps per G step
WARMUP_SCHEDULE_STEPS = 50        # timestep warmup
STUDENT_TIMESTEPS = [1.0, 0.75, 0.5, 0.25]

# Noise schedule
RENOISE_M = 1.0                   # logit-normal mean
RENOISE_S = 1.0                   # logit-normal std

# Discriminator architecture
DISC_HIDDEN_DIM = 256
DISC_COND_DIM = 256
DISC_LAYER_INDICES = [5, 10, 15, 20, 25, 29]

# Text
TEXT_DROP_RATIO = 0.1

# === TUNABLE HYPERPARAMETERS END ===

# Infrastructure (fixed)
OUTPUT_DIR = "research/output"
CHECKPOINT_DIR = "research/checkpoint"


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Clean previous output
    for d in [os.path.join(project_root, OUTPUT_DIR),
              os.path.join(project_root, CHECKPOINT_DIR)]:
        if os.path.exists(d):
            shutil.rmtree(d)

    timesteps_str = " ".join(str(t) for t in STUDENT_TIMESTEPS)
    disc_layers_str = " ".join(str(l) for l in DISC_LAYER_INDICES)
    wandb_name = f"exp-lr{STUDENT_LR}-dlr{DISC_LR}-gi{GEN_UPDATE_INTERVAL}"

    script = f"""#!/bin/bash
set -e
cd {project_root}

echo "============================================================"
echo "  STEP 1: Training"
echo "============================================================"

accelerate launch --num_processes=1 --use_deepspeed \\
    --deepspeed_config_file=training/ds_student_config.json \\
    training/train_ladd.py \\
    --pretrained_model_name_or_path={MODEL_PATH} \\
    --train_data_meta={TRAIN_DATA} \\
    --output_dir={OUTPUT_DIR} \\
    --train_batch_size={TRAIN_BATCH_SIZE} \\
    --gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS} \\
    --max_train_steps={MAX_TRAIN_STEPS} \\
    --learning_rate={STUDENT_LR} \\
    --learning_rate_disc={DISC_LR} \\
    --lr_scheduler=constant_with_warmup \\
    --lr_warmup_steps={LR_WARMUP_STEPS} \\
    --mixed_precision=bf16 \\
    --gradient_checkpointing \\
    --allow_tf32 \\
    --cpu_offload_optimizer \\
    --seed={SEED} \\
    --checkpointing_steps={MAX_TRAIN_STEPS} \\
    --validation_steps=999999 \\
    --num_inference_steps=4 \\
    --image_sample_size={IMAGE_SIZE} \\
    --gen_update_interval={GEN_UPDATE_INTERVAL} \\
    --disc_hidden_dim={DISC_HIDDEN_DIM} \\
    --disc_cond_dim={DISC_COND_DIM} \\
    --disc_layer_indices {disc_layers_str} \\
    --student_timesteps {timesteps_str} \\
    --warmup_schedule_steps={WARMUP_SCHEDULE_STEPS} \\
    --renoise_m={RENOISE_M} \\
    --renoise_s={RENOISE_S} \\
    --text_drop_ratio={TEXT_DROP_RATIO} \\
    --dataloader_num_workers=0 \\
    --report_to=wandb \\
    --tracker_project_name=ladd \\
    --wandb_run_name={wandb_name}

echo ""
echo "============================================================"
echo "  STEP 2: Extracting checkpoint"
echo "============================================================"

python3 -c "
import os, torch
from safetensors.torch import save_file
dirs = sorted([d for d in os.listdir('{OUTPUT_DIR}') if d.startswith('checkpoint-')],
    key=lambda x: int(x.split('-')[1]))
ds = os.path.join('{OUTPUT_DIR}', dirs[-1], 'pytorch_model', 'mp_rank_00_model_states.pt')
sd = torch.load(ds, map_location='cpu', weights_only=False)['module']
os.makedirs('{CHECKPOINT_DIR}/student_transformer', exist_ok=True)
save_file({{k: v.contiguous() for k, v in sd.items()}},
    '{CHECKPOINT_DIR}/student_transformer/model.safetensors')
print(f'Extracted {{len(sd)}} keys')
"

# Clean up large DeepSpeed output
rm -rf {OUTPUT_DIR}
echo "Cleaned up {OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "  STEP 3: Evaluating"
echo "============================================================"

python3 research/evaluate.py

echo ""
echo "Experiment complete."
"""

    # Write and exec the bash script (replaces this Python process entirely)
    fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="ladd_exp_")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    # exec replaces this process — no forked Python holding GPU memory
    os.execv("/bin/bash", ["/bin/bash", script_path])


if __name__ == "__main__":
    main()
