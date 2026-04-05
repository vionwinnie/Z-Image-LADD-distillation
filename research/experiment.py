#!/usr/bin/env python3
"""LADD experiment runner — the ONLY file the research agent modifies.

Hyperparameters are constants at the top. This script generates a bash script
that trains, then evaluates — then execs it so no Python parent holds GPU memory.

Usage:
    python research/experiment.py
"""

import os
import shutil
import tempfile

# ---------------------------------------------------------------------------
# HYPERPARAMETERS — the research agent modifies ONLY this section
# ---------------------------------------------------------------------------

# Model & data (fixed for comparability — do not change)
MODEL_PATH = "models/Z-Image"
TRAIN_DATA = "data/debug/metadata.json"
EMBEDDINGS_DIR = "data/debug/embeddings"
IMAGE_SIZE = 512
SEED = 42

# Training budget (fixed for comparability — do not change)
MAX_TRAIN_STEPS = 500
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1

# === TUNABLE HYPERPARAMETERS START ===

# Learning rates
STUDENT_LR = 5e-6
DISC_LR = 5e-5

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

# Infrastructure (fixed — do not change)
OUTPUT_DIR = "research/output"


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Clean previous output
    out_path = os.path.join(project_root, OUTPUT_DIR)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    timesteps_str = " ".join(str(t) for t in STUDENT_TIMESTEPS)
    disc_layers_str = " ".join(str(l) for l in DISC_LAYER_INDICES)
    wandb_name = f"exp-lr{STUDENT_LR}-dlr{DISC_LR}-gi{GEN_UPDATE_INTERVAL}"

    # The checkpoint lands at research/output/checkpoint-{MAX_TRAIN_STEPS}/student_transformer/
    checkpoint_dir = f"{OUTPUT_DIR}/checkpoint-{MAX_TRAIN_STEPS}"

    script = f"""#!/bin/bash
set -e
cd {project_root}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "  STEP 1: Training ({MAX_TRAIN_STEPS} steps, {IMAGE_SIZE}px)"
echo "============================================================"

accelerate launch --num_processes=1 \\
    training/train_ladd.py \\
    --pretrained_model_name_or_path={MODEL_PATH} \\
    --train_data_meta={TRAIN_DATA} \\
    --embeddings_dir={EMBEDDINGS_DIR} \\
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
    --skip_save \\
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
echo "  STEP 2: Evaluating"
echo "============================================================"

python3 research/evaluate.py --checkpoint {checkpoint_dir}

echo ""
echo "Experiment complete."
"""

    # Write and exec the bash script (replaces this Python process entirely)
    fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="ladd_exp_")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    os.execv("/bin/bash", ["/bin/bash", script_path])


if __name__ == "__main__":
    main()
