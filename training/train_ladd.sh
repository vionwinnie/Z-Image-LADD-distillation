#!/usr/bin/env bash
# LADD Training Launch Script for Z-Image Distillation
# 8-GPU training with FSDP via Accelerate
#
# Usage:
#   bash training/train_ladd.sh
#
# For single-GPU debug:
#   accelerate launch --num_processes=1 training/train_ladd.py \
#       --pretrained_model_name_or_path=models/Z-Image \
#       --train_data_meta=data/debug/metadata.json \
#       --train_batch_size=1 \
#       --max_train_steps=100

set -euo pipefail

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Configuration ---
MODEL_PATH="${MODEL_PATH:-models/Z-Image}"
DATA_META="${DATA_META:-data/train/metadata_subsample.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ladd}"
NUM_GPUS="${NUM_GPUS:-8}"

accelerate launch \
    --config_file training/fsdp_config.yaml \
    training/train_ladd.py \
    --pretrained_model_name_or_path="${MODEL_PATH}" \
    --train_data_meta="${DATA_META}" \
    --clip_embeddings_dir=data/train/clip_embeddings_10k \
    --embeddings_dir=data/train/embeddings_subsample \
    --output_dir="${OUTPUT_DIR}" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=20000 \
    --learning_rate=5e-6 \
    --learning_rate_disc=1e-5 \
    --lr_scheduler=constant_with_warmup \
    --lr_warmup_steps=0 \
    --mixed_precision=bf16 \
    --gradient_checkpointing \
    --allow_tf32 \
    --seed=42 \
    --dataloader_num_workers=4 \
    --checkpointing_steps=2000 \
    --checkpoints_total_limit=3 \
    --validation_steps=2000 \
    --num_inference_steps=4 \
    --image_sample_size=512 \
    --gen_update_interval=3 \
    --disc_layer_indices 5 10 15 20 25 29 \
    --disc_hidden_dim=256 \
    --disc_cond_dim=256 \
    --student_timesteps 1.0 0.75 0.5 0.25 \
    --warmup_schedule_steps=0 \
    --renoise_m=1.0 \
    --renoise_s=1.0 \
    --max_grad_norm=1.0 \
    --skip_baseline_validation \
    --report_to=wandb \
    --tracker_project_name=ladd \
    --wandb_entity=yeun-yeungs \
    --validation_prompts \
        "A beautiful sunset over the ocean with golden clouds" \
        "A cat sitting on a windowsill looking outside" \
        "A futuristic city skyline at night with neon lights" \
        "A watercolor painting of a mountain landscape"
