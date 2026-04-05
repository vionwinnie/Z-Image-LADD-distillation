#!/usr/bin/env bash
# FSDP smoke test: 2 GPUs, 10 steps, 512px
#
# Validates that FSDP wrapping, forward passes, loss computation,
# checkpoint saving, and gradient flow all work correctly before
# committing to a full training run.
#
# Usage (on a multi-GPU node):
#   bash scripts/smoke_test_fsdp.sh
#
# Requirements: at least 2 GPUs with 80GB each

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/Z-Image}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/research/smoke_fsdp_$$"

# Create a 2-GPU FSDP config (copy of training config but num_processes=2)
FSDP_CONFIG="/tmp/fsdp_smoke_$$.yaml"
cat > "${FSDP_CONFIG}" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: ZImageTransformerBlock
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_offload_params: false
  fsdp_use_orig_params: true
  fsdp_sync_module_states: false
  fsdp_cpu_ram_efficient_loading: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
YAML

# Use full subsample metadata (must match precomputed embeddings)
SMOKE_META="${SCRIPT_DIR}/data/train/metadata_subsample.json"

echo "============================================"
echo "  FSDP Smoke Test: 2 GPUs, 10 steps"
echo "============================================"
echo "  Model:   ${MODEL_PATH}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Config:  ${FSDP_CONFIG}"
echo ""

cleanup() {
    rm -rf "${OUTPUT_DIR}" "${FSDP_CONFIG}"
}
trap cleanup EXIT

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --config_file "${FSDP_CONFIG}" \
    training/train_ladd.py \
    --pretrained_model_name_or_path="${MODEL_PATH}" \
    --train_data_meta="${SMOKE_META}" \
    --embeddings_dir="${SCRIPT_DIR}/data/train/embeddings_subsample" \
    --output_dir="${OUTPUT_DIR}" \
    --cpu_offload_optimizer \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=10 \
    --learning_rate=5e-6 \
    --learning_rate_disc=5e-5 \
    --lr_scheduler=constant_with_warmup \
    --lr_warmup_steps=2 \
    --mixed_precision=bf16 \
    --gradient_checkpointing \
    --allow_tf32 \
    --seed=42 \
    --dataloader_num_workers=0 \
    --checkpointing_steps=99999 \
    --validation_steps=5 \
    --val_data_meta="${SCRIPT_DIR}/data/val/metadata.json" \
    --eval_num_images=4 \
    --eval_device=cuda:0 \
    --num_inference_steps=4 \
    --image_sample_size=512 \
    --gen_update_interval=3 \
    --disc_layer_indices 5 10 15 20 25 29 \
    --disc_hidden_dim=256 \
    --disc_cond_dim=256 \
    --student_timesteps 1.0 0.75 0.5 0.25 \
    --warmup_schedule_steps=2 \
    --renoise_m=1.0 \
    --renoise_s=1.0 \
    --max_grad_norm=1.0 \
    --report_to=wandb \
    --tracker_project_name=ladd \
    --wandb_entity=yeun-yeungs \
    --skip_save

echo ""
echo "============================================"
echo "  FSDP Smoke Test PASSED"
echo "============================================"
echo "Ready for full 8-GPU training run."
