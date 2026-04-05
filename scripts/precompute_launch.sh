#!/usr/bin/env bash
# Launch 8 parallel precompute processes, one per GPU.
# Each GPU processes its shard of prompts independently.
#
# Usage:
#   bash scripts/precompute_launch.sh
#
# Override defaults with environment variables:
#   MODEL_DIR=models/Z-Image NUM_GPUS=4 BATCH_SIZE=8 bash scripts/precompute_launch.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/Z-Image}"
DATA_META="${DATA_META:-data/train/metadata_subsample.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/train/teacher_latents}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE="${GUIDANCE:-5.0}"

echo "Precomputing teacher latents: ${NUM_GPUS} GPUs, batch_size=${BATCH_SIZE}"
echo "  Model:  ${MODEL_DIR}"
echo "  Data:   ${DATA_META}"
echo "  Output: ${OUTPUT_DIR}"

pids=()
for rank in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$rank python data/precompute_teacher_latents.py \
        --model_dir "${MODEL_DIR}" \
        --data_meta "${DATA_META}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --rank "$rank" \
        --world_size "${NUM_GPUS}" \
        --num_inference_steps "${NUM_STEPS}" \
        --guidance_scale "${GUIDANCE}" &
    pids+=($!)
done

echo "Launched ${NUM_GPUS} processes: ${pids[*]}"
echo "Waiting for all to finish..."

# Wait for all and track failures
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Process $pid failed"
        ((failed++))
    fi
done

if [ $failed -gt 0 ]; then
    echo "WARNING: $failed process(es) failed"
    exit 1
fi

echo "All precompute processes completed successfully."
