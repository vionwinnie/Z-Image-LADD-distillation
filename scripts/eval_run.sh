#!/usr/bin/env bash
# eval_run.sh -- Evaluate LADD checkpoints with a persistent wandb run.
#
# Manages a single wandb run across multiple checkpoint evaluations so that
# successive val_step values accumulate as a learning curve on one dashboard.
#
# Usage:
#   # Evaluate teacher baseline at step 0
#   bash scripts/eval_run.sh baseline 0
#
#   # Evaluate a specific safetensors checkpoint at step 2000
#   bash scripts/eval_run.sh output/ladd/checkpoint-2000/model.safetensors 2000
#
#   # Evaluate ALL checkpoints in a training output directory
#   bash scripts/eval_run.sh --all output/ladd
#
# Environment variables (all optional, with defaults):
#   WANDB_PROJECT   (default: ladd-eval)
#   NUM_KID_IMAGES  (default: 1000)
#   NUM_PAIRS       (default: 50)
#   IMAGE_SIZE      (default: 512)
#   STEPS           (default: 4)
#   OUTPUT_DIR      (default: output/eval)

set -euo pipefail

# Source credentials if available
if [[ -f /workspace/.bashrc ]]; then
    source /workspace/.bashrc
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-ladd-eval}"
NUM_KID_IMAGES="${NUM_KID_IMAGES:-1000}"
NUM_PAIRS="${NUM_PAIRS:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
STEPS="${STEPS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval}"
RUN_ID_FILE="$OUTPUT_DIR/wandb_run_id.txt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_checkpoint.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
get_run_id() {
    if [[ -f "$RUN_ID_FILE" ]]; then
        cat "$RUN_ID_FILE"
    fi
}

eval_checkpoint() {
    local checkpoint="$1"
    local val_step="$2"

    local run_id
    run_id="$(get_run_id)"

    local run_id_args=()
    if [[ -n "${run_id:-}" ]]; then
        run_id_args=(--wandb_run_id "$run_id")
        echo "[eval_run] Resuming wandb run: $run_id"
    else
        echo "[eval_run] Starting new wandb run"
    fi

    python "$EVAL_SCRIPT" \
        --checkpoint "$checkpoint" \
        --val_step "$val_step" \
        --wandb_project "$WANDB_PROJECT" \
        --num_kid_images "$NUM_KID_IMAGES" \
        --num_image_pairs "$NUM_PAIRS" \
        --image_size "$IMAGE_SIZE" \
        --num_inference_steps "$STEPS" \
        --output_dir "$OUTPUT_DIR" \
        "${run_id_args[@]}"

    echo "[eval_run] Completed: checkpoint=$checkpoint val_step=$val_step"
}

# ---------------------------------------------------------------------------
# Detect checkpoint format in a directory
# ---------------------------------------------------------------------------
detect_checkpoint() {
    local dir="$1"

    # Prefer consolidated safetensors
    if [[ -f "$dir/consolidated.safetensors" ]]; then
        echo "$dir/consolidated.safetensors"
        return
    fi
    if [[ -f "$dir/model.safetensors" ]]; then
        echo "$dir/model.safetensors"
        return
    fi

    # Accelerate save format
    if [[ -f "$dir/student_transformer/model.safetensors" ]]; then
        echo "$dir"
        return
    fi

    # DCP sharded
    if [[ -f "$dir/.metadata" ]]; then
        echo "$dir"
        return
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# --all mode: scan and evaluate all checkpoints
# ---------------------------------------------------------------------------
eval_all() {
    local train_dir="$1"

    if [[ ! -d "$train_dir" ]]; then
        echo "ERROR: Training directory not found: $train_dir" >&2
        exit 1
    fi

    echo "[eval_run] Scanning for checkpoints in: $train_dir"

    # Collect checkpoint dirs sorted by step number
    local found=0
    for ckpt_dir in $(find "$train_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -t'-' -k2 -n); do
        local step
        step="$(basename "$ckpt_dir" | sed 's/checkpoint-//')"

        local ckpt_path
        ckpt_path="$(detect_checkpoint "$ckpt_dir")"

        if [[ -z "$ckpt_path" ]]; then
            echo "[eval_run] SKIP: no recognized format in $ckpt_dir"
            continue
        fi

        echo "[eval_run] Found checkpoint: $ckpt_path (step $step)"
        eval_checkpoint "$ckpt_path" "$step"
        found=$((found + 1))
    done

    if [[ $found -eq 0 ]]; then
        echo "[eval_run] No checkpoints found in $train_dir"
        exit 1
    fi

    echo "[eval_run] Evaluated $found checkpoints"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo "  bash $0 <checkpoint> <val_step>"
    echo "  bash $0 --all <train_output_dir>"
    exit 1
fi

if [[ "$1" == "--all" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Usage: bash $0 --all <train_output_dir>" >&2
        exit 1
    fi
    eval_all "$2"
else
    if [[ $# -lt 2 ]]; then
        echo "Usage: bash $0 <checkpoint> <val_step>" >&2
        exit 1
    fi
    eval_checkpoint "$1" "$2"
fi
