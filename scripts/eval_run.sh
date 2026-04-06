#!/usr/bin/env bash
# eval_run.sh — Evaluate student checkpoints and build a wandb learning curve.
#
# This script manages a single wandb run across multiple checkpoint evaluations.
# Each invocation appends a new val_step to the same run, producing a learning
# curve (KID vs training step) with logged image pairs at each checkpoint.
#
# Usage:
#   # First run: baseline (teacher weights, val_step=0) — creates wandb run
#   bash scripts/eval_run.sh baseline 0
#
#   # Subsequent runs: pipe in new checkpoints
#   bash scripts/eval_run.sh output/ladd/checkpoint-2000/model.safetensors 2000
#   bash scripts/eval_run.sh output/ladd/checkpoint-4000/model.safetensors 4000
#
#   # Or evaluate all checkpoints at once:
#   bash scripts/eval_run.sh --all output/ladd
#
# Environment variables (optional):
#   WANDB_PROJECT   — wandb project name (default: ladd-eval)
#   WANDB_ENTITY    — wandb entity/team
#   NUM_KID_IMAGES  — images for KID computation (default: 200)
#   NUM_PAIRS       — image pairs logged to table (default: 50)
#   IMAGE_SIZE      — generation resolution (default: 512)
#   STEPS           — student inference steps (default: 4)
#   EVAL_OUTPUT_DIR — where to save images/results (default: output/eval)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="$SCRIPT_DIR/eval_checkpoint.py"

# Defaults
WANDB_PROJECT="${WANDB_PROJECT:-ladd-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
NUM_KID_IMAGES="${NUM_KID_IMAGES:-1000}"
NUM_PAIRS="${NUM_PAIRS:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
STEPS="${STEPS:-4}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-output/eval}"
RUN_ID_FILE="$EVAL_OUTPUT_DIR/wandb_run_id.txt"

eval_checkpoint() {
    local checkpoint="$1"
    local val_step="$2"

    echo "============================================================"
    echo "  Evaluating: checkpoint=$checkpoint  val_step=$val_step"
    echo "============================================================"

    # Build wandb args
    local wandb_args="--wandb_project $WANDB_PROJECT"
    if [[ -n "$WANDB_ENTITY" ]]; then
        wandb_args="$wandb_args --wandb_entity $WANDB_ENTITY"
    fi

    # Resume existing run if we have a run ID
    if [[ -f "$RUN_ID_FILE" ]]; then
        local run_id
        run_id=$(cat "$RUN_ID_FILE")
        wandb_args="$wandb_args --wandb_run_id $run_id"
        echo "  Resuming wandb run: $run_id"
    else
        echo "  Creating new wandb run"
    fi

    python "$EVAL_SCRIPT" \
        --checkpoint "$checkpoint" \
        --val_step "$val_step" \
        --num_kid_images "$NUM_KID_IMAGES" \
        --num_image_pairs "$NUM_PAIRS" \
        --image_size "$IMAGE_SIZE" \
        --num_inference_steps "$STEPS" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        $wandb_args

    echo ""
}

eval_all_checkpoints() {
    local checkpoint_dir="$1"

    # Always start with baseline
    eval_checkpoint "baseline" 0

    # Find all checkpoint directories, extract step numbers, sort numerically
    for ckpt in $(find "$checkpoint_dir" -maxdepth 1 -name "checkpoint-*" -type d | sort -t- -k2 -n); do
        local step
        step=$(basename "$ckpt" | sed 's/checkpoint-//')

        # Prefer consolidated safetensors > accelerate save > DCP dir
        if [[ -f "$ckpt/model.safetensors" ]]; then
            eval_checkpoint "$ckpt/model.safetensors" "$step"
        elif [[ -f "$ckpt/student_transformer/model.safetensors" ]]; then
            eval_checkpoint "$ckpt/student_transformer/model.safetensors" "$step"
        else
            # Pass the directory — eval_checkpoint.py handles DCP consolidation
            eval_checkpoint "$ckpt" "$step"
        fi
    done

    echo "============================================================"
    echo "  All checkpoints evaluated!"
    echo "  wandb run ID: $(cat "$RUN_ID_FILE" 2>/dev/null || echo 'N/A')"
    echo "============================================================"
}

# --- Main ---
if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo "  $0 <checkpoint> <val_step>      Evaluate a single checkpoint"
    echo "  $0 --all <checkpoint_dir>        Evaluate all checkpoints in directory"
    echo ""
    echo "Examples:"
    echo "  $0 baseline 0"
    echo "  $0 output/ladd/checkpoint-2000/model.safetensors 2000"
    echo "  $0 --all output/ladd"
    exit 1
fi

mkdir -p "$EVAL_OUTPUT_DIR"

if [[ "$1" == "--all" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Error: --all requires a checkpoint directory"
        exit 1
    fi
    eval_all_checkpoints "$2"
else
    if [[ $# -lt 2 ]]; then
        echo "Error: need both <checkpoint> and <val_step>"
        exit 1
    fi
    eval_checkpoint "$1" "$2"
fi
