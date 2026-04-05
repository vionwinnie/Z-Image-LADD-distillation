#!/usr/bin/env bash
# Precompute smoke test: 2 GPUs, 8 images, batch_size=2
#
# Validates that multi-GPU sharding, batched generation, resume,
# and .pt file saving all work correctly.
#
# Usage (on a multi-GPU node):
#   bash scripts/smoke_test_precompute.sh
#
# Requirements: at least 2 GPUs

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/Z-Image}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="/tmp/smoke_precompute_$$"
SMOKE_META="/tmp/smoke_precompute_meta_$$.json"

# Create a small metadata file (8 prompts)
python -c "
import json
with open('${SCRIPT_DIR}/data/train/metadata_subsample.json') as f:
    data = json.load(f)
with open('${SMOKE_META}', 'w') as f:
    json.dump(data[:8], f)
print('Created smoke metadata with 8 prompts')
"

echo "============================================"
echo "  Precompute Smoke Test: 2 GPUs, 8 images"
echo "============================================"
echo "  Model:   ${MODEL_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo ""

cleanup() {
    rm -rf "${OUTPUT_DIR}" "${SMOKE_META}"
}
trap cleanup EXIT

# --- Run 1: Generate latents on 2 GPUs ---
echo "--- Run 1: Generate 8 latents across 2 GPUs ---"
for rank in 0 1; do
    CUDA_VISIBLE_DEVICES=$rank python data/precompute_teacher_latents.py \
        --model_dir "${MODEL_DIR}" \
        --data_meta "${SMOKE_META}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size 2 \
        --rank "$rank" \
        --world_size 2 \
        --num_inference_steps 50 \
        --guidance_scale 5.0 &
done
wait

# Verify all 8 latents exist
echo ""
echo "--- Verifying outputs ---"
FAIL=0
python -c "
import torch, os, sys
output_dir = '${OUTPUT_DIR}'
files = sorted(os.listdir(output_dir))
print(f'  Files: {len(files)}')
if len(files) != 8:
    print(f'  [FAIL] Expected 8 files, got {len(files)}')
    sys.exit(1)
for f in files:
    t = torch.load(os.path.join(output_dir, f), weights_only=True)
    print(f'  {f}: shape={t.shape}, dtype={t.dtype}, range=[{t.min():.2f}, {t.max():.2f}]')
    if t.shape != torch.Size([16, 64, 64]):
        print(f'  [FAIL] Expected shape [16, 64, 64], got {t.shape}')
        sys.exit(1)
    if not torch.isfinite(t).all():
        print(f'  [FAIL] Non-finite values in {f}')
        sys.exit(1)
print('  [OK] All 8 latents: correct shape, finite values')
" || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "  [FAIL] Latent verification failed"
    exit 1
fi

# --- Run 2: Resume should skip all ---
echo ""
echo "--- Run 2: Testing resume (should skip all 8) ---"
OUTPUT=$(python data/precompute_teacher_latents.py \
    --model_dir "${MODEL_DIR}" \
    --data_meta "${SMOKE_META}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 2 \
    --rank 0 \
    --world_size 1 2>&1)
echo "$OUTPUT"

if echo "$OUTPUT" | grep -q "0 prompts to process"; then
    echo "  [OK] Resume correctly skipped all existing latents"
else
    echo "  [FAIL] Resume did not skip existing latents"
    exit 1
fi

echo ""
echo "============================================"
echo "  Precompute Smoke Test PASSED"
echo "============================================"
echo "Ready for full 8-GPU precompute run."
