# Validation & Evaluation Plan

## Overview

LADD training uses two tiers of evaluation metrics:

1. **Per-step cheap metrics** (inline, zero cost) — discriminator health signals logged every training step.
2. **Expensive image-quality metrics** (async subprocess, every 500 steps) — FID and CLIP score computed on a dedicated GPU without blocking training.

The expensive metrics require a one-time **FID reference precomputation** step before training begins.

---

## Pre-requisite: FID Reference Stats

Before training with eval enabled, generate teacher reference images and extract Inception-v3 statistics. This is a **one-time cost**.

### What it does

1. Loads the teacher model (full Z-Image).
2. Generates images for all 13,173 val prompts at 50 inference steps.
3. Passes all images through Inception-v3, extracts 2048-d features.
4. Saves mean vector (mu), covariance matrix (sigma), and sample count to a `.npz` file.

### How to run

```bash
# Val split (used during training every --eval_steps):
python scripts/precompute_fid_reference.py \
    --model_dir <path-to-pretrained-Z-Image> \
    --split val \
    --device cuda:0 \
    --num_inference_steps 50 \
    --image_size 512 \
    --batch_size 4 \
    --seed 42

# Test split (used for final held-out evaluation):
python scripts/precompute_fid_reference.py \
    --model_dir <path-to-pretrained-Z-Image> \
    --split test \
    --device cuda:0 \
    --num_inference_steps 50 \
    --image_size 512 \
    --batch_size 4 \
    --seed 42
```

### Expected runtime

- ~13,173 images at ~1.5s/image (50-step teacher, batch_size=4 on A100) = **~5-6 hours on one GPU**.
- Inception feature extraction adds ~10 minutes.
- Total: **~6 hours on a single A100**.

### Resume support

The script checks for existing images in the output directory and skips already-generated ones. If it crashes midway, just re-run the same command.

To skip generation entirely and only recompute stats from existing images:

```bash
python scripts/precompute_fid_reference.py \
    --model_dir <path-to-pretrained-Z-Image> \
    --split val \
    --skip_generation \
    --image_dir data/val/teacher_images
```

### Output

Per split (`val` or `test`):

- `data/{split}/teacher_images/` — 13,173 PNG images (named `00000.png` to `13172.png`)
- `data/{split}/fid_reference_stats.npz` — NumPy archive with keys `mu` (2048,), `sigma` (2048, 2048), `num_samples` (scalar)

### Dependencies

```bash
pip install "torchmetrics[image]>=1.0.0" open-clip-torch torchvision
```

Or install the training extras:

```bash
pip install -e ".[training]"
```

---

## Training with Eval Enabled

Once `fid_reference_stats.npz` exists, pass it to the training script:

```bash
python -m training.train_ladd \
    --pretrained_model_name_or_path <model_dir> \
    --train_data_meta data/train/metadata.json \
    --val_data_meta data/val/metadata.json \
    --fid_reference_stats data/val/fid_reference_stats.npz \
    --eval_steps 500 \
    --eval_num_images 2048 \
    ...
```

### New CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_steps` | 500 | Run FID/CLIP eval every N steps |
| `--eval_num_images` | 2048 | Number of val images to generate per eval |
| `--val_data_meta` | `data/val/metadata.json` | Path to val split |
| `--fid_reference_stats` | None | Path to `.npz` from precompute step |
| `--eval_device` | auto (last GPU) | GPU for eval subprocess |
| `--skip_expensive_eval` | false | Disable FID/CLIP entirely |

### How async eval works

1. Training saves a checkpoint every `--checkpointing_steps`.
2. At steps divisible by `--eval_steps`, a background subprocess is spawned.
3. The subprocess loads the student from the checkpoint on `--eval_device`.
4. It generates `--eval_num_images` images from a fixed val subset (seed 42).
5. Computes FID against the reference stats and CLIP score.
6. Logs `eval/fid` and `eval/clip_score` to the same wandb run.
7. Saves results to `<output_dir>/eval_results/step_NNNNNN.json`.

If a previous eval is still running when the next one triggers, it is skipped.

---

## Metrics Reference

### Per-step (inline)

| Metric | What to watch for |
|--------|-------------------|
| `disc/accuracy_real` | Should be >50%. If 100% sustained, D dominates — student gets no gradient signal |
| `disc/accuracy_fake` | Should be >50%. If it drops below 50%, D is losing |
| `disc/logit_gap` | Should be positive. Collapsing toward 0 = D can't distinguish real/fake |
| `disc/layer_{idx}_real` | Per-layer mean logit for real samples |
| `disc/layer_{idx}_fake` | Per-layer mean logit for fake samples. Compare with real to see which layers student has matched |

### Every 500 steps (async)

| Metric | What to watch for |
|--------|-------------------|
| `eval/fid` | Lower is better. Measures distance from teacher distribution. Should trend down over training |
| `eval/clip_score` | Higher is better. Measures text-image alignment. Should stay stable or improve |

---

## File Map

```
training/
  ladd_eval.py                        # Eval module (cheap metrics + expensive FID/CLIP)
  train_ladd.py                       # Training loop (imports ladd_eval, launches subprocess)

scripts/
  precompute_fid_reference.py         # One-time teacher reference generation

data/val/
  metadata.json                       # 13,173 val prompts (stratified split)
  fid_reference_stats.npz             # [TO GENERATE] Inception stats from teacher images
  teacher_images/                     # [TO GENERATE] Teacher-generated reference images

data/test/
  metadata.json                       # 13,173 test prompts (stratified split)
  fid_reference_stats.npz             # [TO GENERATE] Inception stats from teacher images
  teacher_images/                     # [TO GENERATE] Teacher-generated reference images
```
