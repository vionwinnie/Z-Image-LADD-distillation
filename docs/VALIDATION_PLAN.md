# Validation & Evaluation Plan

## Overview

LADD training uses two tiers of evaluation metrics:

1. **Per-step cheap metrics** (inline, zero cost) — discriminator health signals logged every training step.
2. **KID evaluation** (post-training or periodic) — Kernel Inception Distance computed between student and teacher images on the same prompts.

**Why KID over FID:** FID requires thousands of samples (5K+) for a stable covariance estimate over 2048-d Inception features. With 1000 val prompts, FID has high variance. KID is an unbiased estimator using kernel MMD — reliable even at small sample sizes.

---

## Data Splits

Resplit from original 13K val/test to balanced 1K each (stratified by subject/style/camera). Leftovers added to train.

| Split | Prompts | Purpose |
|-------|---------|---------|
| Train | 524,915 | Training data |
| Val | 1,000 | KID eval during/after training |
| Test | 1,000 | Final held-out evaluation |
| Debug | 98 | Single-GPU quick experiments |

---

## KID Evaluation

### How it works

1. Generate N images from student checkpoint on val prompts.
2. Generate N teacher reference images on the same prompts (cached after first run).
3. Pass both image directories to `torch-fidelity` with `kid=True`.
4. Reports `kid_mean ± kid_std` (lower is better; 0 = identical distributions).

### Running evaluation

```bash
# After training:
python research/evaluate.py --checkpoint research/output/checkpoint-2000

# With cached teacher images:
python research/evaluate.py \
    --checkpoint research/output/checkpoint-2000 \
    --teacher_image_dir data/val/teacher_images

# Fewer images for quick check:
python research/evaluate.py \
    --checkpoint research/output/checkpoint-2000 \
    --num_images 50
```

### KID parameters

- `kid_subset_size=100` (or num_images if smaller) — samples per subset
- `kid_subsets=100` — number of bootstrap subsets for mean/std
- Polynomial kernel (degree 3) — default from torch-fidelity

### Expected runtime (single A100)

| Component | 50 images | 1000 images |
|-----------|-----------|-------------|
| Student generation | ~24s | ~8 min |
| Teacher generation (first time only) | ~24s | ~8 min |
| KID computation (Inception + kernel) | ~5s | ~30s |
| **Total (first run)** | **~50s** | **~17 min** |
| **Total (cached teacher)** | **~30s** | **~9 min** |

On 8-GPU cluster, generation can be parallelized (~1 min for 1000 images).

---

## Training with Periodic Eval

For cluster training, KID eval can run as an async subprocess:

```bash
python -m training.train_ladd \
    --pretrained_model_name_or_path <model_dir> \
    --train_data_meta data/train/metadata.json \
    --val_data_meta data/val/metadata.json \
    --eval_steps 2000 \
    --eval_num_images 1000 \
    ...
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_steps` | 500 | Run KID eval every N steps |
| `--eval_num_images` | 2048 | Number of val images to generate per eval |
| `--val_data_meta` | `data/val/metadata.json` | Path to val split (1000 prompts) |
| `--eval_device` | auto (last GPU) | GPU for eval subprocess |
| `--skip_expensive_eval` | false | Disable KID eval entirely |

---

## Per-Step Metrics Reference (inline, free)

| Metric | What to watch for |
|--------|-------------------|
| `disc/accuracy_real` | Should be >50%. If 100% sustained, D dominates |
| `disc/accuracy_fake` | Should be >50%. If <50%, D is losing |
| `disc/logit_gap` | Should be positive. Collapsing to 0 = D can't distinguish |

---

## File Map

```
research/
  evaluate.py                         # KID eval (student vs teacher images)

training/
  ladd_eval.py                        # Image generation + metrics helpers
  train_ladd.py                       # Training loop (launches eval subprocess)

scripts/
  resplit_data.py                     # Resplit val/test to 1K balanced samples

data/val/
  metadata.json                       # 1,000 val prompts (stratified)
  teacher_images/                     # [CACHED] Teacher reference images

data/test/
  metadata.json                       # 1,000 test prompts (stratified)
  teacher_images/                     # [CACHED] Teacher reference images
```
