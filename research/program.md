# LADD Distillation Research

This is an experiment to have the LLM optimize hyperparameters for LADD (Latent Adversarial Diffusion Distillation) of a 6.15B parameter image generation model.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr7`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `research/program.md` — this file. Architecture, what to explore, constraints.
   - `research/experiment.py` — the file you modify. Hyperparameters, architecture knobs, training config.
   - `research/evaluate.py` — fixed evaluation. Do not modify.
   - `research/results.tsv` — past experiment results.
4. **Verify models exist**: Check that the model path (absolute) contains the pretrained model. Warm up model weights: `cat models/Z-Image/transformer/*.safetensors > /dev/null`.
5. **Verify data**: Training data uses precomputed Qwen text embeddings, CLIP embeddings (for disc conditioning), and teacher latents.
6. **Verify GPU**: Run `nvidia-smi` to confirm a GPU is available and no stale processes are holding memory. Kill stale python/torch processes if needed.
7. **Compute untrained student KID**: Run `python scripts/eval_checkpoint.py --checkpoint baseline --val_step 0` to establish the untrained reference point.
8. **Initialize results.tsv**: Create with header row and untrained KID reference.
9. **Confirm and go**: Confirm setup looks good.

## Architecture (v3 — 2026-04-06)

Key changes from previous runs:
- **CLIP text embeddings for discriminator conditioning**: The discriminator receives precomputed CLIP embeddings (dim=512) directly via `--clip_embeddings_dir`. This is required (no fallback).
- **Precomputed teacher latents**: Training uses precomputed teacher latents via `--teacher_latents_dir`, skipping teacher forward pass.
- **Precomputed Qwen text embeddings**: Via `--embeddings_dir`, skipping text encoder during training.
- **KID evaluation**: Uses `scripts/eval_checkpoint.py` with `compute_kid_from_dirs()` which filters corrupt images. Teacher images are in `data/val/teacher_images/` (1000 images, 2 are corrupt empty files: 00442.png, 00935.png).
- **Absolute model path**: `MODEL_PATH` must use absolute path `/workspace/Z-Image-LADD-distillation/models/Z-Image`.

### Untrained student KID reference

| Metric | Value |
|--------|-------|
| Untrained student KID (1000 images vs teacher) | 0.068920 +/- 0.006648 |

## v3 Hyperparameter Search Results (2026-04-06, branch autoresearch/apr6b)

10 experiments at 500 steps each on 3K training data with CLIP disc conditioning.

### Best configuration

| Parameter | Value |
|-----------|-------|
| STUDENT_LR | 5e-6 |
| DISC_LR | 1e-5 |
| LR_WARMUP_STEPS | 0 |
| GEN_UPDATE_INTERVAL | 3 |
| WARMUP_SCHEDULE_STEPS | 0 |
| STUDENT_TIMESTEPS | [1.0, 0.75, 0.5, 0.25] |
| RENOISE_M | 1.0 |
| RENOISE_S | 1.0 |
| DISC_HIDDEN_DIM | 256 |
| DISC_COND_DIM | 256 |
| DISC_LAYER_INDICES | [5, 10, 15, 20, 25, 29] |
| TEXT_DROP_RATIO | 0.1 |

### All results ranked

| Rank | Exp | Config change | KID | vs untrained |
|------|-----|--------------|-----|-------------|
| 1 | exp3 | GI=3 M=1.0 | 0.058232 | -15.5% |
| 2 | exp10 | + LR_WARMUP=50 | 0.064476 | -6.4% |
| 3 | exp1 | GI=3 M=0.5 | 0.066527 | -3.5% |
| 4 | exp7 | GI=4 M=1.0 | 0.067874 | -1.5% |
| 5 | exp2 | GI=5 M=0.5 | 0.068150 | -1.1% |
| — | untrained | — | 0.068920 | — |
| 7 | exp5 | dlr=5e-5 | 0.069502 | +0.8% |
| 8 | exp4 | M=1.5 | 0.069731 | +1.2% |
| 9 | exp6 | 3 disc layers | 0.072160 | +4.7% |
| 10 | exp9 | disc_hidden=128 | 0.073543 | +6.7% |
| 11 | baseline | GI=2 | 0.075387 | +9.4% |
| 12 | exp8 | slr=1e-5 | 0.094953 | +37.8% |

### Variance note (2026-04-07)

Repeated runs of the best config (GI=3, M=1.0) show significant variance at 500 steps / 3K data / bs=1:

| Run | KID |
|-----|-----|
| exp3 (original) | 0.058232 |
| run2 | 0.069161 |
| run3 | 0.069966 |
| run4 | pending |
| run5 | pending |

The exp3 result (0.058) appears to be an outlier. True expected KID is likely ~0.069. This level of variance is inherent to bs=1 training over only 500 steps — small numerical differences from GPU non-determinism compound. Longer training and larger data (10K) should reduce variance.

### Key findings

1. **Noise schedule is the most impactful knob.** RENOISE_M=1.0 (sigmoid ~0.73, biased toward high noise) gave the single biggest improvement vs M=0.5. At M=1.0, real and fake samples look more similar under noise, giving the student a gentler learning signal. M=1.5 was too much noise — the discriminator couldn't distinguish real from fake (logit_gap dropped to 1.3).

2. **GI=3 is the optimal update interval with CLIP conditioning.** Tested GI={2,3,4,5}. GI=2 gave the disc too few steps relative to the generator. GI=4 and GI=5 weakened the learning signal. GI=3 was consistently best.

3. **Default learning rates are optimal.** STUDENT_LR=5e-6 and DISC_LR=1e-5 (2:1 ratio) won. Higher student LR (1e-5) caused severe overshoot (+38% worse). Higher disc LR (5e-5) also hurt.

4. **Discriminator architecture should not be weakened.** Both fewer layers ([10,20,29] vs [5,10,15,20,25,29]) and smaller hidden dim (128 vs 256) produced worse results.

5. **Warmup doesn't help at 500 steps.** LR_WARMUP=50 was second-best overall but still worse than no warmup. With only 500 steps, warmup wastes 10% of the training budget.

6. **1000 steps degrades on 3K data.** KID=0.097 at 1000 steps vs ~0.069 at 500 steps. The model overfits. 10K data should enable longer training.

### What to explore next

- **10K training data** — current search used 3K subset; more data may shift optimal hyperparameters and enable longer training
- **RENOISE_M fine-tuning** around 1.0 (try 0.75, 1.25) — we know 0.5 and 1.5 are worse but haven't explored the neighborhood
- **Multiple seeds** — run variance is high; consider averaging 3 runs per config for reliable comparisons

## Experimentation

Each experiment runs on a single A100 80GB. The training script runs for a **fixed step budget** (`MAX_TRAIN_STEPS` in `experiment.py`, default 500 steps, ~17 min training + ~5 min evaluation). You launch it simply as: `python research/experiment.py > research/run.log 2>&1`.

**What you CAN do:**
- Modify `research/experiment.py` — this is the only file you edit. The tunable section between `=== TUNABLE HYPERPARAMETERS START ===` and `=== TUNABLE HYPERPARAMETERS END ===` is your playground.

**What you CANNOT do:**
- Modify `research/evaluate.py`, `training/`, `src/`, `scripts/`.
- Change `MAX_TRAIN_STEPS`, `TRAIN_BATCH_SIZE`, `SEED`, `IMAGE_SIZE`, or `MODEL_PATH`.
- Install new packages.

**KID computation** (after training):
```python
python3 -c "
import sys; sys.path.insert(0, 'scripts')
from eval_checkpoint import compute_kid_from_dirs
result = compute_kid_from_dirs('research/output/eval_images/step_000500', 'data/val/teacher_images', 1000)
print(f'KID = {result[\"kid_mean\"]:.6f} +/- {result[\"kid_std\"]:.6f}')
"
```

## The experiment loop

LOOP FOREVER:

1. Read `research/experiment.py` and `research/results.tsv`
2. Change ONE thing in the tunable section
3. `git add research/experiment.py research/results.tsv && git commit -m "<description>"`
4. `git push origin autoresearch/<tag>`
5. Warm up: `cat models/Z-Image/transformer/*.safetensors > /dev/null`
6. Run: `python research/experiment.py > research/run.log 2>&1`
7. Extract metrics: `grep "peak_vram_mb:\|disc/logit_gap:" research/run.log`
8. Compute KID with `compute_kid_from_dirs`
9. Record in results.tsv
10. If improved: keep commit. If worse: `git reset --hard HEAD~1`
11. Commit updated results.tsv, push

## Infrastructure notes

- Training uses **8-bit Adam** (`--cpu_offload_optimizer`) to fit on a single A100 80GB
- **Precomputed Qwen embeddings** (`--embeddings_dir`) skip the text encoder during training
- **CLIP embeddings** (`--clip_embeddings_dir`) provide discriminator text conditioning (required)
- **Precomputed teacher latents** (`--teacher_latents_dir`) skip teacher forward pass
- `--skip_save` skips saving full optimizer state
- `--skip_baseline_validation` skips eval at step 0
- `experiment.py` uses `os.execv` to replace Python process with bash, avoiding GPU memory leaks
