# LADD Training Progress

## Context

LADD (Latent Adversarial Diffusion Distillation) training for Z-Image, a 6.15B parameter
image generation model. Student (6.15B, trainable) + teacher (6.15B, frozen) + discriminator
(14M, trainable) + text encoder + VAE. Production target: 8 GPUs with FSDP.
Single-GPU verification needed first.

## Memory Budget (Single A100 80GB)

| Component                | bf16     | fp32 Adam states |
|--------------------------|----------|------------------|
| Student                  | 12 GB    | 24 GB (regular) / 6 GB (8-bit) |
| Teacher                  | 12 GB    | -                |
| Discriminator            | 0.03 GB  | negligible       |
| Text encoder             | ~3 GB   | -                |
| Activations (grad ckpt)  | ~30 GB (512px) / ~10 GB (256px) | - |
| **Total (regular Adam)** | **~81 GB** | **OOM** |
| **Total (8-bit Adam)**   | **~63 GB (512px)** | fits at 256px |

## What Worked

### smoke_test_train.py (11/11 checks pass)
- Added `--pretrained_model_name_or_path` for real weights (was dummy-only)
- Strategic GPU memory management: free/reload teacher between baseline and LADD steps
- Step 11: inference round-trip (save checkpoint -> load -> generate PIL image -> verify)
- All 11 checks pass with both dummy and real 6.15B weights

### ladd_discriminator.py dtype fix
- Input dtype casting in `forward()` to match parameter dtype
- Needed because hidden states from teacher may arrive in different dtype than disc weights

### inference_ladd.py
- Added wandb logging with `--log_to_wandb` flag
- Results logged as a table: idx | student [| teacher] | prompt
- Added `FlowMatchEulerDiscreteScheduler` import (was missing)
- Verified: 10 images generated at 512x512, 4 steps, ~0.48s each

### Research automation framework (research/)
- `program.md`: Agent instructions for autoresearch-style experiment loop
- `experiment.py`: Hyperparameter config that shells out to train_ladd.py
- `evaluate.py`: FID + CLIP score evaluation (uses pre-computed reference stats)
- `results.tsv`: Experiment log

## What Failed and Why

### Attempt 1: DeepSpeed ZeRO-2 with CPU optimizer offload

**Goal**: Offload student's fp32 Adam states to CPU to save ~24GB GPU memory.

**Problems encountered**:

1. **Dual DeepSpeed engines**: Wrapping both student and discriminator in DeepSpeed
   caused `IndexError: list index out of range` in gradient reduction. DeepSpeed engines
   assume single-model training; the GAN alternating D/G update pattern with
   `retain_graph=True` breaks their internal bookkeeping.

2. **Two-Accelerator pattern**: HuggingFace Accelerate docs suggest using two Accelerator
   instances for multi-model DeepSpeed. Failed with `mpi4py` import errors because the
   second `deepspeed.initialize()` tries to create a new process group.

3. **Single-engine approach** (student only in DeepSpeed, disc as plain PyTorch):
   - `DummyOptim` + `DummyScheduler` resulted in **student LR stuck at 0** for the entire
     run. DeepSpeed's internal WarmupLR scheduler wasn't being configured correctly by
     Accelerate's auto-fill of "auto" values.
   - Student weight norms flat in wandb — confirmed student was not learning.

4. **`discriminator.requires_grad_(False)` during gen step**: Added per DeepSpeed GAN
   tutorial to prevent gradient leakage. But this **broke gradient flow** — the discriminator's
   forward pass with frozen parameters doesn't create a computation graph, so
   `g_loss.backward()` can't trace gradients back through disc -> teacher -> student.
   **Student grad norms were zero.**

5. **`student.backward()` vs `accelerator.backward()`**: Using the DeepSpeed engine's
   backward method caused `AssertionError: gradient computed twice for this partition` because
   both `d_loss.backward()` and `g_loss_update.backward()` triggered ZeRO-2's gradient
   reduction hooks on the student's parameters.

6. **Grad norm logging**: Grad norms were captured **after** `optimizer.zero_grad()`,
   always showing 0. With DeepSpeed, even capturing before zero_grad showed 0 because
   ZeRO-2 manages gradients in internal buffers, not on `param.grad`.

7. **Checkpoint save failures**: `PytorchStreamWriter failed writing file data/2` when
   saving the ~24GB optimizer state file. Likely a filesystem limitation with very large
   single files.

8. **`os.execv` + `tee` incompatibility**: The experiment runner used `os.execv` to replace
   the Python process (avoiding GPU memory leak from fork), but this broke the `tee` pipe
   so output wasn't captured.

9. **`os.system` / `subprocess` GPU memory leak**: Parent Python process holds GPU memory
   even without importing torch, because `fork()` duplicates the process memory space.
   Child process then OOMs because parent holds ~36GB.

**Conclusion**: DeepSpeed ZeRO-2 is designed for single-model training. The GAN setup
with alternating D/G updates, cross-model gradient flow, and two optimizers is fundamentally
incompatible with DeepSpeed's assumptions on a single GPU.

### Attempt 2: 8-bit Adam (bitsandbytes)

**Goal**: Use 8-bit Adam to reduce optimizer states from ~24GB to ~6GB. No DeepSpeed.

**Result**: Partially successful.
- LR warmup works correctly (0 -> 1e-5 over 50 steps)
- Student grad norms are **non-zero** (gradients flowing!)
- Discriminator training works
- **OOM at 512px** on gen steps (activation memory for student + teacher forward with
  grad graph exceeds remaining GPU memory)
- **Works at 256px** -- all 50 steps complete, losses finite, grad norms non-zero

**Current status**: 256px training works. 512px needs either multi-GPU or further
memory optimization (e.g., offloading text encoder during gen steps).

## Current Working Configuration

```bash
# 512px, single A100 80GB, 8-bit Adam, precomputed embeddings
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes=1 training/train_ladd.py \
    --pretrained_model_name_or_path=models/Z-Image \
    --train_data_meta=data/debug/metadata.json \
    --embeddings_dir=data/debug/embeddings \
    --output_dir=research/output \
    --cpu_offload_optimizer --skip_save \
    --image_sample_size=512 \
    --mixed_precision=bf16 --gradient_checkpointing --allow_tf32 \
    --learning_rate=5e-6 --learning_rate_disc=5e-5 \
    --gen_update_interval=3 \
    --report_to=wandb --tracker_project_name=ladd
```

Key: `--embeddings_dir` skips the ~3GB text encoder, enabling 512px on a single GPU.
`--skip_save` saves only student weights as safetensors (~12GB), not full optimizer state.

## Validated (baseline-256px-delta-v2)

Run: https://wandb.ai/yeun-yeungs/ladd/runs/952u5wzr

- 50 steps at 256px, 8-bit Adam, `--skip_save`, single A100 80GB
- **LR warmup**: 0 -> 1e-5 over 10 steps (working correctly)
- **Student grad norms**: non-zero on gen steps (every 5th step)
- **Weight deltas**: gradually increasing to 0.006 by step 50 (student is learning)
- **d_loss**: oscillating 0-4 (discriminator active)
- **g_loss**: oscillating -3 to +3 (healthy adversarial dynamics)
- Completed in 21 seconds, clean exit, no OOM, no NFS hang

## Hyperparameter Sweep Results (8 experiments, ~2 hours)

All runs: 500 steps, 512px, debug split (98 prompts), single A100 80GB.
FID computed via torch-fidelity (50 student vs 50 teacher images).

| student_lr | disc_lr | gen_update_interval | FID | Status |
|-----------|---------|---------------------|-----|--------|
| 1e-5 | 1e-4 | 5 | 336.31 | baseline |
| 2e-5 | 1e-4 | 5 | 364.39 | discard |
| 5e-6 | 1e-4 | 5 | 332.52 | - |
| 1e-6 | 1e-4 | 5 | 333.14 | discard |
| 5e-6 | 5e-5 | 5 | 330.67 | - |
| 5e-6 | 2e-4 | 5 | 332.59 | discard |
| 5e-6 | 5e-5 | 1 | 364.11 | discard |
| **5e-6** | **5e-5** | **3** | **318.55** | **winner** |

**Best config: student_lr=5e-6, disc_lr=5e-5, gen_update_interval=3**

Key findings:
- Lower LRs help (5e-6 > 1e-5 > 2e-5) — conservative updates are better early
- 10x disc/student ratio is optimal (same as LADD paper)
- gi=3 is the sweet spot — more frequent gen updates than default (5) but not every step (1)
- FID improved 336 -> 319 (5.2% improvement from baseline)

## Scale-up Validation (2000 steps, best config)

FID with best config over training steps:

| Steps | FID | Notes |
|-------|-----|-------|
| 500 | 318.55 | sweep winner |
| 2000 | 313.19 | still improving |

FID is dropping — confirms the student **can learn**. However FID 313 is still very high
(good distilled models target FID < 30). Visual inspection of 2000-step outputs shows
coarse structure (spatial composition learned) but significant noise artifacts.

This is expected for:
- Only 2000 steps (LADD paper uses 50K-200K)
- 98 prompts (tiny dataset)
- batch_size=1 (very noisy gradients)

## Precomputed Embeddings

Text encoder embeddings precomputed offline to save ~3GB GPU memory during training.
This was the key optimization that unlocked 512px training on a single A100 80GB.

| Split | Prompts | Time | Size | Path |
|-------|---------|------|------|------|
| Debug | 98 | 3s | 106 MB | data/debug/embeddings/ |
| Debug (10) | 10 | 0.6s | 7 MB | data/debug/embeddings_10/ |
| Val | 13K | 6 min | 8.6 GB | data/val/embeddings/ |
| Train | 500K | ~2 hours | ~330 GB | not yet computed |

### Single GPU vs Cluster: precompute or not?

**Single GPU (current)**: Precomputed embeddings are essential. The text encoder
takes ~3GB which is the difference between OOM and fitting at 512px. Worth the
precompute time and disk cost.

**8-GPU cluster with FSDP**: Precomputed embeddings are NOT needed. Each GPU has
~47GB used / 80GB total, leaving ~33GB headroom. The text encoder (3GB) fits
easily. Running it live adds ~30ms per step (~2-3% overhead on a 1-2s step).
Not worth 2 hours precompute + 330GB disk for 2-3% speedup.

## FID Evaluation

- Reference stats precomputed from 13K teacher images: `data/val/fid_reference_stats.npz` (33 MB)
- Student Inception features extracted via `torch-fidelity` (GPU-native, fast)
- FID computed against reference (mu, sigma) using scipy `sqrtm`
- `torchmetrics.FrechetInceptionDistance` abandoned — runs on CPU, takes 10+ minutes for 50 images

## Current Phase: Overfit Test

Running 10 prompts x 2000 steps at 512px with aggressive settings:
- student_lr=1e-4 (20x higher than sweep winner)
- disc_lr=1e-3 (20x higher)
- gen_update_interval=1 (gen update every step)
- text_drop_ratio=0.0 (no CFG dropout)
- Each prompt seen ~200 times

**Goal**: Determine if the student can produce recognizable images when given
maximum learning pressure on a tiny dataset.

### Overfit Test 1: Aggressive LR (FAILED)

- student_lr=1e-4, disc_lr=1e-3, gi=1, 10 prompts, 2000 steps, 512px
- Each prompt seen ~200 times
- **Result: pure noise.** The 20x higher LR caused divergence.
- W&B: `overfit-10prompts-512px-lr1e-4-gi1`

**Lesson learned**: High LR + GAN = unstable. The discriminator at 1e-3 likely
overwhelmed the student, or the student weights diverged from aggressive updates.
This is consistent with our sweep finding that lower LRs are better.

### Overfit Test 2: Winning LR (COMPLETED)

- student_lr=5e-6, disc_lr=5e-5, gi=1, 10 prompts, 2000 steps, 512px
- W&B: `overfit-10prompts-winning-lr-gi1`, eval: `overfit2-winning-lr-eval`
- **Result: semantically correct but blue color shift, worse than 98-prompt run.**
- Images are recognizable and match prompts, but quality degraded vs more diverse data.

**Key insight**: With only 10 prompts + bs=1, gradients are too noisy and biased.
The student oscillates between pleasing individual prompts instead of learning
general features. The blue color shift is mode collapse toward a "safe" mean palette.
The 98-prompt run produced better images because gradient directions average out
over more diverse data.

**Conclusion: the architecture works, the bottleneck is compute/data scale.**

## Readiness Assessment for Full Training: READY

**Evidence that the architecture works:**
1. Text conditioning works — compositions match prompts (98-prompt run)
2. More data = better images (98 prompts > 10 prompts)
3. More steps = better FID (336 → 319 → 313)
4. Gradient flow confirmed (weight deltas growing, grad norms non-zero)
5. Discriminator is active (d_loss oscillating, not collapsed)

**Why single-GPU results are limited:**
- batch_size=1 → extremely noisy gradients
- 98 prompts → not enough diversity for generalization
- 2000 steps → barely started (LADD paper uses 50K-200K)

**What 8-GPU cluster provides:**
- Effective batch size 32 (4 per GPU × 8 GPUs) → 32x less gradient noise
- 500K training prompts → full diversity
- 20K optimizer steps → 10x more training
- FSDP shards memory → no need for 8-bit Adam or precomputed embeddings

## KID Hyperparameter Sweep (autoresearch/apr5, 2026-04-05)

Switched from FID to KID (unbiased for small samples). All runs on single A100 80GB,
debug split (98 prompts), bs=1, 512px. Early stopping disabled (broken at bs=1).
Inline validation generates 1000 images for KID computation.

### Calibration Runs (establishing KID baselines)

| Config | Steps | KID | Notes |
|--------|-------|-----|-------|
| slr=1e-5 dlr=1e-4 gi=5 | 500 | 0.008502 | original baseline (was FID 336.31) |
| slr=5e-6 dlr=5e-5 gi=3 | 500 | 0.008037 | previous best (was FID 313.19) |
| slr=5e-6 dlr=5e-5 gi=3 | 2000 | 0.007229 | longer training helps modestly |

### Noise Schedule Exploration

| RENOISE_M | RENOISE_S | KID | Status |
|-----------|-----------|-----|--------|
| 1.0 (default) | 1.0 | 0.008037 | baseline |
| 0.0 | 1.0 | 0.005505 | better — lower noise bias helps |
| **0.5** | 1.0 | **0.004605** | **best M** |
| -0.5 | 1.0 | 0.005084 | worse, disc_acc_fake=0 |
| 0.5 | 0.5 | 0.005644 | tighter spread worse |
| 0.5 | 1.5 | 0.005590 | wider spread worse |

**Finding**: M=0.5 (sigmoid ≈ 0.62, moderate noise) is optimal. The default M=1.0
was too conservative (high noise makes real/fake hard to distinguish). S=1.0 is
the sweet spot — neither tighter nor wider spread helped.

### GEN_UPDATE_INTERVAL Exploration

| GI | KID | Status |
|----|-----|--------|
| 2 | 0.011020 | much worse (too frequent gen updates) |
| 3 | 0.008037 | previous best |
| 4 | 0.002409 | good |
| 6 | 0.002015 | better |
| **8** | **0.000869** | **best — 89% better than original baseline** |
| 10 | 0.012895 | too few gen steps, disc too powerful |

**Finding**: GI=8 is a major win. The discriminator needs many more steps per gen
update than we thought. GI=3 was far from optimal. The sweet spot is 8 — at 10
the discriminator becomes too powerful and overwhelms the student.

### Simplification Wins

- `LR_WARMUP_STEPS = 0` (was 50): removing warmup had no negative effect
- `WARMUP_SCHEDULE_STEPS = 0` (was 50): removing timestep warmup also fine

### Current Best Config (KID = 0.000869)

```python
STUDENT_LR = 5e-6
DISC_LR = 5e-5
GEN_UPDATE_INTERVAL = 8
RENOISE_M = 0.5
RENOISE_S = 1.0
LR_WARMUP_STEPS = 0
WARMUP_SCHEDULE_STEPS = 0
DISC_HIDDEN_DIM = 256
DISC_LAYER_INDICES = [5, 10, 15, 20, 25, 29]
STUDENT_TIMESTEPS = [1.0, 0.75, 0.5, 0.25]
```

### Still Unexplored
- Discriminator architecture: DISC_HIDDEN_DIM (128, 512), DISC_LAYER_INDICES
- Whether GI=8 shifts optimal LRs or noise schedule
- Interaction effects (e.g. re-tuning M with GI=8)

## Teacher Image Quality Debug (2026-04-05)

Validation teacher images looked blurry. Root cause investigation:

### Bug 1: Scheduler linspace off-by-one

Our custom `FlowMatchEulerDiscreteScheduler.set_timesteps()` used:
```python
timesteps = np.linspace(sigma_max_t, sigma_min_t, num_inference_steps + 1)[:-1]
```
Diffusers uses:
```python
timesteps = np.linspace(sigma_max_t, sigma_min_t, num_inference_steps)
```

**Impact:** With 50 steps, our scheduler's smallest sigma was 0.109 (11% noise remaining).
Diffusers reaches sigma=0.0 (fully denoised). The student was being evaluated against
blurry teacher images that still had residual noise.

**Fix:** Removed `+ 1` and `[:-1]` in `src/zimage/scheduler.py`. Verified timesteps and
sigmas now match diffusers exactly (`torch.allclose` = True).

### Bug 2: Teacher images generated without CFG

`precompute_fid_reference.py` generated teacher images with `guidance_scale=0`.
The official Z-Image usage example uses `guidance_scale=4`. The non-distilled teacher
model requires CFG for sharp, well-composed images — without it, outputs are
unconditional-like and lack detail.

Additionally, `precompute_fid_reference.py` defaults to `--teacache_thresh=0.5`,
which skips ~75% of transformer layer computations for speed. This may further
degrade quality.

**Impact on KID:** All KID measurements to date compared student images against
blurry teacher references. Absolute KID values are unreliable. Relative ordering
between experiments should still hold (same reference for all), but the baseline
quality bar was set too low.

**CFG sweep result:** CFG=5.0 selected after visual comparison on wandb.
Teacher images regenerated with corrected scheduler + CFG=5 + TeaCache disabled.
W&B: `debug-teacher-cfg-sweep` (run mg6h6a9f)

### Bug 3: Student input is pure noise regardless of timestep

The student at timestep t should receive `x_t = (1-t)*teacher_x0 + t*ε`, not
pure noise. At t=1.0 this is pure noise (correct), but at t=0.75/0.5/0.25 the
student needs to see partial teacher structure to learn the remaining denoising.

Without this fix, the student at t=0.25 sees pure noise but is told "you're
almost done" — it has no signal about what to reconstruct.

### Bug 4: No velocity-to-latent conversion

The student predicts velocity v (flow matching formulation), but the code
used raw velocity as the denoised prediction. The correct conversion is:
`x̂_0 = x_t - t * v`

### Bug 5: "Real" discriminator samples are random noise

The LADD paper (Section 3.2) says the "real" distribution for the discriminator
should be teacher-generated images with CFG. Our code used `add_noise(noise1,
noise2, t_hat)` — random noise mixed with random noise, providing no meaningful
"real" signal for the discriminator.

**Fix:** `add_noise(teacher_x0, noise, t_hat)` where teacher_x0 is precomputed
offline with CFG=5, 50 steps. This requires a new precompute step:
`data/precompute_teacher_latents.py` saves raw latents (before VAE decode)
as .pt files (~262KB each).

### Corrected Training Flow (all 5 bugs fixed)

```
OFFLINE:
  teacher_x0 = teacher.generate(prompt, cfg=5, steps=50, output_type="latent")

ONLINE per step:
  1. x_t = (1-t) * teacher_x0 + t * ε           ← student input (Bug 3)
  2. v = student(x_t, t, prompt)                  ← velocity prediction
  3. x̂_0 = x_t - t * v                           ← denoised latent (Bug 4)
  4. fake_noisy = (1-t̂) * x̂_0 + t̂ * ε₁           ← re-noise for disc
  5. real_noisy = (1-t̂) * teacher_x0 + t̂ * ε₂     ← real path (Bug 5)
  6. teacher(fake_noisy) → features → disc → "fake"
  7. teacher(real_noisy) → features → disc → "real"
```

### Files Changed

- `src/zimage/scheduler.py` — linspace fix (Bug 1)
- `training/train_ladd.py` — student input, velocity conversion, real path (Bugs 3-5)
- `training/ladd_utils.py` — TextDataset loads teacher latents
- `data/precompute_teacher_latents.py` (new) — generates teacher latents with CFG
- `data/regenerate_teacher_images.py` (new) — regenerates teacher images with CFG
- `scripts/smoke_test_train.py` — scheduler + CFG checks (Steps 11-12)
- `scripts/smoke_test_proposed.py` (new) — 9 tests validating corrected flow

### Impact on Previous Results

All KID measurements were against blurry teacher references (broken scheduler +
no CFG + TeaCache). The training loop had wrong student inputs, no velocity
conversion, and meaningless "real" samples. **All previous numeric results are
invalid.** Relative ordering between hyperparameter configs may still hold since
all used the same broken setup, but absolute quality was severely limited.

The best hyperparameters from the sweep (GI=8, M=0.5, S=1.0, slr=5e-6,
dlr=5e-5) are being re-validated with the corrected pipeline.

### First Corrected Run (v2, 500 steps)

Config: slr=5e-6, dlr=5e-5, GI=8, M=0.5, S=1.0, debug split (98 prompts),
512px, single A100 80GB, precomputed teacher latents (CFG=5, 50 steps).

- W&B training: https://wandb.ai/yeun-yeungs/ladd/runs/au7vsbzy
- W&B eval: https://wandb.ai/yeun-yeungs/ladd/runs/idj1oc8i
- 500 steps completed, peak VRAM 68.4 GB
- Final d_loss=0.0, g_loss=1.75

**Observation:** d_loss collapses to 0 — discriminator perfectly classifies
real vs fake. With proper teacher latents as "real", the adversarial signal
is much stronger than before (noise-vs-noise was trivial). The hyperparameters
tuned against the broken pipeline likely need re-tuning:
- GI=8 may be too many disc steps now (disc too strong)
- disc_lr=5e-5 may need to be lower
- student_lr=5e-6 may need to be higher to keep up

### Precomputed Data

| Split | Type | Count | Size | Path | Status |
|-------|------|-------|------|------|--------|
| Debug | Teacher latents (.pt) | 98 | 25.8 MB | data/debug/teacher_latents/ | Done |
| Debug | Embeddings | 98 | 106 MB | data/debug/embeddings/ | Done |
| Val | Teacher images (CFG=5) | 130/1000 | partial | data/val/teacher_images/ | Interrupted |
| Val | Teacher latents | 0 | - | data/val/teacher_latents/ | Not started |

## Next Steps

1. Re-tune hyperparameters with corrected pipeline (disc LR, GI, student LR)
2. Complete val teacher image regeneration (for KID eval)
3. Precompute val teacher latents (for val-time training metrics)
4. Precompute train split teacher latents (~2.5 GB for 10K prompts)
5. Launch longer run (2000+ steps) once hyperparameters stabilized
6. Update `train_ladd.sh` with validated hyperparameters
7. Launch 8-GPU production run

## Dependencies Installed

omegaconf, deepspeed, mpi4py, wandb, bitsandbytes, torch-fidelity, torchmetrics, scipy
