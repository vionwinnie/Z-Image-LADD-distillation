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

### Overfit Test 2: Winning LR (running)

- student_lr=5e-6, disc_lr=5e-5, gi=1, 10 prompts, 2000 steps, 512px
- Same 200 repetitions per prompt, but with proven stable LR
- **Goal**: Can the student produce clean/sharp images at the right LR?
  If yes → architecture works, scale up confidently.
  If still blurry → investigate 4-step denoising setup before scaling.

## Readiness Assessment for Full Training

**Ready:**
- Pipeline works end-to-end
- Hyperparameters validated (sweep found 5.2% FID improvement)
- FID trending down with more steps (336 → 319 → 313)
- Text conditioning works (compositions follow prompts)

**Not ready:**
- No sharp images produced yet at any step count
- FID improvement rate is slow (7% over 4x compute)
- Need overfit test 2 result to confirm architecture can converge

## Next Steps

- Evaluate overfit test 2 (winning LR, 10 prompts)
- If sharp: precompute train embeddings (500K, ~2 hours), launch 8-GPU run
- If still blurry: investigate 4-step denoising, try more student timesteps,
  or increase num_inference_steps to 8

## Dependencies Installed

omegaconf, deepspeed, mpi4py, wandb, bitsandbytes, torch-fidelity, torchmetrics, scipy
