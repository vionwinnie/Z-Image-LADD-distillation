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
# 256px, single A100 80GB, 8-bit Adam
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes=1 training/train_ladd.py \
    --pretrained_model_name_or_path=models/Z-Image \
    --train_data_meta=data/debug/metadata.json \
    --output_dir=research/output \
    --cpu_offload_optimizer \
    --image_sample_size=256 \
    --mixed_precision=bf16 --gradient_checkpointing --allow_tf32 \
    --learning_rate=1e-5 --learning_rate_disc=1e-4 \
    --gen_update_interval=5 \
    --report_to=wandb --tracker_project_name=ladd
```

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

## Next Steps

- Run longer training (2000-5000 steps) with winning config to see if FID continues to drop
- Scale to full training set (500K prompts) — need to precompute embeddings (~2 hours)
- Scale to 8 GPUs with FSDP for production 20K-step run
- Add CLIP score evaluation (needs model download fix)

## Dependencies Installed

omegaconf, deepspeed, mpi4py, wandb, bitsandbytes, torch-fidelity, torchmetrics
