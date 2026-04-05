# FSDP Training Plan for LADD Distillation (8x A100 80GB)

## Context

We have 6 hours of 8x A100 80GB compute. Goal: precompute teacher latents
(50 steps, CFG=5) then train LADD for 20K steps.

**Two-phase approach:**
- **Phase A (~4h)**: Batched precompute of teacher latents across 8 GPUs
- **Phase B (~2h)**: FSDP training with precomputed latents, 20K steps

Current precompute script processes 1 image at a time. Current training uses plain DDP.

---

## Architecture Decisions

### FSDP wrapping strategy

| Model | Params | Trainable | FSDP strategy | Why |
|-------|--------|-----------|---------------|-----|
| Student | 6B | Yes | `FULL_SHARD` | Shards optimizer states (48GB -> 6GB/GPU) |
| Teacher | 6B | No (frozen) | **Not wrapped** -- replicated per GPU | Only does single forwards for disc features. 12GB/GPU is affordable on 80GB A100 |
| Discriminator | 14M | Yes | `FULL_SHARD` (via accelerator.prepare) | Tiny, goes through prepare() alongside student. Overhead negligible |

### Phase A: Batched precompute (~4 hours)

- Modify `data/precompute_teacher_latents.py` to support batch_size=4-8
- 8 GPUs run independently via simple multi-process launcher
- Each GPU processes its shard of prompts (total prompts / 8)
- 50 steps, CFG=5.0, `torch.no_grad()`
- Saves `.pt` files per prompt (existing format)

**Throughput estimate:**
- Single image (current): ~21s/image/GPU
- Batched (batch_size=4): ~8-12s/image/GPU (1.5-2.5x speedup)
- 8 GPUs x 4h x ~300-450 images/GPU/hour = **~9,600-14,400 unique latents**

**VRAM per GPU during precompute:**
- Teacher model: 12 GB (bf16)
- CFG doubles batch: batch_size=4 -> 8 through model
- Activations (no_grad): ~4-8 GB
- Total: ~20-25 GB -- fits easily on 80GB A100

### Phase B: FSDP training (~2 hours)

Existing training loop, unchanged except FSDP wrapping:
1. Load precomputed teacher_x0 from batch
2. Re-noise both sides (existing)
3. Teacher single forward for disc features -> disc update -> gen update
4. At ~2.8 it/s -> 20K steps in ~2h

To hit ~2.8 it/s (up from ~2 it/s), reduce `gradient_accumulation_steps` from 4 to 2.
This halves the micro-batches per optimizer step, giving ~2x wall-clock speedup per step.
Effective batch size drops from 128 to 64 (4 per-GPU × 8 GPUs × 2 grad_accum).
Batch size 64 is still reasonable for adversarial training stability.

### Memory budget per GPU (training)

| Component | Size |
|-----------|------|
| Student (FSDP sharded weights) | ~1.5 GB |
| Student optimizer states (sharded) | ~6 GB |
| Teacher (full replica, bf16) | ~12 GB |
| Discriminator | ~0.03 GB |
| Activations + grad checkpointing | ~4-6 GB |
| **Total** | **~24-26 GB** |

Comfortable margin on 80GB A100.

---

## Files to Modify

### 0. MODIFY: `data/precompute_teacher_latents.py` -- Batched multi-GPU precompute

Current script processes 1 image at a time on 1 GPU. Changes needed:

**0a. Add batched generation + multi-GPU args**

```python
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--rank", type=int, default=0, help="GPU rank (0-7)")
parser.add_argument("--world_size", type=int, default=1, help="Total GPUs")
```

Shard prompts across GPUs: each GPU processes `prompts[rank::world_size]`.
Process in batches of `batch_size` using the same denoising loop from `pipeline.py:generate()`
but returning latents (`output_type="latent"`).

**0b. Add launcher script `scripts/precompute_launch.sh`**

```bash
#!/usr/bin/env bash
# Launch 8 parallel precompute processes, one per GPU
for rank in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$rank python data/precompute_teacher_latents.py \
        --model_dir models/Z-Image \
        --data_meta data/train/metadata_subsample.json \
        --output_dir data/train/teacher_latents \
        --batch_size 4 \
        --rank $rank \
        --world_size 8 \
        --num_inference_steps 50 \
        --guidance_scale 5.0 &
done
wait
```

**0c. Subsample training prompts**

Create `data/train/metadata_subsample.json` with ~10K prompts (stratified sample
from 525K). This is the maximum we can precompute in ~4h. Use existing
`scripts/resplit_data.py` pattern for stratified sampling.
Each latent will be seen ~2x per epoch; over 20K steps with effective batch 64,
we consume 1.28M samples total, so each of ~10K latents is seen ~128 times.

---

### 1. NEW: `training/fsdp_config.yaml` -- Accelerate FSDP config

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: ZImageTransformerBlock
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_offload_params: false
  fsdp_use_orig_params: true
  fsdp_sync_module_states: true
  fsdp_cpu_ram_efficient_loading: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
```

Key choices:
- `TRANSFORMER_BASED_WRAP` with `ZImageTransformerBlock` -- wraps each of the 30 transformer blocks as a separate FSDP unit
- `fsdp_use_orig_params: true` -- required for per-parameter optimizer groups and gradient clipping
- `FULL_STATE_DICT` -- allows saving/loading full checkpoints from rank 0
- `fsdp_sync_module_states: true` -- ensures consistent init across ranks
- `fsdp_cpu_ram_efficient_loading: true` -- loads model once on rank 0, broadcasts

---

### 2. MODIFY: `training/train_ladd.py` -- FSDP compatibility fixes

Training loop logic stays the same. teacher_x0 loaded from precomputed latents
(existing path at line 773-774). Only FSDP-related fixes needed.

**2a. Fix `accelerator.unwrap_model(student)` forward calls (lines 789, 798)**

Under FSDP, forward MUST go through the wrapped module (triggers all-gather):

```python
# BEFORE (line 789):
student_out, _ = accelerator.unwrap_model(student)(...)

# AFTER:
student_out, _ = student(...)
```

Same fix at line 798 (no_grad path).

**2b. Fix checkpoint saving for FSDP (lines 1030-1031, 1079-1093)**

`accelerator.unwrap_model(student).state_dict()` under FSDP returns only the local shard.
Use `accelerator.get_state_dict(student)` which handles FSDP full gathering:

```python
# BEFORE:
unwrapped_student = accelerator.unwrap_model(student)
torch.save(unwrapped_student.state_dict(), ...)

# AFTER:
state_dict = accelerator.get_state_dict(student)
if accelerator.is_main_process:
    torch.save(state_dict, ...)
del state_dict
```

**2c. Fix weight delta tracking for FSDP (lines 954-965)**

Under FSDP with `fsdp_use_orig_params=true`, `named_parameters()` through
`unwrap_model` may return sharded data. Simplest fix: gate behind
`accelerator.distributed_type != "FSDP"` or disable for multi-GPU runs
(it's a debug diagnostic).

---

### 3. MODIFY: `training/train_ladd.sh`

```bash
accelerate launch \
    --config_file training/fsdp_config.yaml \
    training/train_ladd.py \
    --pretrained_model_name_or_path="${MODEL_PATH}" \
    --train_data_meta=data/train/metadata_subsample.json \
    --teacher_latents_dir=data/train/teacher_latents \
    --output_dir="${OUTPUT_DIR}" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=20000 \
    --learning_rate=5e-6 \
    --learning_rate_disc=5e-5 \
    --lr_scheduler=constant_with_warmup \
    --lr_warmup_steps=500 \
    --mixed_precision=bf16 \
    --gradient_checkpointing \
    --allow_tf32 \
    --seed=42 \
    --dataloader_num_workers=4 \
    --checkpointing_steps=2000 \
    --checkpoints_total_limit=3 \
    --validation_steps=2000 \
    --num_inference_steps=4 \
    --image_sample_size=512 \
    --gen_update_interval=3 \
    --disc_layer_indices 5 10 15 20 25 29 \
    --disc_hidden_dim=256 \
    --disc_cond_dim=256 \
    --student_timesteps 1.0 0.75 0.5 0.25 \
    --warmup_schedule_steps=500 \
    --renoise_m=1.0 \
    --renoise_s=1.0 \
    --max_grad_norm=1.0 \
    --report_to=wandb \
    --tracker_project_name=ladd \
    --wandb_entity=yeun-yeungs \
    --validation_prompts \
        "A beautiful sunset over the ocean with golden clouds" \
        "A cat sitting on a windowsill looking outside" \
        "A futuristic city skyline at night with neon lights" \
        "A watercolor painting of a mountain landscape"
```

Changes from current:
- `--config_file training/fsdp_config.yaml` replaces `--multi_gpu --num_processes=8 --mixed_precision=bf16`
- `--train_data_meta` points to subsampled prompts matching precomputed latents
- `--gradient_accumulation_steps=2` (was 4; halved to hit ~2.8 it/s for 20K steps in 2h)
- `--checkpointing_steps=2000` (was 1000; FSDP checkpoint gathering is expensive)

---

## Performance Estimate

**Phase A (precompute):**
- 8 GPUs, batch_size=4, 50 steps CFG=5
- ~8-12s/image -> ~9.6K-14.4K unique latents in 4h
- Each of ~10K latents seen ~128x over 20K steps (effective batch 64)

**Phase B (training):**
- 8 GPUs FSDP, precomputed latents, ~2.8 it/s
- 20K steps in ~2h
- Effective batch size: 4 x 8 x 2 = 64

**Total: ~6 hours**

---

## Verification

### Step 1: Single-GPU smoke tests (done locally)

These validate basic correctness on any machine with 1 GPU:

1. **Create prompt subsample**: `python scripts/subsample_prompts.py --n 10000`
2. **Precompute smoke test** (1 GPU, 4 images, batch_size=2):
   ```bash
   python data/precompute_teacher_latents.py \
       --model_dir models/Z-Image \
       --data_meta /tmp/smoke_meta.json \
       --output_dir /tmp/smoke_latents \
       --batch_size 2
   ```
   Verify: `.pt` files saved, shape `[16, 64, 64]`, re-run to confirm resume skips existing.
3. **Single-GPU training** (1 GPU, 256px, cpu_offload, 10 steps):
   Confirms losses are finite, discriminator is active, forward/backward work.
   (512px OOMs on 1 GPU — expected, confirms FSDP is needed.)

### Step 2: Two-GPU verification (on multi-GPU node)

**Run both of these before the full 8-GPU run.**

#### 2a. Precompute (2 GPUs, 8 images)

```bash
bash scripts/smoke_test_precompute.sh
```

Validates multi-GPU sharding, batched generation, `.pt` file saving, and resume.
Checks: 8 latents saved with shape `[16, 64, 64]`, finite values, resume skips existing.

#### 2b. FSDP training (2 GPUs, 10 steps)

```bash
bash scripts/smoke_test_fsdp.sh
```

This runs `train_ladd.py` with 2-GPU FSDP for 10 steps at 512px. Checks:

1. **No OOM**: 2x A100 80GB should fit student (FSDP sharded) + teacher (replicated) + disc
2. **Losses finite**: `d_loss` and `g_loss` printed in summary should be non-NaN, non-Inf
3. **Checkpoint saved**: verifies `checkpoint-5/student_transformer/pytorch_model.bin` exists
   with full (non-sharded) student weights via `accelerator.get_state_dict()`
4. **FSDP active**: look for `DistributedType.FSDP` in accelerator state log line
5. **Memory**: `peak_vram_mb` in summary should be ~25-30 GB per GPU (not 60GB+)
6. **Gradient flow**: `grad_norm/student` should be non-zero on gen steps (steps 0, 3, 6, 9)

If this passes, the 8-GPU full run is safe to launch.

### Step 3: Full run checks (during 8-GPU training)

7. **Memory check**: `logs["gpu/memory_allocated_gb"]` should be ~24-26 GB per GPU
8. **Gradient flow**: `grad_norm/student` should be non-zero on gen steps
9. **Teacher x0 quality**: Decode a few precomputed latents through VAE, visually inspect

---

## Implementation Order

**Phase 1: FSDP training changes (done)**
1. ~~Create `training/fsdp_config.yaml`~~
2. ~~Fix `train_ladd.py` FSDP issues (unwrap_model, checkpointing, weight tracking)~~
3. ~~Update `training/train_ladd.sh` launch command~~

**Phase 2: Precompute improvements (done)**
4. ~~Create `data/precompute_teacher_latents.py` for batched multi-GPU~~
5. ~~Create `scripts/precompute_launch.sh`~~
6. ~~Create `scripts/subsample_prompts.py` for ~10K stratified prompts~~

**Phase 3: Smoke tests (done locally, pending multi-GPU)**
7. ~~Subsample created: 10K prompts~~
8. ~~Precompute smoke test: 4 latents, shapes correct, resume works~~
9. ~~Single-GPU training: 10 steps at 256px, losses finite~~
10. **2-GPU precompute smoke test**: `bash scripts/smoke_test_precompute.sh` — **run on multi-GPU node**
11. **2-GPU FSDP training smoke test**: `bash scripts/smoke_test_fsdp.sh` — **run on multi-GPU node**

**Phase 4: Full run (on 8x A100 node)**
12. `bash scripts/precompute_launch.sh` (~4h)
13. `bash training/train_ladd.sh` (~2h)
