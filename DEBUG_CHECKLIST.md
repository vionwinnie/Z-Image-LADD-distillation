# Debug Checklist

## smoke_test_train.py

Single-GPU test validating the full LADD training pipeline end-to-end.

```bash
# With real weights:
python scripts/smoke_test_train.py --pretrained_model_name_or_path=models/Z-Image

# With dummy weights (no model download needed):
python scripts/smoke_test_train.py --dummy

# With precomputed embeddings (skips text encoder):
python scripts/smoke_test_train.py --pretrained_model_name_or_path=models/Z-Image \
    --embeddings_dir=data/train/embeddings_subsample
```

### Checks

| # | Check | What it validates |
|---|-------|-------------------|
| 1 | Model creation | Student, teacher, VAE, text encoder (or precomputed embeddings), discriminator all load |
| 2 | encode_prompt | Text encoder produces variable-length embeddings, or precomputed embeddings load correctly |
| 3 | calculate_shift | Dynamic shift computation returns correct float for given image_seq_len |
| 4 | sample_student_timestep | Warmup regime samples from last 2 timesteps; post-warmup favors t=1.0 |
| 5 | Baseline MSE training | Student forward → MSE loss → backward → optimizer step updates weights |
| 6 | LADD forward + loss | Student → re-noise → teacher (no_grad) → discriminator → hinge loss is finite |
| 7 | Disc gradient flow | d_loss.backward() gives disc non-zero grads; teacher stays frozen; disc weights update |
| 8 | Student gradient flow | student → teacher (WITH grad) → disc → g_loss.backward() gives student grads; teacher params stay grad-free |
| 9 | Checkpoint round-trip | Save student state_dict → reload into fresh model → parameters match exactly |
| 10 | D/G update schedule | 20 steps with gen_update_interval=5 → 20 disc updates, 4 gen updates |
| 11 | Scheduler matches diffusers | FlowMatchEulerDiscreteScheduler timesteps/sigmas match diffusers reference |
| 12 | CFG parameter | pipeline.generate accepts guidance_scale parameter |

## smoke_test_fsdp.sh

Multi-GPU (2×GPU) test validating FSDP wrapping with the real training script.

```bash
bash scripts/smoke_test_fsdp.sh
```

### Checks

| # | Check | What it validates |
|---|-------|-------------------|
| 1 | FSDP wrapping | Student wrapped with TRANSFORMER_BASED_WRAP on ZImageTransformerBlock |
| 2 | Sharding | FULL_SHARD strategy distributes params across 2 GPUs |
| 3 | Mixed precision | bf16 forward/backward with fp32 optimizer states |
| 4 | Forward pass | FSDP-wrapped student produces correct outputs across ranks |
| 5 | Loss computation | d_loss and g_loss are finite on all ranks |
| 6 | Gradient sync | Gradients synchronized across ranks during backward |
| 7 | Optimizer step | Student weights update correctly under FSDP |
| 8 | Checkpoint save | DCP sharded checkpoint saved with .metadata file |
| 9 | Validation | Image generation works with FSDP-wrapped student (all ranks participate) |
| 10 | 10 steps complete | Full 10-step training loop completes without hang or crash |

### FSDP Config

```yaml
fsdp_sharding_strategy: FULL_SHARD
fsdp_state_dict_type: SHARDED_STATE_DICT
fsdp_offload_params: false  # true causes index_select device mismatch with embeddings
fsdp_sync_module_states: false  # true causes NaN from meta-device initialization
fsdp_cpu_ram_efficient_loading: false  # must match sync_module_states
```

## Known Issues & Resolutions

1. **NaN from `fsdp_cpu_ram_efficient_loading: true`** — non-rank-0 processes create student on meta device with improperly materialized weights. Fix: keep `false`.
2. **FSDP CPU offload device mismatch** — `fsdp_offload_params: true` moves embedding weights to CPU, but `index_select` needs index and weights on same device. Fix: keep `false`, or ensure no text encoder is loaded.
3. **Activation checkpointing breaks `return_hidden_states`** — checkpoint wrapper invalidates hidden state references during backward recomputation, causing NaN. Fix: do not apply activation checkpointing to models that collect intermediate hidden states.
4. **Teacher on GPU** — teacher must be a plain model (no FSDP wrapping). FSDP wrapping the teacher caused NaN from param reshard invalidating hidden state tensors.
5. **Validation hangs without text encoder** — when using `--embeddings_dir`, validation must use precomputed val embeddings via `generate_from_embeddings`, not `pipeline.generate` which requires a text encoder.
6. **Token ordering** — Z-Image concatenates image tokens first, text tokens second. The discriminator's `_extract_image_features()` relies on this via `x_item_seqlens`.
7. **Variable-length sequences** — Z-Image processes inputs as lists of variable-length tensors. `student_input_list` is constructed via `noise.unsqueeze(2).unbind(dim=0)`.

## Pre-Training Sanity Checks

- [ ] d_loss starts near ~2.0 (random init) and trends toward ~1.0 (equilibrium)
- [ ] g_loss oscillates but doesn't diverge to large negative values
- [ ] Neither loss goes to NaN
- [ ] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set to reduce fragmentation
- [ ] Precomputed embeddings count matches dataset count (e.g., 10000 for subsample)
- [ ] CLIP embeddings count matches dataset count
