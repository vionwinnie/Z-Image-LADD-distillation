# LADD Distillation of Z-Image — Analysis & Implementation Plan

## Table of Contents

- [Codebase Analysis](#codebase-analysis)
- [LADD Paper Summary](#ladd-paper-summary)
- [Key Differences: LADD vs Existing Distillation Code](#key-differences-ladd-vs-existing-distillation-code)
- [Ambiguities & Open Questions](#ambiguities--open-questions)
- [Proposed Implementation Plan](#proposed-implementation-plan)
- [References & Adapted Code](#references--adapted-code)

---

## Codebase Analysis

### Z-Image Architecture (from `Z-Image` repo)

Z-Image is a **DiT (Diffusion Transformer)** operating in latent space with flow matching:

| Component | Details |
|-----------|---------|
| **Backbone** | `ZImageTransformer2DModel` — 30 layers, dim=3840, 30 heads |
| **Modulation** | AdaLN with tanh gating, `ADALN_EMBED_DIM=256` |
| **Normalization** | RMSNorm throughout |
| **Position encoding** | 3-axis RoPE (dims=[32,48,48], theta=256) |
| **Patch embedding** | 2×2 spatial patches, 16 input channels |
| **Text encoder** | Qwen3-based causal LM, extracts `hidden_states[-2]` |
| **VAE** | AutoencoderKL, 16 latent channels, kept in fp32 |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` with dynamic shifting |
| **Inference** | 8 steps, guidance_scale=0.0 (turbo model already exists) |

**Architecture flow** (`transformer.py:474-571`):
1. Patchify input latents → linear embed
2. Noise refiner (2 blocks with AdaLN, processes latent tokens)
3. Context refiner (2 blocks without modulation, processes text tokens)
4. Concatenate latent + text tokens
5. 30 main transformer blocks (joint attention over unified sequence)
6. Final layer → unpatchify

**Key detail**: The transformer takes `x` as a **list of tensors** (variable-length sequences per batch item), not a fixed-size batch tensor. This is important for multi-aspect-ratio support.

### VideoX-Fun Training Infrastructure

The training code lives in `VideoX-Fun/scripts/z_image/`. Key files:

| File | Purpose |
|------|---------|
| `train.py` | Baseline flow matching training |
| `train_distill.py` | **DMD-style distillation (existing, ~1900 lines)** |
| `train_distill_lora.py` | LoRA variant of distillation |
| `train_lora.py` | LoRA fine-tuning |
| `train_distill.sh` | Launch script for distillation |

**Training framework**: HuggingFace Accelerate with optional DeepSpeed Zero / FSDP.

**Data loading**: `TextDataset` from `videox_fun/data/dataset_image_video.py` — loads a JSON of text prompts. Format:
```json
[{"text": "a photo of a cat"}, {"text": "sunset over mountains"}, ...]
```
No images are loaded for distillation — latents are generated from noise on-the-fly.

### Existing `train_distill.py` — DMD Implementation

The existing distillation script uses **three copies of the transformer**:

| Role | Variable | Trainable | Purpose |
|------|----------|-----------|---------|
| Student/Generator | `generator_transformer3d` | Yes | Learns to denoise in fewer steps |
| Teacher | `real_score_transformer3d` | Frozen | Provides "real" score |
| Discriminator/Critic | `fake_score_transformer3d` | Yes | Provides "fake" score |

**Training loop** (`train_distill.py:1637-1890`):

1. **Generator update** (every `gen_update_interval` steps):
   - Sample noise, multi-step denoise with student
   - Re-noise the result at a random timestep
   - Get fake score (discriminator prediction) and real score (teacher prediction)
   - DMD loss = normalized MSE between student output and (student output - score difference)
   - Backprop through student

2. **Critic update** (every step):
   - Generate samples with student (no grad)
   - Re-noise, feed to discriminator
   - Denoising loss = MSE between discriminator prediction and student's clean prediction
   - Backprop through discriminator

**Key hyperparameters** from `train_distill.sh`:
- `learning_rate=2e-05` (student), `learning_rate_critic=2e-06` (discriminator)
- `denoising_step_indices_list=[1000, 875, 750, 625, 500, 375, 250, 125]` (8 steps)
- `max_grad_norm=0.05`
- `gradient_checkpointing` enabled
- `trainable_modules "."` (all modules trainable)
- `image_sample_size=1328`

---

## LADD Paper Summary

**Paper**: [Latent Adversarial Diffusion Distillation](https://arxiv.org/abs/2403.12015) (Sauer et al., Stability AI, 2024)

### Core Idea

LADD improves upon ADD (Adversarial Diffusion Distillation) by:
1. Operating entirely in **latent space** (no pixel-space decoding)
2. Using **generative features** from the frozen teacher as discriminator backbone (not DINOv2)
3. Unifying teacher and discriminator feature extractor into one model
4. Using only **adversarial loss** (no distillation loss when training on synthetic data)

### Architecture

```
[Text Prompt] → Teacher (frozen, multi-step + CFG) → Synthetic clean latent x₀
                                                        ↓
                                               Add noise at t ∈ {1, 0.75, 0.5, 0.25}
                                                        ↓
                                              Student (trainable) → x̂₀ (denoised)
                                                        ↓
                                               Re-noise at t̂ ~ LogitNormal(1, 1)
                                                        ↓
                                        Frozen Teacher forward pass (extract features)
                                                        ↓
                                         Discriminator Heads (per attention block)
                                                        ↓
                                              Adversarial hinge loss
```

### Discriminator Design

- **NOT a full model** — just lightweight heads on frozen teacher features
- One independent head per attention block in the teacher
- Token sequences are **reshaped to 2D spatial layout** before processing with 2D convolutions
- Conditioned on: noise level `t̂` + pooled CLIP/text embeddings
- Follows Projected GAN paradigm (StyleGAN-XL / ADD)

### Loss Functions

| Loss | When Used | Details |
|------|-----------|---------|
| **L_adv (Adversarial)** | Always | Hinge loss via discriminator heads |
| **L_distill (Distillation)** | Only with real data | MSE between student and teacher denoised outputs |
| **DPO (optional)** | Post-training | LoRA rank-256, 3k iterations |

**When training on synthetic data (recommended): only adversarial loss is used.**

### Noise Schedules

**Student input timesteps** (4-step model):
- Discrete set: `t ∈ {1, 0.75, 0.5, 0.25}`
- Warm-up (first 500 high-res iterations): `p = [0, 0, 0.5, 0.5]`
- Main phase: `p = [0.7, 0.1, 0.1, 0.1]` (biased toward t=1 for single-step quality)

**Re-noising for discriminator**:
- `t̂ ~ LogitNormal(m=1, s=1)`
- High noise → discriminator checks global structure
- Low noise → discriminator checks local texture/detail

### Flow Matching Formulation

```
x_t = (1 - t) * x₀ + t * ε,  where t ∈ [0, 1]
```

### Key Results

- SD3-Turbo (8B): Matches full SD3 quality in **4 unguided steps**
- Single-step generation already outperforms all single-step baselines
- Student size has the largest impact on performance (teacher can be smaller)
- Training: 10k iterations for ablation, final model trained longer

---

## Key Differences: LADD vs Existing Distillation Code

| Aspect | Existing `train_distill.py` (DMD) | LADD Paper |
|--------|-----------------------------------|------------|
| **Discriminator architecture** | Full 3rd copy of transformer (~15B params) | Lightweight conv heads on frozen teacher features |
| **Loss function** | DMD loss (normalized score difference MSE) | Adversarial hinge loss only |
| **Memory footprint** | 3× full transformer | 1× student + 1× teacher + tiny disc. heads |
| **Training data** | Text prompts → noise-to-image | Teacher generates synthetic latents from prompts |
| **Re-noising distribution** | Uniform or discrete sampling | LogitNormal(m=1, s=1) |
| **Student timesteps** | 8 evenly spaced (1000→125) | 4 discrete {1, 0.75, 0.5, 0.25} with learned probabilities |
| **Discriminator features** | Full model output (single prediction) | Per-block features (multi-scale) |
| **CFG at inference** | Not clear | Unguided (no CFG needed) |

### Memory Estimation for 8×A100 (80GB each)

**Existing DMD approach**:
- 3× transformer in bf16: ~90GB (models alone)
- Optimizer states for 2 trainable models: ~60GB
- Activations + gradients: significant
- **Very tight on 8×A100** — requires aggressive FSDP sharding

**LADD approach**:
- 1× student (trainable) in bf16: ~30GB
- 1× teacher (frozen) in bf16: ~30GB
- Discriminator heads: <1GB
- Optimizer states for student only: ~30GB
- **Much more comfortable on 8×A100**

---

## Ambiguities & Open Questions

### 1. Implementation Strategy

The existing `train_distill.py` provides valuable infrastructure (Accelerate setup, data loading, checkpointing, distributed training) but implements DMD, not LADD. Options:

- **(A) Modify in-place**: Add `--use_ladd` flag to existing `train_distill.py`
  - Pro: Single file, easy toggling, reuse all infra
  - Con: File is already 1900 lines, mixing two methods is messy

- **(B) Clean implementation from scratch**: Build on Z-Image inference repo
  - Pro: Clean, minimal, easy to understand
  - Con: Must rewrite all training infra (Accelerate, FSDP, checkpointing)

- **(C) Fork and substantially modify** (recommended):
  - Fork `train_distill.py` → `train_ladd.py`
  - Keep infrastructure (Accelerate, data loading, checkpointing)
  - Replace 3rd transformer with lightweight discriminator heads
  - Replace DMD loss with adversarial hinge loss
  - Pro: Clean separation, reuse infra, easy to explain differences
  - Con: Some code duplication with original

### 2. Data Pipeline — What Counts as "Data"?

The assignment says: *"Source and prepare a dataset suitable for LADD distillation"* and *"images, latent targets, and any conditioning signals."*

**However**, LADD training uses **text prompts only**. The teacher generates synthetic latents on-the-fly. No pre-existing images are needed.

**Options**:
- **Prompt-only dataset**: Curate diverse text prompts, formatted as JSON for `TextDataset`
- **Image+prompt dataset**: Pre-encode images through VAE for a reconstruction-style approach
- **Hybrid**: Prompts for distillation, plus a small image set for validation/comparison

**Recommendation**: Prompt-only is correct for LADD, but we should document clearly why no images are needed, and include sample images for evaluation.

**Potential prompt sources**:
- [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) — 14M real user prompts
- [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) — captions from high-aesthetic images
- [JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB) — Midjourney prompts
- Custom curated set for diversity (people, landscapes, objects, abstract, text rendering)

### 3. Discriminator Head Architecture — Specifics Unclear

The paper says heads follow StyleGAN-XL / ADD projected discriminators with 2D convolutions instead of 1D. But exact architecture details are sparse:

- How many conv layers per head?
- What's the conditioning mechanism for noise level + text embeddings?
- What's the spatial resolution after reshaping tokens to 2D?
- Do all 30 attention blocks get a head, or a subset?

**Approach**: Reference the [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl) and [ADD](https://arxiv.org/abs/2311.17042) code for discriminator head design, adapt to Z-Image's token dimensions.

### 4. Z-Image "Turbo" Already Exists

The Z-Image repo ships `Z-Image-Turbo` which runs at 8 steps with `guidance_scale=0.0`. This appears to already be a distilled model (possibly via the existing DMD approach). Our LADD distillation would be:
- A re-distillation demonstrating the LADD method specifically
- Potentially targeting fewer steps (4 steps, matching the paper)
- A comparison point: LADD 4-step vs existing Turbo 8-step

### 5. Teacher Feature Extraction — Requires Model Modification

LADD needs intermediate features from the teacher's attention blocks. The current `ZImageTransformer2DModel.forward()` only returns the final output. We need to:
- Add a `return_intermediate_features=True` flag
- Collect token sequences after each of the 30 attention blocks
- This is a non-trivial modification to `transformer.py`

### 6. Token-to-2D Reshaping for Discriminator

Z-Image uses variable-length token sequences (concatenated text + image tokens, padded to multiples of 32). To reshape to 2D for conv heads:
- Must separate image tokens from text tokens
- Image tokens follow a known spatial layout: `(F_tokens, H_tokens, W_tokens)`
- Size info is available from `patchify_and_embed` return values

### 7. Flow Matching Formulation Alignment

Z-Image's scheduler uses `timesteps = (1000 - t) / 1000` normalization (mapping 0→1000 to 1→0). LADD paper uses `t ∈ [0, 1]` directly. Need to carefully align these conventions:
- Z-Image: `noisy = (1 - sigma) * x0 + sigma * noise`, where sigma derived from timestep
- LADD: `x_t = (1 - t) * x0 + t * epsilon`
- These are the same formulation but with different variable naming

### 8. Multi-GPU Memory Strategy

With 8×A100 (80GB each):
- Student (~30GB bf16) + Teacher (~30GB bf16) + disc heads (<1GB) = ~61GB model weights
- Need FSDP to shard student + optimizer states across GPUs
- Teacher can be loaded with `requires_grad=False` and potentially in lower precision
- Gradient checkpointing essential for student
- Batch size likely 1 per GPU with gradient accumulation

---

## Proposed Implementation Plan

### Phase 1: Data Pipeline (Day 1-2)

**Goal**: Curate a prompt dataset formatted for `TextDataset`.

1. Source prompts from DiffusionDB or similar open dataset
2. Filter for quality, diversity, and English language
3. Create `metadata.json`:
   ```json
   [
     {"text": "A serene mountain landscape at sunset with snow-capped peaks"},
     {"text": "Portrait of a young woman with curly hair, studio lighting"},
     ...
   ]
   ```
4. Create splits:
   - `debug/metadata.json` — 100 prompts for smoke testing
   - `train/metadata.json` — 10K+ prompts for full training
5. Write preprocessing/filtering script
6. Document data sourcing decisions in `data/README.md`

### Phase 2: LADD Architecture Integration (Day 2-4)

**Goal**: Implement LADD as `train_ladd.py`, forked from `train_distill.py`.

#### 2a. Teacher Feature Extraction
- Modify `ZImageTransformer2DModel.forward()` to optionally return intermediate features
- Add `return_hidden_states=True` parameter
- Collect features after each of the 30 main transformer blocks

#### 2b. Discriminator Heads
- Implement `LADDDiscriminatorHead` module:
  - Reshape token sequence to 2D spatial layout
  - 2D conv layers (following Projected GAN design)
  - Conditioned on noise level + pooled text embeddings
  - Output: real/fake logit per spatial position
- Implement `LADDDiscriminator` that wraps multiple heads (one per block, or a subset)

#### 2c. Training Loop
- Fork `train_distill.py` → `train_ladd.py`
- Remove `fake_score_transformer3d` (no 3rd model)
- Add discriminator heads initialization
- Implement:
  - Logit-normal re-noising distribution
  - 4-step student timesteps with probability scheduling
  - Adversarial hinge loss
  - Warm-up phase (500 steps, restricted timesteps)
- Add `--use_ladd` flag for clear toggling
- Separate optimizers for student and discriminator heads

#### 2d. Key Files to Create/Modify
| File | Action |
|------|--------|
| `train_ladd.py` | New — forked from `train_distill.py` |
| `ladd_discriminator.py` | New — discriminator head architecture |
| `transformer.py` | Modified — add intermediate feature extraction |
| `train_ladd.sh` | New — launch script |

### Phase 3: Smoke Testing (Day 3-5)

**Goal**: Validate the full pipeline on CPU/single GPU.

1. `smoke_test.py`:
   - Forward pass with tiny batch (1-2 samples)
   - Verify tensor shapes through entire pipeline
   - Check gradient flow: student params, discriminator heads
   - Verify teacher is frozen (no gradients)
   - Loss computation (adversarial loss values reasonable)
   - Checkpoint save/load round-trip

2. Logging:
   - Student loss, discriminator loss per block
   - Gradient norms (student, each discriminator head)
   - GPU memory usage
   - Learning rate schedule

3. Debug checklist:
   - [ ] Tensor shapes match at every stage
   - [ ] dtypes correct (bf16 for models, fp32 for loss)
   - [ ] Teacher features extracted from correct layers
   - [ ] 2D reshaping preserves spatial layout
   - [ ] Logit-normal distribution produces valid timesteps
   - [ ] Gradient only flows through student and disc heads
   - [ ] Checkpoint save/load preserves all state

### Phase 4: Full Training on 8×A100 (Day 5-6)

**Goal**: 20,000 steps of LADD distillation.

Launch command (draft):
```bash
accelerate launch \
  --num_processes=8 \
  --mixed_precision=bf16 \
  scripts/z_image/train_ladd.py \
  --pretrained_model_name_or_path=models/Z-Image \
  --train_data_meta=data/train/metadata.json \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=20000 \
  --learning_rate=2e-05 \
  --learning_rate_disc=1e-04 \
  --lr_scheduler=constant_with_warmup \
  --lr_warmup_steps=500 \
  --checkpointing_steps=1000 \
  --gradient_checkpointing \
  --mixed_precision=bf16 \
  --image_sample_size=1024 \
  --seed=42 \
  --output_dir=output_ladd \
  --report_to=tensorboard
```

### Phase 5: Inference Demo (Day 6-7)

**Goal**: Compare student (4-step) vs teacher (50-step).

1. `inference.py`:
   - Load student checkpoint
   - Generate images from representative prompts
   - Side-by-side comparison with teacher
   - Measure latency per image

2. Evaluation prompts (diverse set):
   - People / portraits
   - Landscapes / nature
   - Objects / products
   - Abstract / artistic
   - Text rendering
   - Complex multi-element scenes

---

## References & Adapted Code

| Source | What We Use | License |
|--------|-------------|---------|
| [Z-Image](https://github.com/Tongyi-MAI/Z-Image) | Model architecture, inference pipeline | Apache-2.0 |
| [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) | Training infrastructure, data loading, distributed setup | Apache-2.0 |
| [LADD Paper](https://arxiv.org/abs/2403.12015) | Method — discriminator design, loss functions, noise schedules | — |
| [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl) | Reference for projected discriminator head architecture | — |
| [ADD Paper](https://arxiv.org/abs/2311.17042) | Reference for adversarial diffusion distillation | — |
