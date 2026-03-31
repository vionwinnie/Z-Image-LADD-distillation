# LADD Distillation of Z-Image: Training Guide

This guide explains how LADD (Latent Adversarial Diffusion Distillation) works at every level — from the math to the code — so you can walk through the full pipeline confidently.

---

## Table of Contents

1. [What LADD Does](#1-what-ladd-does)
2. [How Z-Image Works (The Teacher)](#2-how-z-image-works-the-teacher)
3. [LADD Architecture: Three Components](#3-ladd-architecture-three-components)
4. [The Training Loop Step by Step](#4-the-training-loop-step-by-step)
5. [Gradient Flow: What Gets Updated and Why](#5-gradient-flow-what-gets-updated-and-why)
6. [Training Stability](#6-training-stability)
7. [Code Map](#7-code-map)
8. [Data Pipeline](#8-data-pipeline)
9. [Running Training](#9-running-training)
10. [Monitoring and Debugging](#10-monitoring-and-debugging)
11. [Inference with the Distilled Student](#11-inference-with-the-distilled-student)

---

## 1. What LADD Does

Z-Image generates images in 28–50 denoising steps. Each step is a full forward pass through a 6B-parameter transformer. LADD trains a **student** model (same architecture, initialized from the same weights) to produce comparable quality in **4 steps** — a ~12x speedup.

The key idea: use the pretrained teacher model itself as a discriminator backbone, training entirely in latent space. No pixel-space decoding, no external feature extractors (like DINOv2), no real images needed.

### Why not simpler distillation?

| Method | Problem |
|--------|---------|
| **Progressive Distillation** (match teacher's 2-step output in 1 step) | Quality degrades below 4 steps; blurry outputs |
| **Consistency Models** (self-consistency constraint) | No perceptual sharpness signal; outputs lack detail |
| **ADD** (adversarial + DINOv2 discriminator in pixel space) | Must decode to RGB → huge VRAM; capped at 518×518 |
| **LADD** (adversarial in latent space, teacher as discriminator) | Solves all of the above |

---

## 2. How Z-Image Works (The Teacher)

Z-Image is a **Scalable Single-Stream Diffusion Transformer (S3-DiT)** trained with **flow matching**.

### Architecture at a glance

```
Text prompt → Qwen3 LLM → text embeddings (2560-dim)
                              ↓ cap_embedder (Linear 2560→3840)
                              ↓
Image → VAE Encoder → latent (16-ch, 64×64) → Patchify (2×2) → image tokens
                              ↓
              Context Refiner (2 layers, text self-attention)
              Noise Refiner   (2 layers, image self-attention + adaLN)
                              ↓
              Concatenate: [image tokens | text tokens] → single sequence
                              ↓
              30 × S3-DiT Block:
                  adaLN (timestep-conditioned scale + gate)
                  Self-Attention (30 heads × 128 dim, 3D RoPE, QK-Norm)
                  SwiGLU FFN (3840 → 10240 → 3840)
                              ↓
              Final Layer → Unpatchify → predicted velocity
```

**Key numbers:**
- 6B parameters, hidden dim 3840, 30 layers
- VAE: 8× spatial downscale, 16 latent channels
- With 2×2 patching: a 1024×1024 image → 1024 image tokens

### Flow matching formulation

The forward process is a linear interpolation:

```
x_t = (1 - t) · x_0 + t · ε       where t ∈ [0, 1], ε ~ N(0, I)
```

The model predicts velocity `v = ε - x_0`, and the denoised output is:

```
x̂_0 = x_t - t · v_θ(x_t, t)
```

At inference, Z-Image runs 28–50 Euler steps with classifier-free guidance (scale 3.0–5.0).

---

## 3. LADD Architecture: Three Components

### 3.1 Student Model

- Same `ZImageTransformer2DModel` architecture as the teacher
- Initialized from the teacher's pretrained weights
- **Trainable** — this is what we're optimizing
- Learns to denoise in 1–4 steps instead of 28–50

### 3.2 Teacher Model (dual role)

The teacher is a **frozen** copy of the pretrained Z-Image model. It serves two roles at **different times**:

**Role A — Feature Extractor (during each training step):**
The teacher receives a re-noised latent and runs a single forward pass. We don't use its final output — instead we extract **intermediate hidden states** after each of the 30 transformer blocks. These features are fed to the discriminator heads.

This works because the teacher was trained to denoise images, so its internal representations encode rich knowledge about image structure at every noise level.

**Note: the teacher is NOT used as a data generator in our implementation.** In the original LADD paper (applied to SD3), the teacher generates synthetic latents offline via multi-step sampling with CFG. In our implementation, the "real" samples are constructed directly from fresh Gaussian noise at the discriminator's noise level `t_hat` — there are no pre-generated latents or real images. The teacher's output distribution at any noise level implicitly defines what "real" looks like through its features.

### 3.3 Discriminator

Lightweight 2D convolutional heads attached at 6 of the teacher's 30 layers (indices `[5, 10, 15, 20, 25, 29]`).

Each head:
1. Receives image tokens from that layer — shape `(B, num_image_tokens, 3840)`
2. Projects down: `Linear(3840 → 256)`
3. FiLM conditioning: `t_hat` embedding + pooled text embedding → scale/shift modulation
4. 2D conv blocks: `Conv(256→256, 3×3) → GroupNorm → SiLU → Conv(256→128, 3×3) → GroupNorm → SiLU`
5. Output: `Conv(128→1, 1×1) → mean pool → scalar logit`

**Why 2D convolutions?** The image tokens have a spatial layout (e.g., 32×32 for 1024×1024 images). 2D convolutions preserve spatial structure and support variable aspect ratios. ADD used 1D convolutions on flattened sequences, which conflates spatial dimensions.

**Why multiple layers?** Different transformer layers capture different abstraction levels:
- Early layers (5, 10): textures, local patterns
- Middle layers (15, 20): object composition, relationships
- Late layers (25, 29): semantics, prompt alignment

The student can't "game" all scales simultaneously.

---

## 4. The Training Loop Step by Step

Here is exactly what happens each iteration (from `training/train_ladd.py`):

### Step 1: Encode text prompts

```python
cap_feats = encode_prompt(tokenizer, text_encoder, prompts, max_seq_len=512)
# → list of tensors, each (seq_len_i, 2560), variable length per prompt
```

Uses Qwen3's chat template, extracts second-to-last hidden state.

### Step 2: Sample student timestep

```python
# Discrete set: t ∈ {1.0, 0.75, 0.5, 0.25}
# Warm-up (steps 0–500): p = [0, 0, 0.5, 0.5]  ← easy tasks first
# Main (steps 500+):    p = [0.7, 0.1, 0.1, 0.1] ← focus on hardest (t=1.0)
```

`t=1.0` means pure noise (hardest), `t=0.25` means light noise (easiest). The warm-up schedule prevents early instability.

### Step 3: Create noisy student input

```python
noise = torch.randn(B, 16, H, W)  # pure Gaussian noise
student_input = add_noise(torch.zeros_like(noise), noise, sigma_t)
# For t=1.0: student_input ≈ pure noise (student must generate from scratch)
# For t=0.25: student_input = 0.75·zeros + 0.25·noise (mild noise)
```

### Step 4: Student denoises

```python
student_pred = student(student_input, t, cap_feats)
# → predicted clean latent x̂_0, shape (B, 16, H, W)
```

### Step 5: Re-noise for discrimination

Both the student's output and a "real" sample are re-noised at a fresh timestep `t_hat`:

```python
t_hat = logit_normal_sample(B, m=1.0, s=1.0)  # LogitNormal(1, 1)
fresh_noise = torch.randn_like(student_pred)

# Fake path (from student):
student_renoised = (1 - t_hat) · student_pred.detach() + t_hat · fresh_noise

# Real path (from fresh noise):
real_noisy = (1 - t_hat) · torch.zeros_like(noise) + t_hat · fresh_noise
# Equivalent to: t_hat · fresh_noise (since "real" clean latent is zeros)
```

**Why re-noise?** Two reasons:
1. The teacher expects noised inputs — it was trained on them
2. `t_hat` controls feedback granularity: high `t_hat` = structural feedback, low `t_hat` = textural feedback

### Step 6: Extract teacher features

```python
# Fake path — graph connected (gradients flow through teacher to student)
_, fake_extras = teacher(student_renoised, t_hat, cap_feats, return_hidden_states=True)
fake_hidden_states = fake_extras["hidden_states"]  # list of 30 tensors

# Real path — detached (no gradient needed)
with torch.no_grad():
    _, real_extras = teacher(real_noisy, t_hat, cap_feats, return_hidden_states=True)
    real_hidden_states = real_extras["hidden_states"]
```

Each `hidden_states[l]` has shape `(B, seq_len, 3840)` — the full token sequence (image + text) after transformer block `l`.

### Step 7: Discriminator classifies

```python
fake_result = discriminator(fake_hidden_states, t_hat, x_item_seqlens, cap_item_seqlens)
real_result = discriminator(real_hidden_states, t_hat, x_item_seqlens, cap_item_seqlens)
# Each result contains {"logits": {layer_idx: (B,)}, "total_logit": (B,)}
```

The discriminator extracts image-only tokens (image tokens come first in the concatenated sequence), reshapes them to 2D, and runs the conv heads.

### Step 8: Compute losses and update

```python
d_loss, g_loss = LADDDiscriminator.compute_loss(real_result, fake_result)
# Hinge loss:
#   d_loss = mean(relu(1 - real_logits)) + mean(relu(1 + fake_logits))
#   g_loss = -mean(fake_logits)
```

**Discriminator updates every step:**
```python
d_loss.backward()
clip_grad_norm_(discriminator.parameters(), max_grad_norm)
optimizer_disc.step()
```

**Student updates every `gen_update_interval` steps (default 5):**
```python
# Fresh forward pass with gradients flowing through student → teacher → discriminator
g_loss_update.backward()  # gradients: D → teacher ops → student weights
clip_grad_norm_(student.parameters(), max_grad_norm)
optimizer_student.step()
```

---

## 5. Gradient Flow: What Gets Updated and Why

This is the critical part to understand. The gradient chain for the **student update**:

```
Student weights (θ)           ← UPDATED
       ↓  (forward pass)
  v_θ(x_t, t)                ← student's velocity prediction
       ↓  (differentiable)
  x̂₀ = x_t - t·v_θ          ← student's denoised output
       ↓  (linear, differentiable)
  x̂_t̂ = (1-t̂)·x̂₀ + t̂·ε'   ← re-noised student output
       ↓  (forward pass, graph connected)
  Teacher(x̂_t̂, t̂)           ← frozen weights, but computation graph flows through
       ↓  (slicing)
  features at layers [5,10,15,20,25,29]
       ↓  (conv + linear)
  Discriminator heads          ← UPDATED (separate optimizer step)
       ↓
  g_loss = -mean(fake_logits) ← scalar loss
       ↓  (backprop through entire chain)
  ∂g_loss/∂θ                  ← gradient reaches student
```

### Key distinction: frozen ≠ detached

```python
# FROZEN: weights won't update, but gradients flow through the operations
teacher.requires_grad_(False)
features = teacher(x_hat_t_hat)  # graph connected, grads flow through

# DETACHED: no gradients at all — graph is severed
with torch.no_grad():
    features = teacher(x_hat_t_hat)  # graph disconnected
```

For the **fake** path, the teacher is frozen but NOT detached — gradients flow through its matrix multiplications, attention, and FFN all the way back to the student's output. The teacher's weights are never updated, but the computational graph through its operations provides the gradient pathway.

For the **real** path, the teacher IS detached (`torch.no_grad()`) — no gradients needed since there's nothing to update upstream.

### What gets updated each step

| Component | Updated? | How often | Optimizer |
|-----------|:--------:|:---------:|-----------|
| Student (6B params) | Yes | Every `gen_update_interval` steps (default 5) | AdamW, lr=1e-5 |
| Discriminator heads (~10M params) | Yes | Every step | AdamW, lr=1e-4 |
| Teacher (6B params) | No | Never | — |
| VAE | No | Never | — |
| Text encoder (Qwen3) | No | Never | — |

The discriminator learns 10× faster (higher LR, more frequent updates) than the student. This is intentional — the discriminator must stay ahead of the student to provide useful gradients.

---

## 6. Training Stability

Adversarial training is notoriously unstable. LADD uses several mechanisms to keep training stable:

### 6.1 Re-noising as a stabilizer

The re-noising at `t_hat` prevents the discriminator from latching onto trivial artifacts. Without re-noising, the discriminator could distinguish real from fake based on subtle pixel-level statistics (a common GAN failure mode). Adding noise forces discrimination based on meaningful structure.

The `LogitNormal(1, 1)` distribution for `t_hat` ensures a smooth mixture of noise levels — the discriminator can't specialize on any single scale.

### 6.2 Student initialized from teacher weights

Unlike standard GANs (which start from random init), the student begins as a copy of the teacher. At step 0:
- Student's outputs ≈ Teacher's outputs
- Discriminator sees nearly identical distributions
- Gradients are small and well-behaved

The student only needs to learn trajectory shortcuts, not the entire data distribution.

### 6.3 Warm-up schedule

```
Steps 0–500:   t sampled from {0.5, 0.25} only  (easy denoising)
Steps 500+:    t=1.0 emphasized at 70%            (hard denoising)
```

Easy tasks first → stable equilibrium → gradually increase difficulty.

### 6.4 Hinge loss

The hinge loss saturates when the discriminator is confident:
```python
d_loss = mean(relu(1 - real_logits)) + mean(relu(1 + fake_logits))
```

Once `real_logits > 1` or `fake_logits < -1`, the loss contribution is zero. This prevents the discriminator from becoming arbitrarily strong, which would produce exploding gradients for the student.

### 6.5 Update frequency asymmetry

The discriminator updates every step, but the student updates every 5 steps (`gen_update_interval=5`). This gives the discriminator time to form a stable "opinion" before the student adjusts — reducing oscillation.

### 6.6 Gradient clipping

Both student and discriminator use `max_grad_norm=1.0`. This caps the magnitude of any single update, preventing spikes from destabilizing training.

### Warning signs during training

| Signal | Meaning | Action |
|--------|---------|--------|
| `d_loss → 0` quickly | Discriminator wins too easily | Lower disc LR, increase `gen_update_interval` |
| `g_loss` oscillates wildly | Unstable equilibrium | Reduce both LRs, check gradient norms |
| Both losses flat | Mode collapse or stalled training | Check if student outputs are all identical |
| Gradient norms spike | Exploding gradients through teacher | Reduce `max_grad_norm`, enable `gradient_checkpointing` |
| `d_loss` and `g_loss` both hover near ~1.0 | Healthy adversarial equilibrium | This is good — keep training |

---

## 7. Code Map

```
Z-Image-LADD-distillation/
│
├── src/zimage/
│   ├── transformer.py          # MODIFIED: added return_hidden_states to forward()
│   │                           #   When True, collects hidden states after each of 30 blocks
│   │                           #   Returns (output, {"hidden_states": [...], "x_item_seqlens": [...], ...})
│   ├── pipeline.py             # Inference pipeline (generate function with CFG)
│   ├── autoencoder.py          # VAE (AutoencoderKL, 16 latent channels)
│   └── scheduler.py            # FlowMatchEulerDiscreteScheduler
│
├── src/config/
│   └── __init__.py             # Constants: hidden_dim=3840, n_layers=30, etc.
│
├── training/
│   ├── train_ladd.py           # Main training script (~850 lines)
│   │                           #   - Accelerate-based distributed training
│   │                           #   - Alternating D/G updates with configurable interval
│   │                           #   - Warm-up timestep schedule
│   │                           #   - Checkpointing + validation + TensorBoard
│   ├── ladd_discriminator.py   # LADDDiscriminatorHead + LADDDiscriminator (~210 lines)
│   │                           #   - 6 heads at layers [5, 10, 15, 20, 25, 29]
│   │                           #   - FiLM conditioning on t_hat + text embeddings
│   │                           #   - Hinge GAN loss
│   ├── ladd_utils.py           # Utilities (~232 lines)
│   │                           #   - logit_normal_sample, add_noise, encode_prompt
│   │                           #   - TextDataset, DiscreteSampling
│   └── train_ladd.sh           # 8-GPU launch script with env var overrides
│
├── data/
│   ├── prepare_prompts.py      # Downloads benchmarks, classifies, creates debug split
│   ├── generate_prompts.py     # Gap-fills via Claude API to ~10K prompts
│   ├── all_classified_prompts.json   # 2,553 prompts (pre-computed)
│   ├── debug/metadata.json     # 73 prompts for smoke testing
│   └── train/                  # Populated by generate_prompts.py
│
├── inference.py                # Standard Z-Image inference
└── batch_inference.py          # Batch inference from prompt file
```

### The one change to Z-Image source

In `src/zimage/transformer.py`, the `forward()` method was extended:

```python
def forward(self, x, t, cap_feats, ..., return_hidden_states=False):
    ...
    hidden_states_list = []
    for block in self.blocks:
        x = block(x, ...)
        if return_hidden_states:
            hidden_states_list.append(x.clone())
    ...
    if return_hidden_states:
        return output, {"hidden_states": hidden_states_list,
                        "x_item_seqlens": x_item_seqlens,
                        "cap_item_seqlens": cap_item_seqlens}
    return output, {}
```

This is the hook that lets the teacher expose its intermediate representations to the discriminator.

---

## 8. Data Pipeline

LADD does not use real images. The training data is just **text prompts**.

### Debug split (ready to use)

`data/debug/metadata.json` — 73 prompts, 1 per Subject×Style cell. Use for smoke testing.

### Full training set

1. Run `python data/prepare_prompts.py` to download and classify 2,553 benchmark prompts
2. Run `python data/generate_prompts.py` (requires `ANTHROPIC_API_KEY`) to gap-fill to ~10K prompts
3. Output: `data/train/metadata.json`

### Format

```json
[
    {"text": "A red sports car parked on a cobblestone street"},
    {"text": "An astronaut riding a horse on the moon, oil painting"},
    ...
]
```

### Data taxonomy (MECE)

Prompts are classified across three axes to ensure coverage:
- **Subject** (14 categories): People, Animals, Vehicles, Architecture, Food, ...
- **Style** (7 categories): Photorealistic, Digital Art, Oil Painting, Anime, ...
- **Camera** (8 categories): Close-up, Wide Angle, Aerial, Macro, ...

---

## 9. Running Training

### Prerequisites

```bash
# Install dependencies
uv pip install -e ".[training]"

# Download Z-Image weights to models/Z-Image/
# (see SETUP.md for instructions)
```

### Single-GPU smoke test

```bash
accelerate launch --num_processes=1 training/train_ladd.py \
    --pretrained_model_name_or_path=models/Z-Image \
    --train_data_meta=data/debug/metadata.json \
    --train_batch_size=1 \
    --max_train_steps=100 \
    --image_sample_size=512 \
    --output_dir=output/smoke_test
```

**What to verify:**
- [ ] Forward pass completes without shape errors
- [ ] `d_loss` and `g_loss` are logged (not NaN)
- [ ] Gradient norms are finite
- [ ] Checkpoint saves at step 100
- [ ] Validation images are generated (check `output/smoke_test/samples/`)

### Full 8-GPU training run

```bash
# Set paths
export MODEL_PATH=models/Z-Image
export DATA_META=data/train/metadata.json
export OUTPUT_DIR=output/ladd
export NUM_GPUS=8

# Launch
bash training/train_ladd.sh
```

**Key hyperparameters in the launch script:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_train_steps` | 50,000 | Assignment asks for 20K; 50K is the default |
| `learning_rate` | 1e-5 | Student LR — conservative for stability |
| `learning_rate_disc` | 1e-4 | Discriminator LR — 10× student for faster adaptation |
| `gen_update_interval` | 5 | Student updates every 5th step |
| `image_sample_size` | 512 | Resolution (square) |
| `train_batch_size` | 1 | Per-GPU; effective batch = 8 with 8 GPUs |
| `gradient_checkpointing` | enabled | Required to fit 2× 6B models in VRAM |
| `mixed_precision` | bf16 | Standard for A100s |
| `checkpointing_steps` | 500 | Save every 500 steps |
| `warmup_schedule_steps` | 500 | Easy timesteps for first 500 steps |

### Resume from checkpoint

```bash
# Resume from latest checkpoint
accelerate launch training/train_ladd.py \
    ... \
    --resume_from_checkpoint=latest
```

---

## 10. Monitoring and Debugging

### TensorBoard

```bash
tensorboard --logdir=output/ladd/logs
```

**Metrics logged each step:**
- `d_loss` — discriminator's classification loss (hinge)
- `g_loss` — generator's adversarial loss (how well student fools discriminator)
- `g_loss_update` — the actual loss used for student's backward pass
- `student_t_mean` — average student timestep sampled this step
- `t_hat_mean` — average re-noising timestep
- `lr_student`, `lr_disc` — current learning rates

### Healthy training curves

```
d_loss:  starts ~1.0, stays in [0.3, 1.5] range with slight decrease
g_loss:  starts ~1.0, gradually decreases toward 0.3–0.5
Neither should go to 0 or spike above 5.
```

### Validation images

Generated every `validation_steps` (default 1000) to `output/ladd/samples/`. These use the student with 4-step inference and no CFG. Compare across checkpoints to assess quality progression.

### Memory budget (per GPU, A100 80GB)

| Component | Approx. VRAM |
|-----------|-------------|
| Student (bf16) | ~12 GB |
| Teacher (bf16) | ~12 GB |
| Student activations (grad checkpointing) | ~8 GB |
| Teacher activations (for backward through ops) | ~15 GB |
| Discriminator | ~0.5 GB |
| Optimizer states (student + disc) | ~15 GB |
| **Total** | **~62 GB** |

If OOM: reduce `image_sample_size` to 256, or enable gradient accumulation.

---

## 11. Inference with the Distilled Student

After training, the student checkpoint is at `output/ladd/student_transformer/pytorch_model.bin`.

### 4-step inference (no CFG)

```python
from src.zimage.pipeline import generate

images = generate(
    model_path="output/ladd/student_transformer",
    prompt="A red sports car on a cobblestone street",
    num_inference_steps=4,    # 4 steps instead of 50
    guidance_scale=0.0,       # no CFG needed
    height=1024, width=1024,
)
```

### Step configurations

| Mode | Timesteps | Quality | Speed |
|------|-----------|---------|-------|
| 4-step | t ∈ {1.0, 0.75, 0.5, 0.25} | Best | ~4× faster than teacher |
| 2-step | t ∈ {1.0, 0.5} | Good | ~12× faster |
| 1-step | t = 1.0 only | Acceptable | ~25× faster |

### Comparing student vs teacher

Run the same prompts through both:
- **Teacher:** 50 steps, CFG scale 3.5
- **Student:** 4 steps, no CFG

The student's outputs should have comparable quality but may show:
- Slightly weaker prompt alignment (object counts, spatial relationships)
- Occasional object merging or duplication
- Less fine-grained detail at 1-step; nearly matching at 4-step
