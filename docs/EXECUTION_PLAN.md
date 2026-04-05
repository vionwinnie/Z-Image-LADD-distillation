# LADD Distillation of Z-Image — Implementation Plan

## Context

We're implementing LADD (Latent Adversarial Diffusion Distillation) for Z-Image as a take-home assignment. The goal is a working distilled student model that generates images in 4 steps (vs teacher's 50+), trained on 8×A100 GPUs for 20K steps.

**Working repo**: `Z-Image-LADD-distillation` (fork of Z-Image at `git@github.com:vionwinnie/Z-Image-LADD-distillation.git`)

The VideoX-Fun repo already has a DMD-style distillation script (`train_distill.py`), but LADD differs significantly in discriminator architecture and loss function. We'll fork the infrastructure but rewrite the core training logic.

---

## File Plan

### New Files to Create

| File | Purpose |
|------|---------|
| `training/train_ladd.py` | Main LADD training script (~800 lines) |
| `training/ladd_discriminator.py` | Discriminator head architecture (~150 lines) |
| `training/ladd_utils.py` | Sampling utilities, logit-normal dist (~100 lines) |
| `training/train_ladd.sh` | 8-GPU launch script |
| `data/debug/metadata.json` | 100 prompts for smoke testing |
| `data/train/metadata.json` | 10K+ prompts for full training |
| `data/prepare_prompts.py` | Prompt sourcing/filtering script |
| `scripts/smoke_test.py` | End-to-end validation script |
| `scripts/inference_ladd.py` | Student inference + comparison |

### Existing Files to Modify

| File | Change |
|------|--------|
| `src/zimage/transformer.py` | Add `return_hidden_states` parameter to `forward()` to extract intermediate features from attention blocks |

---

## Implementation Steps

### Step 1: Transformer Feature Extraction

**File**: `src/zimage/transformer.py` — `ZImageTransformer2DModel.forward()`

Add optional `return_hidden_states=True` parameter. When enabled, collect the unified token sequence after each of the 30 main transformer blocks (line 564 in current code). Return them as a list alongside the normal output.

```python
# In forward(), after the main loop:
hidden_states_list = []
for layer in self.layers:
    unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
    if return_hidden_states:
        hidden_states_list.append(unified)
```

Also need to extract the **image-only tokens** from the unified sequence for the discriminator (strip text tokens). The image token count per sample is available from `x_item_seqlens`.

### Step 2: Discriminator Heads

**File**: `training/ladd_discriminator.py`

Based on LADD/Projected GAN design — lightweight 2D conv heads on teacher features:

```
LADDDiscriminatorHead:
  - Reshape tokens to 2D spatial layout (H_tokens × W_tokens)
  - Conv2d(dim, 256, 3, padding=1) + GroupNorm + SiLU
  - Conv2d(256, 128, 3, padding=1) + GroupNorm + SiLU  
  - Conv2d(128, 1, 1) → per-patch logit
  - Conditioning: timestep embedding + pooled text embedding added to features via FiLM (scale + shift)

LADDDiscriminator:
  - Collection of heads, one per selected teacher layer (e.g., layers [5, 10, 15, 20, 25, 29])
  - Not all 30 blocks — subset for memory/compute efficiency
  - Adversarial hinge loss computation built-in
```

Conditioning injection: timestep + text embeddings → MLP → (scale, shift) → FiLM modulation on intermediate conv features.

### Step 3: Training Utilities

**File**: `training/ladd_utils.py`

Copy from VideoX-Fun (minimal):
- `DiscreteSampling` class (~50 lines) — distributed timestep sampling
- `RectifiedFlow_TrigFlowWrapper` (~16 lines) — trigflow scaling factors
- `sample_trigflow_timesteps` (~6 lines) — timestep sampling

Add new:
- `logit_normal_sample(batch_size, m=1.0, s=1.0)` — for re-noising distribution
- `TextDataset` (~15 lines) — JSON prompt loader with text dropout
- `encode_prompt()` — text encoding via Qwen3 (adapted from VideoX-Fun)

### Step 4: LADD Training Script

**File**: `training/train_ladd.py`

Fork structure from `VideoX-Fun/scripts/z_image/train_distill.py` but rewrite core loop.

**Model setup** (2 models, not 3):
- Student (`student_transformer`) — trainable, initialized from pretrained Z-Image
- Teacher (`teacher_transformer`) — frozen, same pretrained weights
- Discriminator heads (`discriminator`) — trainable, randomly initialized

**Training loop** (per step):

1. **Sample text prompts** from dataset
2. **Encode prompts** through Qwen3 text encoder
3. **Sample student timestep** `t` from {1, 0.75, 0.5, 0.25} with probability schedule:
   - Warmup (first 500 steps): p=[0, 0, 0.5, 0.5]
   - Main: p=[0.7, 0.1, 0.1, 0.1]
4. **Generate noisy input**: pure noise (for t=1) or interpolated
5. **Student forward pass**: denoise → get student prediction `x̂₀`
6. **Re-noise student output**: sample `t̂ ~ LogitNormal(1, 1)`, compute `x_t̂ = (1-t̂)*x̂₀ + t̂*ε`
7. **Teacher forward pass** (no grad): feed re-noised student output, extract intermediate features
8. **Discriminator heads**: process teacher features → per-patch logits (fake)
9. **Real data path**: teacher generates clean latents (or use noise at same `t̂` through teacher), extract features → discriminator → real logits
10. **Losses**:
    - Generator (student) loss: `L_G = -mean(D(fake))` (hinge)
    - Discriminator loss: `L_D = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))` (hinge)
11. **Alternating updates**: discriminator every step, student every `gen_update_interval` steps

**Distributed training**: Use Accelerate with FSDP for student model sharding. Teacher loaded with `torch.no_grad()` in bf16.

**Key hyperparameters** (from paper + VideoX-Fun defaults):
- Student LR: 2e-5, Discriminator LR: 1e-4
- Batch size: 1 per GPU, gradient accumulation: 4
- Max grad norm: 0.05
- Optimizer: AdamW (bf16)
- Scheduler: constant with 500-step warmup
- Gradient checkpointing: enabled for student

### Step 5: Data Pipeline

**Files**: `data/prepare_prompts.py`, `data/generate_prompts.py`

#### MECE Taxonomy (3 orthogonal axes)

Every prompt is tagged along three axes. The axes are mutually exclusive within each dimension and collectively exhaustive across the space Z-Image should cover.

**Axis 1: Subject (14 categories)** — what's in the image

| ID | Category | Definition (exclusive boundary) |
|----|----------|--------------------------------|
| S1 | People / Portraits | Humans as primary subject (solo, group, full-body, headshot) |
| S2 | Animals | Non-human living creatures as primary subject |
| S3 | Food & Beverage | Prepared food, drinks, ingredients, plated dishes |
| S4 | Indoor Scenes | Interior environments where the space is the subject (not a person in a room) |
| S5 | Outdoor / Landscape | Natural or rural exterior environments |
| S6 | Architecture / Urban | Buildings, cityscapes, streets, infrastructure |
| S7 | Vehicles | Cars, planes, boats, bikes, spacecraft |
| S8 | Plants / Nature | Flora, gardens, forests — vegetation as primary subject |
| S9 | Fashion / Clothing | Garments, accessories, fashion-forward compositions |
| S10 | Objects / Artifacts | Tools, furniture, devices, household items |
| S11 | Text / Typography | Signs, logos, posters, text as the visual focus |
| S12 | World Knowledge | Named landmarks, historical events, famous artworks, cultural icons |
| S13 | Chinese Cultural | Chinese-specific: hanfu, calligraphy, festivals, architecture, mythology |
| S14 | Abstract / Imagination | Non-representational, surreal, fantasy, impossible scenes |

**Axis 2: Style (7 categories)** — how it's rendered

| ID | Style | Definition |
|----|-------|-----------|
| T1 | Photorealistic | Looks like a real photograph |
| T2 | Traditional Art | Oil painting, watercolor, ink wash, impressionist, ukiyo-e |
| T3 | Digital Illustration | Concept art, anime/manga, comic book, character design |
| T4 | 3D / CGI | Rendered 3D, Unreal Engine, isometric, clay render |
| T5 | Cinematic / Film | Movie still, dramatic lighting, letterboxed, color graded |
| T6 | Graphic Design | Vector, flat design, poster, infographic, pixel art |
| T7 | Mixed / Experimental | Collage, mixed media, glitch art, double exposure |

**Axis 3: Camera / Composition (8 categories)** — technical framing

| ID | Technique | Definition |
|----|-----------|-----------|
| C1 | Standard / Eye-level | Default framing, no special technique |
| C2 | Macro / Close-up | Extreme detail, shallow DOF on small subject |
| C3 | Wide Angle / Panoramic | Expansive view, potential distortion |
| C4 | Aerial / Bird's Eye | Overhead perspective, drone-style |
| C5 | Low Angle / Worm's Eye | Looking up, dramatic perspective |
| C6 | Bokeh / Shallow DOF | Background blur, subject isolation |
| C7 | Long Exposure / Motion | Light trails, motion blur, smooth water |
| C8 | Dramatic Lighting | Golden hour, backlighting, low-key, high-key, chiaroscuro |

#### Prompt Sources

**Benchmark datasets** (all public, download via HuggingFace or GitHub):

| Source | Prompts | Format | What it gives us |
|--------|---------|--------|-----------------|
| PartiPrompts | 1,632 | HF `nateraw/parti-prompts` | 11 content categories + 4 challenge levels |
| MJHQ-30K | 30,000 | HF `playgroundai/MJHQ-30K` | 10 content categories, high quality |
| DPG-Bench | 1,065 | GitHub | Long/dense descriptions |
| DrawBench | 200 | GitHub | Skill-based: counting, spatial, colors, text |
| T2I-CompBench | ~6,000 | GitHub | Compositional: color/shape/texture binding, spatial |
| GenEval | 553 | GitHub | Object-focused: counting, position, attributes |

**Total available**: ~39,450 benchmark prompts

#### Pipeline steps (`data/prepare_prompts.py`):

1. **Ingest all benchmarks** — download/load all 6 sources
2. **Deduplicate** — exact + fuzzy matching (edit distance < 0.1)
3. **Filter** — remove <5 words, NSFW, non-English/Chinese
4. **Classify** — use an LLM (Claude API) to tag each prompt with (Subject, Style, Camera) from our MECE taxonomy
5. **Compute coverage matrix** — count prompts per (Subject × Style) cell (14×7 = 98 cells)
6. **Identify gaps** — cells with <50 prompts

#### Gap-filling (`data/generate_prompts.py`):

7. **LLM generation** — for each underfilled cell, generate prompts to reach target density:
   - Target: ~100 prompts per (Subject × Style) cell = ~9,800 base
   - Camera/composition axis applied as an overlay (~30% get non-standard camera technique)
   - Chinese-language prompts: generate ~500 across all subjects (focused on S12, S13)
   - Total target: **~10,000 prompts**
8. **Quality filter** — LLM review pass to remove low-quality generated prompts

#### Output format:
```json
[
  {
    "text": "A serene mountain landscape at sunset...",
    "subject": "S5",
    "style": "T1",
    "camera": "C8",
    "source": "parti_prompts",
    "language": "en"
  },
  ...
]
```

- `data/train/metadata.json` — full 10K prompts
- `data/debug/metadata.json` — 1 prompt per (Subject × Style) cell = ~100 prompts
- `data/coverage_matrix.csv` — prompt counts per cell for auditability

#### Rationale:
- LADD paper used prompts from the model's original training distribution — we approximate Z-Image's distribution using established benchmarks
- MECE taxonomy ensures no overlap (mutual exclusivity) and no blind spots (collective exhaustiveness)
- Z-Image paper emphasizes: Chinese culture (S13), text rendering (S11), world knowledge (S12), style diversity (OneIG/PRISM-Bench) — all explicitly covered
- Coverage matrix makes gaps visible and auditable
- Category labels enable per-category quality analysis in evaluation

### Step 6: Smoke Test

**File**: `scripts/smoke_test.py`

Validates on CPU/single GPU:
1. Load student + teacher (tiny or full model with small batch)
2. Forward pass → check output shapes
3. Teacher feature extraction → check intermediate shapes
4. Discriminator heads → check logit shapes
5. Compute adversarial loss → check scalar, finite
6. Backward pass → check gradients exist on student + discriminator, not on teacher
7. Optimizer step → check weights change
8. Checkpoint save/load → check round-trip
9. Log: memory usage, loss values, grad norms

### Step 7: Launch Script

**File**: `training/train_ladd.sh`

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 \
  training/train_ladd.py \
  --pretrained_model_name_or_path=models/Z-Image \
  --train_data_meta=data/train/metadata.json \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=20000 \
  --learning_rate=2e-05 \
  --learning_rate_disc=1e-04 \
  --checkpointing_steps=1000 \
  --gradient_checkpointing \
  --image_sample_size=1024 \
  --seed=42 \
  --output_dir=output_ladd
```

### Step 8: Inference Script

**File**: `scripts/inference_ladd.py`

- Load student checkpoint
- Generate images from 10+ representative prompts
- Compare side-by-side with teacher (50-step)
- Measure and print latency
- Save comparison grid

---

## Verification

1. **Smoke test passes**: `python scripts/smoke_test.py` completes without error
2. **Gradient check**: Student and discriminator params have non-zero gradients; teacher has none
3. **Loss decreases**: Over 100 steps on debug data, generator loss trends down
4. **Checkpoint round-trip**: Save at step N, reload, continue training — losses consistent
5. **Multi-GPU**: `accelerate launch --num_processes=2` works on dev instance
6. **Inference**: Student produces coherent images in 4 steps

---

## Execution Order

**Phase A — Parallel workstreams:**

| Workstream | Steps | Notes |
|-----------|-------|-------|
| Data Pipeline | Step 5 (prepare_prompts.py, generate_prompts.py) | Independent — no code dependencies on training |
| Training Setup | Steps 1-4 in parallel: transformer mod, discriminator heads, utils, training script | Interdependent but can be developed concurrently by writing interfaces first |

**Phase B — Sequential (after A completes):**

6. Smoke test (Step 6) — needs both data + training code
7. Launch script (Step 7) — for 8×A100
8. Inference script (Step 8) — demo
