# Cloud Setup & Handoff Guide

## Quick Start (RunPod / Any Cloud GPU)

### 1. Clone and Setup Environment

```bash
git clone git@github.com:vionwinnie/Z-Image-LADD-distillation.git
cd Z-Image-LADD-distillation

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv venv
uv pip install -e ".[training]"
source .venv/bin/activate
```

### 2. Download Model Weights

Z-Image model weights are required. Download from HuggingFace:

```bash
mkdir -p models
# Option A: Use huggingface-cli
huggingface-cli download Tongyi-MAI/Z-Image --local-dir models/Z-Image

# Option B: If Turbo variant is desired for comparison
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir models/Z-Image-Turbo
```

Expected directory structure after download:
```
models/Z-Image/
  transformer/
    config.json
    *.safetensors (or *.safetensors.index.json + shards)
  vae/
    config.json
    *.safetensors
  text_encoder/
    config.json
    model*.safetensors
    tokenizer.json (or in tokenizer/ subfolder)
  tokenizer/
    ...
  scheduler/
    scheduler_config.json
```

### 3. Prepare Dataset

#### Step A: Download and classify benchmark prompts
```bash
python data/prepare_prompts.py
```

This downloads PartiPrompts, GenAI-Bench, and DrawBench, classifies them into our MECE taxonomy, and outputs:
- `data/all_classified_prompts.json` — ~2,500 classified prompts
- `data/debug/metadata.json` — ~70 debug prompts (1 per Subject x Style cell)

#### Step B: Fill gaps to reach ~10K prompts (requires Claude API key)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Dry run first to see what gaps exist
python data/generate_prompts.py --dry-run

# Generate prompts to fill gaps
python data/generate_prompts.py
```

This outputs:
- `data/train/metadata.json` — full ~10K training prompts
- `data/coverage_matrix.csv` — prompt counts per (Subject x Style) cell

If you don't have an API key, you can use the debug split (`data/debug/metadata.json`) for smoke testing.

### 4. Smoke Test (Single GPU or CPU)

```bash
# Quick validation that everything works
python scripts/smoke_test.py \
  --pretrained_model_name_or_path=models/Z-Image \
  --train_data_meta=data/debug/metadata.json
```

This validates:
- Forward pass shapes and dtypes
- Teacher feature extraction (intermediate hidden states)
- Discriminator head outputs
- Gradient flow (student + discriminator trainable, teacher frozen)
- Loss computation (hinge loss, finite values)
- Checkpoint save/load round-trip

### 5. Full Training (8x A100)

```bash
# Launch on 8 GPUs
bash training/train_ladd.sh
```

Or manually:
```bash
accelerate launch \
  --num_processes=8 \
  --mixed_precision=bf16 \
  training/train_ladd.py \
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

Monitor training:
```bash
tensorboard --logdir=output_ladd/logs
```

### 6. Inference

```bash
python scripts/inference_ladd.py \
  --student_checkpoint=output_ladd/checkpoint-20000 \
  --teacher_model=models/Z-Image \
  --output_dir=inference_results
```

---

## File Structure

```
Z-Image-LADD-distillation/
  ANALYSIS.md              # Detailed analysis of LADD vs DMD, Z-Image architecture, ambiguities
  EXECUTION_PLAN.md        # Full implementation plan with MECE taxonomy, steps, verification
  SETUP.md                 # This file — cloud setup instructions
  
  src/zimage/
    transformer.py         # MODIFIED — added return_hidden_states for intermediate feature extraction
    pipeline.py            # Original Z-Image inference pipeline
    autoencoder.py         # VAE
    scheduler.py           # FlowMatchEulerDiscreteScheduler
  
  training/
    train_ladd.py          # Main LADD training script (Accelerate-based)
    ladd_discriminator.py  # Discriminator heads (2D conv + FiLM conditioning)
    ladd_utils.py          # Utilities: TextDataset, encode_prompt, logit-normal sampling
    train_ladd.sh          # 8-GPU launch script
  
  data/
    prepare_prompts.py     # Downloads benchmarks, classifies into MECE taxonomy
    generate_prompts.py    # Fills gaps via Claude API to reach ~10K prompts
    all_classified_prompts.json  # Pre-classified benchmark prompts (~2,553)
    debug/metadata.json    # Debug split (~73 prompts)
    train/metadata.json    # Full training split (~10K, after generate_prompts.py)
  
  scripts/
    smoke_test.py          # TODO: End-to-end validation script
    inference_ladd.py      # TODO: Student inference + teacher comparison
  
  inference.py             # Original Z-Image inference script
  pyproject.toml           # Dependencies (base + [training] extra)
```

## What's Done

- [x] Transformer modified for intermediate feature extraction (`return_hidden_states`)
- [x] LADD discriminator heads implemented (2D conv, FiLM conditioning, hinge loss)
- [x] Training utilities (logit-normal sampling, TextDataset, prompt encoding)
- [x] Full LADD training script with Accelerate distributed training
- [x] 8-GPU launch script
- [x] Data pipeline: benchmark download + MECE classification
- [x] Data pipeline: LLM gap-filling script
- [x] Benchmark prompts downloaded and classified (2,553 prompts)
- [x] Debug dataset created (73 prompts)
- [x] uv environment configured with all dependencies

## What's TODO

- [ ] Run `generate_prompts.py` to fill dataset to ~10K (needs `ANTHROPIC_API_KEY`)
- [ ] Write `scripts/smoke_test.py` — end-to-end validation
- [ ] Write `scripts/inference_ladd.py` — inference + comparison script
- [ ] Run smoke test on single GPU to validate pipeline
- [ ] Launch full 20K-step training on 8x A100
- [ ] Run inference demo and collect sample images

## Key Architecture Decisions

### Why LADD over the existing DMD distillation?
- The VideoX-Fun repo already has DMD distillation (`train_distill.py`) using 3 full transformer copies
- LADD uses lightweight discriminator heads instead of a 3rd transformer — saves ~30GB GPU memory
- LADD uses adversarial hinge loss (not DMD score-difference loss)
- See `ANALYSIS.md` for full comparison

### LADD Training Loop (simplified)
1. Sample prompts, encode with Qwen3
2. Sample timestep t from {1.0, 0.75, 0.5, 0.25} (biased toward t=1.0)
3. Student denoises from noise at timestep t
4. Re-noise student output at t_hat ~ LogitNormal(1, 1)
5. Teacher processes re-noised output, returns intermediate features
6. Discriminator heads classify features as real/fake
7. Hinge loss: student tries to fool discriminator, discriminator tries to distinguish
8. Alternating updates: discriminator every step, student every N steps

### Data Pipeline
- Prompts sourced from 3 established benchmarks: PartiPrompts, GenAI-Bench, DrawBench
- Classified into a 3-axis MECE taxonomy: Subject (14) x Style (7) x Camera (8)
- Gaps filled to ~100 prompts per (Subject x Style) cell using Claude API
- Total target: ~10K prompts
- See `EXECUTION_PLAN.md` for full taxonomy tables

## Environment Requirements

- Python >= 3.10
- CUDA-capable GPU(s) for training
- ~160GB GPU memory total for 8-GPU training (student + teacher in bf16 + discriminator heads)
- ~100GB disk for model weights + checkpoints
- Packages: torch >= 2.5, transformers, diffusers, accelerate, safetensors, datasets, anthropic (for data gen)
