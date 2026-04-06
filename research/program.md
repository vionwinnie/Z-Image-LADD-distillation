# LADD Distillation Research — Anti-Mode-Collapse Hyperparameter Search

This is an experiment to have the LLM optimize hyperparameters for LADD (Latent Adversarial Diffusion Distillation) of a 6.15B parameter image generation model.

## Background: Mode Collapse Problem

A full-scale run (4K steps, 8 GPUs, gradient_accumulation=8) with these settings collapsed:

```
slr=5e-6 dlr=5e-5 GI=3 warmup=10 renoise_m=1.0 renoise_s=1.0
disc_hidden_dim=256 disc_layer_indices=[5,10,15,20,25,29]
```

**Symptoms**: Discriminator became too powerful → generator started producing garbage for every prompt. This is classic GAN mode collapse where the discriminator wins the adversarial game.

**Hypothesis**: The discriminator-to-generator balance is off. With effective batch size 8 (grad_accum=8), the discriminator gets much stronger gradient signal per update. At GI=3 (3 disc steps per gen step), the ratio is even worse. The noise schedule (M=1.0, high noise) may also make the discriminator's job too easy since heavily noised real and fake samples are easier to distinguish at feature level.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `PROGRESS.md` — training history, what worked, what failed, current best config.
   - `research/experiment.py` — the file you modify. Hyperparameters, architecture knobs, training config.
   - `research/evaluate.py` — fixed evaluation. Do not modify.
   - `research/results.tsv` — past experiment results.
4. **Verify models exist**: Check that `models/Z-Image/` contains the pretrained model and that `data/train/embeddings_latent_subset/` contains precomputed text embeddings. If not, tell the human.
5. **Verify GPU**: Run `nvidia-smi` to confirm a GPU is available and no stale processes are holding memory. Kill stale python/torch processes if needed.
6. **Initialize results.tsv**: If starting fresh, create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the calibration runs, then the experimentation.

## Data

Training uses **3072 prompts** with precomputed teacher latents and text embeddings:

- `data/train/metadata_latent_subset.json` — 3072 entries filtered from 10K subsample, only entries with teacher latents
- `data/train/embeddings_latent_subset/` — precomputed text embeddings for those 3072 entries
- `data/train/teacher_latents_subset/` — symlinked teacher latent files (reindexed to 0..3071)

Validation uses 1000 teacher images from `data/val/teacher_images/` for KID computation, and precomputed val embeddings from `data/val/embeddings/`.

## Experimentation

Each experiment runs on a single A100 80GB. The training script runs for a **fixed step budget** (`MAX_TRAIN_STEPS` in `experiment.py`, default 500 steps, ~10 min training + ~5 min evaluation). You launch it simply as: `python research/experiment.py > research/run.log 2>&1`.

**Budget**: ~3 hours = ~12 experiment runs (including baseline). Every slot is precious — prioritize high-information experiments.

**What you CAN do:**
- Modify `research/experiment.py` — this is the only file you edit. The tunable section between `=== TUNABLE HYPERPARAMETERS START ===` and `=== TUNABLE HYPERPARAMETERS END ===` is your playground: learning rates, noise schedule, discriminator architecture, training dynamics, timestep configuration.

**What you CANNOT do:**
- Modify `research/evaluate.py`. It is read-only. It contains the fixed KID evaluation.
- Modify anything under `training/`, `src/`, `scripts/`. The core codebase is frozen.
- Change `MAX_TRAIN_STEPS`, `TRAIN_BATCH_SIZE`, `SEED`, `IMAGE_SIZE`, or `MODEL_PATH`. These are fixed for comparability.
- Install new packages.

**The goal is simple: get the lowest KID (kid_mean) without mode collapse.** Since the step budget is fixed, you don't need to worry about training time — it's always the same. Everything in the tunable section is fair game.

**Wandb logging**: Each run logs KID and 20 sample images to wandb under project `ladd`. Check wandb for visual quality alongside KID numbers.

**Discriminator health** is a hard constraint. An experiment is automatically discarded regardless of KID if:
- Any loss is NaN or Inf
- `disc_accuracy_real > 95%` AND `disc_accuracy_fake > 95%` (student stuck, disc too powerful)
- `disc_accuracy_real < 55%` AND `disc_accuracy_fake < 55%` (discriminator collapsed)
- `logit_gap > 15` (discriminator diverging from student)
- `logit_gap < 0.1` after warmup (discriminator provides no learning signal)

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Your very first run should always be to establish the baseline (the collapsed config), so you will run the training script as is.

## Output format

Everything runs in a single process. Training computes disc metrics inline, then runs KID evaluation at the final step (generates 20 student images, computes KID against 1000 cached teacher images, logs sample images to wandb).

The log contains two summary blocks:

**1. Training summary** (printed at end of training):
```
---
training_steps:     500
early_stopped:      False
d_loss:  1.234567
disc/accuracy_fake:  0.750000
disc/accuracy_real:  0.680000
disc/logit_gap:  2.345678
g_loss:  -0.567890
peak_vram_mb:       72300.0
---
```

**2. KID from inline validation** (printed by logger):
```
KID = 0.123456 ± 0.005678
```

**Early stopping**: Disabled (the health-check logic is broken at batch_size=1). Every run takes the full step budget (~15 min). Budget your experiments accordingly.

**At the end**, the bash script prints a `--- RESULTS ---` block summarizing all metrics. Extract key info:

```
grep "early_stopped:\|KID = \|peak_vram_mb:" research/run.log
```

## Logging results

When an experiment is done, log it to `research/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	kid_mean	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. kid_mean achieved (e.g. 0.1234) — use 0.000000 for crashes
3. peak VRAM in GB, round to .1f (check `nvidia-smi` during run) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried, including key param values

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr6`).

LOOP FOREVER:

1. Look at the git state: `git log --oneline -5` and `cat research/results.tsv`
2. Read `research/experiment.py` to see current config
3. Tune `experiment.py` with ONE experimental idea — change one thing in the tunable section
4. git commit: `git add research/experiment.py && git commit -m "<description>"`
5. Run the experiment: `python research/experiment.py > research/run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "early_stopped:\|KID = \|peak_vram_mb:" research/run.log`
7. If the grep output is empty or KID is missing, the run crashed. Run `tail -n 50 research/run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
9. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
10. If kid_mean improved (lower), you "advance" the branch, keeping the git commit
11. If kid_mean is equal or worse, you `git reset --hard HEAD~1` back to where you started

**Timeout**: Every run takes ~15 minutes (10 min training + 5 min eval). If a run exceeds 25 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the LADD paper references in the code, re-read PROGRESS.md for clues, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

## What to explore — Anti-Mode-Collapse Focus

The central problem is **discriminator dominance** at scale. The full run (8 GPU, grad_accum=8, 4K steps) collapsed. These experiments aim to find a stable discriminator–generator balance.

With ~12 runs in 3 hours, prioritize high-information experiments. The baseline config matches the collapsed run.

### Priority 1: Weaken the discriminator (CRITICAL — start here)

The discriminator is too strong. Multiple levers to weaken it:

**`GEN_UPDATE_INTERVAL`** — fewer disc steps per gen step:
- Current: 3 (collapsed). Try: **1** (equal updates), **2** (mild disc advantage)
- At GI=1, disc and gen update equally. This is the most direct way to rebalance.
- With effective batch size 8 at scale, even GI=1 may still favor the disc.

**`DISC_LR`** — lower disc learning rate:
- Current: 5e-5 (collapsed). Try: **1e-5**, **2e-5**
- Lower disc LR = slower disc learning = more room for generator to catch up.
- Previous experiment found 1e-5 helped slightly with GI=3.

**`DISC_HIDDEN_DIM`** — smaller discriminator capacity:
- Current: 256. Try: **128**, **64**
- Fewer parameters = less capacity to memorize real/fake distinction.
- This directly limits how powerful the discriminator can become.

### Priority 2: Noise schedule (helps mask real/fake difference)

Higher noise makes real and fake samples more similar, giving the discriminator less signal. But too much noise means the generator gets no useful feedback.

**`RENOISE_M`** (logit-normal mean):
- Current: 1.0 (sigmoid≈0.73, biased toward high noise). Try: **0.0** (balanced), **0.5**, **1.5**
- Lower M → less noise → sharper discrimination signal but potentially unstable
- Higher M → more noise → disc sees near-pure noise (too easy)

**`RENOISE_S`** (logit-normal std):
- Current: 1.0. Try: **0.5** (tight cluster), **1.5** (wide spread)
- Tight S forces all samples to similar noise level — could stabilize training

### Priority 3: Discriminator architecture (fewer features to exploit)

**`DISC_LAYER_INDICES`** — which transformer layers the disc sees:
- Current: [5,10,15,20,25,29] (6 layers). Try: **[10,20,29]** (3 layers), **[15,29]** (2 layers)
- Fewer layers = less information for disc to exploit = weaker disc
- Early layers have low-level features, late layers have semantic features

### Priority 4: Learning rate rebalancing

**`STUDENT_LR`** — boost the generator:
- Current: 5e-6. Try: **1e-5** (2x faster generator)
- Higher student LR helps the generator keep up with a strong discriminator
- But too high causes instability (2e-5 was too aggressive in prior experiments)

### Priority 5: Warmup and schedule

**`WARMUP_SCHEDULE_STEPS`** — timestep warmup:
- Current: 10. Try: **0** (no warmup), **50**, **100**
- Longer warmup eases the generator in before full adversarial pressure

**`LR_WARMUP_STEPS`** — LR warmup:
- Current: 0. Try: **50**, **100**
- Warmup prevents early disc dominance by starting both models with low LR

### Suggested experiment order (12 runs)

1. **Baseline** — run as-is (matches collapsed config)
2. **GI=1** — most direct anti-collapse intervention
3. **dlr=1e-5** — halve disc learning rate
4. **Best of 2-3 + disc_hidden_dim=128** — weaker disc architecture
5. **Best so far + RENOISE_M=0.0** — balanced noise
6. **Best so far + RENOISE_M=0.5** — moderate noise
7. **Best so far + disc_layers=[10,20,29]** — fewer disc features
8. **Best so far + slr=1e-5** — faster generator
9. **Best so far + RENOISE_S=0.5** — tight noise spread
10. **Best so far + warmup_schedule=50** — longer warmup
11. **Best so far + disc_hidden_dim=64** — even weaker disc
12. **Combine best findings** — put it all together

Adapt based on results — if GI=1 alone solves collapse, explore noise schedule more aggressively. If disc_hidden_dim=128 is a big win, try 64.

### Not tunable
- `STUDENT_TIMESTEPS`: fixed at [1.0, 0.75, 0.5, 0.25] — do not change
- `TEXT_DROP_RATIO`: 0.1 — standard CFG dropout, unlikely to matter much
- `DISC_COND_DIM`: 256 — tied to hidden dim, change only if hidden dim changes

## Infrastructure notes

- Training uses **8-bit Adam** (`--cpu_offload_optimizer`) to fit on a single A100 80GB
- **Precomputed embeddings** (`--embeddings_dir`) skip the ~3GB text encoder during training
- **Precomputed teacher latents** (`--teacher_latents_dir`) provide real teacher outputs per prompt
- `--skip_save` saves only student weights as safetensors (~12GB), not full optimizer state
- `experiment.py` uses `os.execv` to replace the Python process with bash, avoiding GPU memory leaks from the parent process
- Before each run, check `nvidia-smi` for stale processes — kill them or you'll OOM
- `--max_grad_norm=1.0` for gradient clipping (matches full run config)
- 20 sample images + KID logged to wandb per run
