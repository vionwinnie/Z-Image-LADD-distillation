# LADD Distillation Research Program

You are an ML research agent running experiments to find the best hyperparameters
for LADD (Latent Adversarial Diffusion Distillation) of a 6.15B parameter
image generation model. You will iterate autonomously: propose a change, run it,
evaluate results, keep or discard, repeat.

## Goal

Minimize **FID score** (Frechet Inception Distance) on the validation prompt set
while keeping discriminator metrics healthy. Lower FID = better image quality.
Secondary metric: CLIP score (higher = better text-image alignment).

## In-Scope Files

You may **read** all files, but you may only **modify**:

- `research/experiment.py` — the training script. All hyperparameters and
  architectural choices live here. This is your single point of modification.

You may **NOT modify**:

- `research/evaluate.py` — the evaluation script. Computes FID, CLIP score,
  discriminator accuracy, and logit gap from a checkpoint. Immutable.
- `research/program.md` — this file.
- Anything under `src/`, `training/`, `scripts/` — the core codebase.

## Experiment Loop

Run this loop forever until interrupted by the human:

```
1. READ current state:
   - `git log --oneline -5` (recent history)
   - `cat research/results.tsv` (past experiment results)
   - `cat research/experiment.py` (current config)

2. PROPOSE a change:
   - Edit research/experiment.py with ONE hypothesis
   - Write a clear commit message explaining what and why
   - `git add research/experiment.py && git commit -m "<description>"`

3. RUN the experiment:
   - `python research/experiment.py 2>&1 | tee research/run.log`
   - If it crashes, read the last 50 lines, attempt a fix, re-run ONCE
   - If it crashes again, log as "crash" and move on

4. EVALUATE:
   - `python research/evaluate.py --checkpoint research/checkpoint 2>&1 | tee research/eval.log`
   - Extract: fid, clip_score, disc_accuracy_real, disc_accuracy_fake, logit_gap

5. LOG results:
   - Append a row to research/results.tsv

6. DECIDE:
   - If FID IMPROVED (lower): KEEP the commit
   - If FID same or worse: DISCARD via `git reset --hard HEAD~1`
   - Exception: if disc metrics show collapse (accuracy > 95% or < 55%),
     discard even if FID improved — it's unstable

7. REPEAT from step 1
```

## Hyperparameter Space

These are the knobs in `experiment.py` you should explore, roughly in priority order:

### Tier 1: Learning rates (most impactful)
- `STUDENT_LR`: {1e-6, 5e-6, 1e-5, 2e-5, 5e-5}
- `DISC_LR`: {5e-5, 1e-4, 2e-4, 5e-4}
- The ratio DISC_LR / STUDENT_LR ~ 10x is a good starting point

### Tier 2: Training dynamics
- `GEN_UPDATE_INTERVAL`: {1, 2, 3, 5, 10} — D steps per G step
- `WARMUP_STEPS`: {0, 50, 100, 200} — LR warmup (scaled for time budget)
- `WARMUP_SCHEDULE_STEPS`: {0, 50, 100} — timestep schedule warmup

### Tier 3: Noise schedule
- `RENOISE_M`: {0.0, 0.5, 1.0, 1.5} — logit-normal mean
- `RENOISE_S`: {0.5, 1.0, 1.5} — logit-normal std

### Tier 4: Architecture (only if Tier 1-3 are exhausted)
- `DISC_HIDDEN_DIM`: {128, 256, 512}
- `DISC_LAYER_INDICES`: different subsets of teacher layers
- `STUDENT_TIMESTEPS`: {[1.0, 0.5], [1.0, 0.75, 0.5, 0.25], [1.0, 0.8, 0.6, 0.4, 0.2]}

## Health Checks (hard constraints)

An experiment is **automatically discarded** regardless of FID if:
- Any loss is NaN or Inf
- `disc_accuracy_real > 95%` AND `disc_accuracy_fake > 95%` (student stuck)
- `disc_accuracy_real < 55%` AND `disc_accuracy_fake < 55%` (discriminator collapsed)
- `logit_gap > 15` (discriminator diverging)
- `logit_gap < 0.1` after warmup (discriminator useless)
- Training crashes or exceeds time budget by >20%

## Strategy Tips

- Start with learning rate sweeps. LR is almost always the most impactful knob.
- Change ONE thing at a time. If you change LR and noise schedule together,
  you can't attribute the result.
- If stuck (3+ consecutive discards), try a more radical change or revisit
  a previously discarded direction with a small modification.
- Read the results.tsv history before proposing. Don't repeat failed experiments.
- Simpler is better: if removing a feature gives equal FID, keep the simpler version.
- The first experiment should always be the BASELINE (default hyperparameters).

## Time Budget

Each experiment runs for a fixed number of training steps defined by
`MAX_TRAIN_STEPS` in `experiment.py`. The default is 500 steps, which takes
approximately 25 minutes on a single A100 80GB with DeepSpeed ZeRO-2.

Do not change `MAX_TRAIN_STEPS` unless explicitly told to by the human.
Comparable experiments require identical compute budgets.

## Results Format

`research/results.tsv` is tab-separated with this header:

```
commit	fid	clip_score	disc_acc_real	disc_acc_fake	logit_gap	status	description
```

- `status`: one of `keep`, `discard`, `crash`
- `description`: one-line summary of what changed
