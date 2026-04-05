# LADD Distillation Research

This is an experiment to have the LLM optimize hyperparameters for LADD (Latent Adversarial Diffusion Distillation) of a 6.15B parameter image generation model.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `PROGRESS.md` — training history, what worked, what failed, current best config.
   - `research/experiment.py` — the file you modify. Hyperparameters, architecture knobs, training config.
   - `research/evaluate.py` — fixed evaluation. Do not modify.
   - `research/results.tsv` — past experiment results.
4. **Verify models exist**: Check that `models/Z-Image/` contains the pretrained model and that `data/debug/embeddings/` contains precomputed text embeddings. If not, tell the human.
5. **Verify GPU**: Run `nvidia-smi` to confirm a GPU is available and no stale processes are holding memory. Kill stale python/torch processes if needed.
6. **Initialize results.tsv**: If starting fresh, create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the calibration runs, then the experimentation.

## Calibration runs (do these first)

Previous experiments measured FID, not KID. We need KID baselines before we can compare new results. Run these 3 configs **before** starting the experiment loop. Do NOT change `MAX_TRAIN_STEPS` for any of these — use whatever value is currently set in `experiment.py`. Log results to `results.tsv` as `keep` (these are reference points, not new experiments). Do not discard or reset after these runs.

**Run 1 — Original baseline** (recover KID for the config that scored FID 336.31):
```python
STUDENT_LR = 1e-5
DISC_LR = 1e-4
GEN_UPDATE_INTERVAL = 5
LR_WARMUP_STEPS = 50
WARMUP_SCHEDULE_STEPS = 50
# all other tunable params at defaults
```
Log as: `baseline-recal: slr=1e-5 dlr=1e-4 gi=5 (was FID 336.31)`

**Run 2 — First improvement** (recover KID for the config that scored FID 332.52):
```python
STUDENT_LR = 5e-6
DISC_LR = 1e-4
GEN_UPDATE_INTERVAL = 5
LR_WARMUP_STEPS = 50
WARMUP_SCHEDULE_STEPS = 50
# all other tunable params at defaults
```
Log as: `recal: slr=5e-6 dlr=1e-4 gi=5 (was FID 332.52)`

**Run 3 — Current best** (recover KID for the config that scored FID 313.19 at 2000 steps):
```python
STUDENT_LR = 5e-6
DISC_LR = 5e-5
GEN_UPDATE_INTERVAL = 3
LR_WARMUP_STEPS = 50
WARMUP_SCHEDULE_STEPS = 50
# all other tunable params at defaults
```
Log as: `recal-best: slr=5e-6 dlr=5e-5 gi=3 (was FID 313.19)`

After these 3 runs, the best KID among them becomes your baseline to beat. Reset `experiment.py` to the config from Run 3 (current best), commit, and begin the experiment loop.

## Experimentation

Each experiment runs on a single A100 80GB. The training script runs for a **fixed step budget** (`MAX_TRAIN_STEPS` in `experiment.py`, default 500 steps, ~10 min training + ~5 min evaluation). You launch it simply as: `python research/experiment.py > research/run.log 2>&1`.

**What you CAN do:**
- Modify `research/experiment.py` — this is the only file you edit. The tunable section between `=== TUNABLE HYPERPARAMETERS START ===` and `=== TUNABLE HYPERPARAMETERS END ===` is your playground: learning rates, noise schedule, discriminator architecture, training dynamics, timestep configuration.

**What you CANNOT do:**
- Modify `research/evaluate.py`. It is read-only. It contains the fixed KID evaluation.
- Modify anything under `training/`, `src/`, `scripts/`. The core codebase is frozen.
- Change `MAX_TRAIN_STEPS`, `TRAIN_BATCH_SIZE`, `SEED`, `IMAGE_SIZE`, or `MODEL_PATH`. These are fixed for comparability.
- Install new packages.

**The goal is simple: get the lowest KID (kid_mean).** Since the step budget is fixed, you don't need to worry about training time — it's always the same. Everything in the tunable section is fair game: learning rates, noise schedule parameters, discriminator hidden dim, discriminator layer selection, student timesteps, warmup, gen update interval. The only constraint is that the code runs without crashing and the discriminator stays healthy.

**Discriminator health** is a hard constraint. An experiment is automatically discarded regardless of KID if:
- Any loss is NaN or Inf
- `disc_accuracy_real > 95%` AND `disc_accuracy_fake > 95%` (student stuck, disc too powerful)
- `disc_accuracy_real < 55%` AND `disc_accuracy_fake < 55%` (discriminator collapsed)
- `logit_gap > 15` (discriminator diverging from student)
- `logit_gap < 0.1` after warmup (discriminator provides no learning signal)

**Simplicity criterion**: All else being equal, simpler is better. A small KID improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. Fewer discriminator layers with equal KID? Keep. Removing warmup with equal KID? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Everything runs in a single process. Training computes disc metrics inline, then runs KID evaluation at the final step (inline validation: offloads training models to CPU, generates student images, computes KID against cached teacher images).

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

**2. KID from inline validation** (printed by eval subprocess):
```
[Eval step 500] KID = 0.123456 ± 0.005678
```

**Early stopping**: Training has built-in health checks every 50 steps (after warmup). If disc metrics fail 3 consecutive checks (collapsed, diverging, or stuck), training aborts early and skips KID evaluation entirely. Log these as `discard` and move on — you just saved 10+ minutes on a bad config.

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

Example:

```
commit	kid_mean	peak_vram_gb	status	description
a1b2c3d	0.1500	72.3	keep	baseline: slr=5e-6 dlr=5e-5 gi=3
b2c3d4e	0.1420	72.3	keep	renoise_m=0.5 (was 1.0)
c3d4e5f	0.1600	72.1	discard	renoise_m=0.0 (uniform noise, worse)
d4e5f6g	0.0000	0.0	crash	disc_hidden_dim=1024 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr5`).

LOOP FOREVER:

1. Look at the git state: `git log --oneline -5` and `cat research/results.tsv`
2. Read `research/experiment.py` to see current config
3. Tune `experiment.py` with ONE experimental idea — change one thing in the tunable section
4. git commit: `git add research/experiment.py && git commit -m "<description>"`
5. Run the experiment: `python research/experiment.py > research/run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "early_stopped:\|KID = \|peak_vram_mb:" research/run.log`
7. If early_stopped is True: log as `discard` with kid_mean=0.0 and move on immediately. You just saved 10+ minutes.
8. If the grep output is empty, the run crashed. Run `tail -n 50 research/run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
9. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
10. If kid_mean improved (lower), you "advance" the branch, keeping the git commit
11. If kid_mean is equal or worse, you `git reset --hard HEAD~1` back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Good experiments take ~15 minutes (10 min training + 5 min eval). Bad experiments with early stopping finish in ~2-3 minutes (training aborts, eval is skipped). If a run exceeds 25 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the LADD paper references in the code, re-read PROGRESS.md for clues, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Good experiments take ~15 min, early-stopped bad experiments take ~3 min. With a mix, expect ~6-8 experiments/hour, for a total of about 50-65 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept.

## What to explore

These are the knobs in the tunable section of `experiment.py`, roughly in priority order. Current best values are marked. Unexplored dimensions are the biggest opportunities.

### Priority 1: Noise schedule (UNEXPLORED — start here)
- `RENOISE_M`: {0.0, 0.5, **1.0**, 1.5} — logit-normal mean. Controls which noise levels the discriminator sees. Only default tested.
- `RENOISE_S`: {0.5, **1.0**, 1.5} — logit-normal std. Controls spread of noise sampling. Only default tested.
- The LADD paper specifically tuned these. They directly shape the training signal.

### Priority 2: Student timesteps (UNEXPLORED — product decision)
- `STUDENT_TIMESTEPS`: {[1.0, 0.5], **[1.0, 0.75, 0.5, 0.25]**, [1.0, 0.8, 0.6, 0.4, 0.2]}
- 2-step = 2x faster inference. 5-step = potentially better quality. We've only tried 4-step.

### Priority 3: Training dynamics (partially explored)
- `GEN_UPDATE_INTERVAL`: {1, 2, **3**, 5, 10} — tested 1, 3, 5. Winner is 3. Try 2 and 4 to confirm.
- `LR_WARMUP_STEPS`: {0, **50**, 100, 200} — only 50 tested
- `WARMUP_SCHEDULE_STEPS`: {0, **50**, 100} — only 50 tested

### Priority 4: Discriminator architecture (UNEXPLORED)
- `DISC_HIDDEN_DIM`: {128, **256**, 512} — only 256 tested
- `DISC_LAYER_INDICES`: {**[5,10,15,20,25,29]**, [10,20,29], [3,7,11,15,19,23,27,29]} — only default tested

### Priority 5: Learning rates (mostly explored)
- `STUDENT_LR`: {1e-6, **5e-6**, 1e-5, 2e-5} — well explored, 5e-6 wins
- `DISC_LR`: {**5e-5**, 1e-4, 2e-4} — well explored, 5e-5 wins (10x ratio)
- Only revisit if a noise schedule or architecture change shifts the optimal LR

### Low priority (probably fine at defaults)
- `TEXT_DROP_RATIO`: 0.1 — standard CFG dropout, unlikely to matter much
- `DISC_COND_DIM`: 256 — tied to hidden dim, change only if hidden dim changes

## Infrastructure notes

- Training uses **8-bit Adam** (`--cpu_offload_optimizer`) to fit on a single A100 80GB
- **Precomputed embeddings** (`--embeddings_dir`) skip the ~3GB text encoder during training
- `--skip_save` saves only student weights as safetensors (~12GB), not full optimizer state
- `experiment.py` uses `os.execv` to replace the Python process with bash, avoiding GPU memory leaks from the parent process
- Before each run, check `nvidia-smi` for stale processes — kill them or you'll OOM
- Checkpoint lands at `research/output/checkpoint-{MAX_TRAIN_STEPS}/student_transformer/`
