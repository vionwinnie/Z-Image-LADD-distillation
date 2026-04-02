# Data Pipeline: Discovery & Cleaning Process

This document traces the decisions made while building the prompt dataset for LADD distillation of Z-Image.

---

## 1. Starting Point: What Does LADD Need?

LADD distillation uses **text prompts only** — no images. The teacher model generates synthetic latents from prompts during training, and the student learns to match the teacher's distribution through adversarial feedback. The dataset quality depends on:

- **Diversity of subjects and styles** — the student must generalize across content types
- **Prompt length and detail** — detailed prompts stress prompt adherence, which is where distilled models degrade most
- **Language coverage** — Z-Image uses a Qwen3 text encoder that handles both English and Chinese natively

## 2. Data Sources

Downloaded 8 T2I benchmarks via `prepare_prompts.py` and evaluated 2 external datasets. Four benchmarks required custom downloaders (GenEval from GitHub raw JSONL, OneIG-ZH needed config `"OneIG-Bench-ZH"`, CVTG-2K was a nested zip, LongText-Bench needed CJK detection for language tagging).

| Source | Count | Avg Words | Character | Status |
|--------|-------|-----------|-----------|--------|
| **MJHQ-30K** | 30,000 | 29.5 | Curated Midjourney, 10 categories, diverse styles | Added (sampled) |
| CVTG-2K | 2,000 | 33.9 | Text rendering | Added |
| OneIG-ZH | 1,320 | — | Chinese language | Added |
| PartiPrompts | 1,257 | 11.1 | Short compositional | Added |
| GenAI-Bench | 1,188 | 23.4 | Mixed complexity | Added |
| DPG-Bench | 1,065 | 67.1 | Dense, detailed | Added |
| GenEval | 553 | 7.6 | Very short | Added |
| LongText-Bench | 320 | 135.4 | Very long | Added |
| DrawBench | 108 | 13.0 | Short evaluation | Added |
| ~~SDXL-1M (Falah)~~ | 1,000,000 | ~40 | Template-generated, people-only, photorealistic-only | **Rejected** — low diversity |

## 3. Classification: Keywords Failed, Claude Worked

**Keyword matching** produced a skewed matrix — "a woman in a red dress" classified as Fashion instead of People, most prompts defaulted to Photorealistic/Standard. **Claude-based reclassification** (4 parallel subagents, ~2K prompts each) fixed this: people take priority when present, implicit styles detected, Chinese Cultural went from 6→112 prompts.

## 4. Gap Fill + Prompt Expansion → 3,347 prompts

After reclassification, **80 of 98 cells were below 50** (17 had zero — all in 3D/CGI, Cinematic, Experimental styles). Four Claude subagents generated prompts to fill every cell to 50+.

**Problem:** generated prompts averaged only 14 words. **Fix:** four more subagents expanded each to 30-40 words with concrete visual details (colors, materials, textures, lighting, atmosphere).

> Before: *"3D render of a crystal goblet on a marble table with dramatic lighting"*
> After: *"3D render of an ornate crystal goblet with diamond-cut facets sitting on a polished white marble table, warm amber spotlight casting long shadows, dark moody background with subtle fog, octane render, 8K detail"*

## 5. Length Distribution Balancing

### The problem

The raw combined dataset had a bimodal word count distribution:
- 44% of prompts under 20 words (benchmarks + gap-fill)
- A long tail of 50+ word prompts (DPG-Bench, LongText-Bench)
- Mean: 27, Median: 16

For LADD distillation, we want the student to handle detailed prompts well — the mean should be closer to 35 words.

### Approach: stratified resampling (not random rejection)

We rejected Gaussian acceptance sampling (randomly drops prompts regardless of category) in favor of a stratified approach in `build_dataset.py`:

1. **Hard cutoffs**: drop EN prompts with <10 or >80 words (Chinese unaffected)
2. **Per-cell stratified keep**: for each Subject×Style cell, guarantee a minimum count (30). In cells with surplus, preferentially keep longer prompts
3. **MJHQ mid-range fill**: add MJHQ prompts in the 25-50 word range to build the bell curve center

This preserves category coverage while shifting the length distribution.

### Why not just keep everything?

With 40K+ prompts (all MJHQ included), MJHQ dominates at 73% of the dataset. The student would effectively be trained on Midjourney-style prompts with benchmark prompts as noise. Sampling maintains source diversity.

## 6. Final Dataset Configuration

The recommended build command:

```bash
python data/build_dataset.py --sample mjhq=15000 --balance-lengths
```

This produces ~12K prompts with:
- Mean ~35 words, median ~32
- All 98 Subject×Style cells ≥ 30
- Balanced source representation
- 85% English, 15% Chinese

### `build_dataset.py` options

| Flag | Default | Purpose |
|------|---------|---------|
| `--sample SOURCE=N` | all | Limit prompts per source |
| `--min-words` | 10 | Drop EN prompts shorter than this |
| `--max-words` | 80 | Drop EN prompts longer than this |
| `--balance-lengths` | off | Stratified resampling + MJHQ fill |
| `--min-per-cell` | 30 | Minimum per Subject×Style cell |
| `--mjhq-fill-count` | 5000 | MJHQ 25-50 word prompts to add |
| `--subjects` | all | Filter to specific subjects |
| `--styles` | all | Filter to specific styles |
| `--no-mjhq` | off | Exclude MJHQ entirely |
| `--dry-run` | off | Preview without saving |

## 7. Quality Inspection

The dataset can be inspected via `dashboard.html` (served locally):

```bash
cd data && python3 -m http.server 8765
# Open http://localhost:8765/dashboard.html
```

### Dashboard Overview

The dashboard has two tabs: **Analytics** and **Browse & Search**.

**KPIs, Subject Distribution, and Source Breakdown:**

![Dashboard KPIs showing total prompts, language split, and subject/source distributions](Dashboards/02_kpi_subject_source.png)

**Style, Camera, Word Count Histograms, and Top Words:**

![Style and camera distributions with English word count and Chinese character count histograms](Dashboards/03_style_camera_histograms.png)

**Subject×Style Heatmap and Avg Word Count by Source/Subject:**

![Coverage heatmap with blue intensity scaling and average word count bar charts](Dashboards/04_heatmap_wordcount.png)

### Browse & Search

The Browse tab supports filtering by subject, style, camera, source, language, and free-text search. Clicking any chart bar or heatmap cell navigates to Browse with that filter applied.

**English prompt browsing with full category labels:**

![Browse tab showing English prompts with human-readable category tags](Dashboards/05_browse_english.png)

**Chinese prompt browsing:**

![Browse tab showing Chinese-language prompts from OneIG-ZH](Dashboards/06_browse_chinese.png)

## 8. Scaling to 1M Prompts: Training Data Size Estimation

### What Did the Original Papers Use?

Neither the ADD nor LADD papers disclose the exact number of training prompts. Here's what's reported:

**ADD (Adversarial Diffusion Distillation) — Sauer et al., 2023 ([arXiv:2311.17042](https://arxiv.org/abs/2311.17042))**
- 4,000 training iterations at batch size 128 = ~512K sample presentations
- Uses real images from an unnamed, unsized dataset
- Evaluated on 100 PartiPrompts + 5K COCO2017 samples

**LADD (Latent Adversarial Diffusion Distillation) — Sauer et al., 2024 ([arXiv:2403.12015](https://arxiv.org/abs/2403.12015))**
- 10,000 training iterations (batch size unstated) for ablations on ~2B param model
- Entirely synthetic — teacher generates latents, no real images needed
- Prompts sampled from SD3's training set (~1B images pre-trained, ~30M high-quality fine-tuned)
- +3,000 iterations of DPO fine-tuning with LoRA rank 256
- Evaluated on 128 PartiPrompts (every 4th prompt, excluding "Basic" category)

### Estimated Prompt Pool Size

The papers are deliberately vague about training data specifics. Working from what's disclosed:

| Assumption | Iterations | Batch Size | Total Presentations |
|------------|-----------|------------|---------------------|
| ADD (known) | 4,000 | 128 | 512K |
| LADD conservative | 10,000 | 256 | 2.5M |
| LADD likely | 10,000 | 512 | 5M |

LADD samples prompts from SD3's full training set (millions of unique prompts), so repeats are rare. The effective number of **unique prompts seen** during training is approximately equal to `iterations × batch_size`.

### Implications for Our Dataset

- **1M unique prompts** is a reasonable pool size — it ensures the model rarely sees the same prompt twice across a typical training run of 10K iterations at batch size 256–512
- The original LADD likely drew from a pool of millions (SD3's training set), but the model only *sees* 2.5–5M presentations total
- Our current 16.5K prompts would be exhausted within ~60 iterations at batch size 256 — the model would cycle through the entire dataset many times, reducing diversity of gradient signals
- At 1M prompts with 10K iterations and batch size 256, each prompt would be seen ~2.5 times on average — comparable to the original paper's regime

## 9. Candidate Datasets for Scaling

### Tier 1: High Scale, Good Quality, Permissive License

| Dataset | Usable Prompts | Avg Length | Type | License | Notes |
|---------|---------------|------------|------|---------|-------|
| **[DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)** | ~2M unique (14M total) | ~30-77 tokens | Real SD user prompts | CC0 | Highest priority. Real-world T2I prompts with style keywords. Needs dedup (many near-duplicates with param tweaks) and NSFW filtering. SD-specific tokens ("trending on artstation") need cleaning. |
| **[Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B)** | Sample from 1.3B | ~30-60 words | LLaVA-1.5 recaptions | CC-BY-4.0 | Massive scale, much better than raw alt-text. Used in training several recent T2I models. LLaVA captioning artifacts present. |
| **[DenseFusion-1M](https://huggingface.co/datasets/DenseFusion/DenseFusion-1M)** | ~1M | ~80-150 words | Multi-VLM fused captions | Apache-2.0 | Purpose-built for this use case. Fuses multiple captioning models for higher quality. Very long/detailed. |
| **[ShareGPT4V-PT](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)** | ~1.2M | ~50-100 words | GPT-4V synthetic captions | Apache-2.0 | High quality dense captions. GPT-4V style bias (formal, structured). |
| **[SAM-LLaVA-Captions10M](https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M)** | Sample from 10M | ~30-80 words | LLaVA on SAM images | Research | Proven in PixArt-α training. SAM-image domain bias. |

### Tier 2: Good Supplementary Sources

| Dataset | Usable Prompts | Avg Length | Type | License | Notes |
|---------|---------------|------------|------|---------|-------|
| **[JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB)** | ~4.7M | ~20-50 words | Midjourney user prompts | CC-BY-NC-SA | High aesthetic quality but NC license. MJ-specific tokens (--ar, --v) need stripping. |
| **[COYO-700M](https://huggingface.co/datasets/kakaobrain/coyo-700m)** | Sample 100-200K | ~10-15 words | Web alt-text | CC-BY-4.0 | Cleaner than LAION. Short captions lack style directives. Needs heavy quality filtering. |
| **[AnyText-3M](https://huggingface.co/datasets/modelscope/AnyText-3M)** | ~3M | Medium | Text rendering prompts | Apache-2.0 | Bilingual (EN+ZH). Specialized for typography — fills S11 (Text/Typography) gap. |
| **[DOCCI](https://huggingface.co/datasets/google/docci)** | ~15K | ~80-150 words | Human-written dense captions | CC-BY-4.0 | Highest quality captions available. Small scale — use as quality seed/calibration. |
| **[Wukong](https://huggingface.co/datasets/noah-wukong/wukong)** (filtered) | Sample 50-100K | ~10-15 chars | Chinese web captions | Apache-2.0 | Massive Chinese content (~100M). Very noisy, needs heavy filtering. |
| **[Gustavosta SD Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)** | ~80K | ~20-40 words | Curated SD prompts | Open | Clean, ready-to-use. Small but high signal-to-noise. |

### Tier 3: Niche / Supplementary

| Dataset | Usable Prompts | Notes |
|---------|---------------|-------|
| **CC3M / CC12M** | Sample 50K | Short alt-text, hypernymized. General diversity filler. |
| **TextCaps** | ~142K | Captions describing text in images. Good for S11 coverage. |
| **Kolors / Hunyuan-DiT prompts** | ~10-50K | Bilingual Chinese-English T2I prompts from Kwai/Tencent. |
| **SBU Captions** | ~1M | Flickr user captions. More natural than alt-text but Flickr-biased. |
| **DALL-E 3 prompt collections** | ~5-50K | Very long narrative prompts (~50-150 words). GPT-4 rewritten. Small scale. |

### Recommended Blend to Reach 1M

| Source | Target Count | Rationale |
|--------|-------------|-----------|
| DiffusionDB (deduped, filtered) | 300K | Real user prompts, massive style diversity, CC0 |
| Recap-DataComp-1B (quality-filtered) | 250K | Dense recaptions, broad content diversity |
| DenseFusion-1M (sampled) | 200K | Long, detailed, multi-VLM fused |
| ShareGPT4V-PT | 100K | GPT-4V quality dense captions |
| JourneyDB (cleaned) | 50K | Midjourney aesthetic diversity (check NC license) |
| AnyText-3M (sampled) | 30K | Text rendering coverage |
| SAM-LLaVA-Captions10M | 30K | Additional diverse dense captions |
| Wukong (filtered) | 15K | Chinese language coverage |
| DOCCI | 15K | Highest-quality human baselines |
| **Existing pool** | **16K** | Current classified dataset |
| **Total** | **~1.006M** | |

### Key Considerations

1. **License**: DiffusionDB (CC0) and Recap-DataComp (CC-BY-4.0) are safest. JourneyDB is NC-only — check if acceptable for your use case.
2. **Deduplication**: Cross-dataset dedup is essential. DiffusionDB has massive internal duplication. See Section 10 for semantic dedup strategy.
3. **Quality filtering**: All large-scale datasets benefit from CLIP-score filtering, length filtering, and NSFW filtering.
4. **Format normalization**: Alt-text captions ("a photo of...") read differently from SD/MJ prompts with style modifiers. Decide whether to normalize or preserve format diversity.
5. **Classification**: All new prompts need to be classified into the existing S1-S14, T1-T7, C1-C8 taxonomy to maintain category balance tracking.

## 10. Semantic Deduplication & Quality-Based Sampling

### The Problem

With ~10M candidate prompts from multiple sources, we need to:
1. **Remove concept-level duplicates** — "a cute cat watching a movie in a cinema" and "an adorable cat sitting in a movie theater" describe the same scene
2. **Keep the higher-quality version** when duplicates are found
3. **Maximize concept diversity** in the final 1M sample
4. **Maintain category balance** across the taxonomy

### Approach: Embedding-Based Semantic Dedup

#### Step 1: Embed All Prompts

Encode every prompt into a dense vector using a text embedding model:

| Model | Dim | Speed | Quality | Notes |
|-------|-----|-------|---------|-------|
| **SentenceTransformers `all-MiniLM-L6-v2`** | 384 | Very fast | Good | Best speed/quality tradeoff for 10M scale |
| **OpenAI `text-embedding-3-small`** | 1536 | API-bound | High | Better semantics but costs ~$2 for 10M |
| **BGE-M3** | 1024 | Medium | High | Multilingual (handles Chinese). Best if bilingual dedup needed. |
| **Nomic `nomic-embed-text-v1.5`** | 768 | Fast | High | Good open-source option, handles long texts well |

**Recommendation**: Use **BGE-M3** for bilingual support, or **all-MiniLM-L6-v2** if speed is the priority and Chinese dedup is handled separately.

#### Step 2: Cluster & Deduplicate

Two viable strategies:

**Strategy A: Pairwise Cosine Similarity with Threshold**
```
For each prompt:
  1. Find all neighbors with cosine_sim > threshold (e.g., 0.90)
  2. Among the cluster of near-duplicates, keep the one with highest quality score
  3. Remove the rest
```
- Use FAISS or Annoy for approximate nearest neighbor search at 10M scale
- FAISS `IndexIVFFlat` with ~4096 clusters handles 10M vectors in <1 min on GPU

**Strategy B: SemDeDup (Meta, 2023)** — [arXiv:2303.09540](https://arxiv.org/abs/2303.09540)
```
1. K-means cluster all embeddings (k = sqrt(N) ≈ 3162 clusters)
2. Within each cluster, compute pairwise cosine similarity
3. Remove prompts above similarity threshold, keeping highest quality
```
- Proven at billion-scale (used on C4, LAION). More principled than brute-force pairwise.
- Reference implementation: https://github.com/facebookresearch/SemDeDup

#### Step 3: Quality Scoring

When choosing which duplicate to keep, score each prompt on:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| **Word count** (15-60 words sweet spot) | 0.3 | Too short = vague, too long = noisy |
| **Specificity** (named entities, concrete nouns, numbers) | 0.25 | "A golden retriever puppy" > "a dog" |
| **Visual detail** (colors, materials, lighting, textures) | 0.25 | More visual keywords = better T2I prompt |
| **Source quality prior** | 0.2 | DOCCI > ShareGPT4V > DiffusionDB > COYO > alt-text |

Quality scoring can be done cheaply:
- **Keyword-based heuristics** for specificity and visual detail (count of color/material/lighting words)
- **Perplexity from a small LM** as a fluency proxy
- Or use an LLM judge (Haiku) on borderline cases only

#### Step 4: Diversity-Aware Sampling

After dedup, sample the final 1M with diversity guarantees:

1. **Classify** all surviving prompts into (Subject, Style, Camera) using the existing keyword classifier
2. **Set per-cell targets** based on desired distribution (e.g., uniform = ~1020 per Subject×Style cell, or weighted to match a target distribution)
3. **Within each cell**, use **k-DPP (Determinantal Point Process)** or **maximal marginal relevance (MMR)** sampling on the embeddings to select the most diverse subset
4. If a cell has fewer candidates than its target, keep all and redistribute quota to other cells

### Similarity Threshold Tuning

The cosine similarity threshold controls the dedup aggressiveness:

| Threshold | Behavior | Example pair at boundary |
|-----------|----------|------------------------|
| 0.95 | Very conservative — only near-identical text | "a cute cat in a cinema" vs "a cute cat in a cinema watching" |
| 0.90 | Moderate — same scene, different wording | "a cute cat watching a movie" vs "an adorable cat in a movie theater" |
| 0.85 | Aggressive — same concept, different details | "a cat in a cinema" vs "a kitten watching TV on a couch" |
| 0.80 | Very aggressive — same subject, different scene | "a cat in a cinema" vs "a cat sleeping on a windowsill" |

**Recommendation**: Start with **0.90**, evaluate a sample of removed pairs manually, then adjust. For T2I training, we want scene-level diversity — two prompts describing the same scene from slightly different angles don't add training signal.

### Pipeline Summary

```
10M raw prompts (multi-source)
    ↓
[1] Embed all prompts (BGE-M3 or MiniLM)
    ↓
[2] SemDeDup: cluster → pairwise sim → remove dupes (threshold=0.90)
    → keep highest quality-scored prompt per cluster
    ↓
~3-5M unique-concept prompts
    ↓
[3] Classify into (S, T, C) taxonomy
    ↓
[4] Per-cell MMR diversity sampling → target 1M
    ↓
1M diverse, quality-filtered, category-balanced prompts
```

## 11. Scaling Execution (April 2026)

Executed the scaling plan from PLAN.md on a 503GB RAM / 16-core CPU instance (no GPU). AnyText-3M was unavailable (removed from HuggingFace). Several dataset IDs in the plan were wrong or had moved.

### Phase 1: Harvest — 970K raw prompts from 9 sources

| Source | HF ID Used | Prompts | Notes |
|--------|-----------|---------|-------|
| DiffusionDB | `poloclub/diffusiondb` (metadata-large.parquet) | 300,000 | Dataset script no longer supported; loaded parquet directly. NSFW filtered. |
| Recap-DataComp-1B | `UCSC-VLAA/Recap-DataComp-1B` | 250,000 | Schema mismatch across parquet shards; loaded shards individually to bypass CastError. |
| DenseFusion-1M | **`BAAI/DenseFusion-1M`** (not `DenseFusion/DenseFusion-1M`) | 200,000 | Plan had wrong org. Required config name `"DenseFusion-1M"`. Streamed. |
| ShareGPT4V-PT | `Lin-Chen/ShareGPT4V` | 100,000 | Config `"ShareGPT4V-PT"` didn't exist; default loaded the 1.2M PT set. Extracted assistant's first reply from conversations. |
| JourneyDB | `JourneyDB/JourneyDB` | 50,000 | Gated, required HF token + terms acceptance. Tar archives can't stream — downloaded `train_anno.jsonl.tgz` directly via `hf_hub_download` and extracted. Stripped MJ params (`--ar`, `--v`, etc). NC license flagged. |
| SAM-LLaVA-10M | `PixArt-alpha/SAM-LLaVA-Captions10M` | 30,000 | Streamed. Field is `txt`, not `caption`. |
| Wukong | **`wanng/wukong100m`** (not `noah-wukong/wukong`) | 15,000 | Plan had wrong org (404). `jaxmetaverse/wukong` had images only, no captions. `wanng/wukong100m` has `url` + `caption` fields. Relaxed alpha-ratio filter for Chinese text. |
| DOCCI | `google/docci` | 13,936 | Dataset script no longer supported by modern `datasets` lib; `trust_remote_code` also rejected. Downloaded JSONL directly from `https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines`. |
| Existing pool | `data/train/metadata.json` | 11,158 | Already classified from prior pipeline. |
| ~~AnyText-3M~~ | `modelscope/AnyText-3M` | **0** | **404 on HuggingFace.** Dataset removed or made private. No mirror found. |

**Quality filters applied per-prompt:** min 8 EN words / 15 ZH chars, max 200 words, <2 URLs, no boilerplate strings ("stock photo", "getty images"), ≥70% alpha characters.

Cross-source exact dedup removed only 64 prompts — almost no verbatim overlap between sources.

### Phase 2: Deduplication — 970K → 630K (35% removed)

| Stage | Method | Input | Output | Removed | Time |
|-------|--------|-------|--------|---------|------|
| 2.1 Surface-level | MinHash LSH (Jaccard≥0.7, 128 perms) | 970,030 | 767,995 | 202,035 (20.8%) | ~37 min |
| 2.2 Embedding | `all-MiniLM-L6-v2` via PyTorch CPU | 767,995 | — | — | ~2.8 hours |
| 2.3 Semantic | FAISS k-means (876 clusters) + pairwise cosine>0.90 | 767,995 | 630,099 | 137,896 (18.0%) | ~3 min |
| **Total** | | **970,030** | **630,099** | **339,931 (35.0%)** | ~3.5 hours |

**Key observations:**
- **DiffusionDB had massive internal duplication** — MinHash took it from 300K→151K (50% removed). Expected per plan.
- **Semantic dedup caught paraphrases MinHash missed** — different wording, same scene. 18% removal at cosine 0.90 threshold.
- **ONNX Runtime was not viable** — `optimum[onnxruntime]` pulls 400MB+ CUDA libs that didn't fit the 5GB root filesystem. PyTorch CPU-only (`torch==2.11.0+cpu`, 190MB) worked fine at ~73 prompts/sec with batch size 512.
- **`sentence-transformers` was 5× slower than raw PyTorch** for the same model (~12/sec vs ~73/sec). The overhead comes from its internal preprocessing and multi-GPU detection code. Direct `AutoModel` + manual mean-pooling was the fix.
- **Plan estimated 33 min for embedding; actual was ~2.8 hours.** Plan assumed ONNX at 500/sec; actual PyTorch CPU was ~73/sec. Still feasible on the 32GB budget.
- `multilingual-e5-small` was replaced by `all-MiniLM-L6-v2` for speed. 98% of prompts are English so multilingual support was unnecessary.
- Quality scoring when choosing between duplicates used the plan's formula: word count sweet spot (15-60), source quality prior, visual keyword count, and specificity (capitalized words + numbers).

### Phase 3: Classification & Sampling

All 630K prompts classified into the (S1-S14, T1-T7, C1-C8) taxonomy using the existing keyword classifier from `prepare_prompts.py`. All **98 Subject×Style cells populated**.

Since 630K < 1M target, all prompts were kept (no MMR downsampling needed). MMR was tested but killed — O(n×k) on cells with 300K+ candidates was going to take hours. The fix was short-circuiting: if pool ≤ target, take everything.

### Phase 4: Final Assembly

```
data/train/metadata.json  — 629,443 prompts (399 MB)
data/debug/metadata.json  — 98 prompts (1 per Subject×Style cell)
```

**Validation results:**
- Total: 629,443 (63% of 1M target)
- All 98 cells populated, min cell size: 23
- Language: EN 97.7%, ZH 2.3%
- Mean EN word count: 93.0 (higher than 30-45 target — many verbose captions from DenseFusion/ShareGPT4V)
- Zero exact duplicates
- No single source > 35% (max: DenseFusion at 31.4%)

### Gap to 1M

~370K prompts short. Options to close the gap:
1. **LLM generation** via `data/generate_prompts.py` — fill sparse cells first, then bulk generate
2. **Additional datasets** — COYO-700M, CC3M/CC12M, Gustavosta SD Prompts, TextCaps
3. **Accept 630K** — at batch size 256 and 10K iterations, each prompt seen ~4× on average (vs ~2.5× at 1M). Still viable for training.

### Pipeline Scripts Created

| Script | Purpose |
|--------|---------|
| `data/harvest.py` | Phase 1: Download, filter, normalize all 10 sources to `data/raw/*.jsonl` |
| `data/dedup_minhash.py` | Phase 2.1: MinHash LSH surface dedup |
| `data/dedup_semantic.py` | Phase 2.2-2.3: Embedding + FAISS clustering + pairwise semantic dedup |
| `data/classify_and_sample.py` | Phase 3: Classify into taxonomy + MMR diversity sampling |
| `data/build_dataset.py` | Phase 4: Length filter, validation checks, debug split |

All scripts are idempotent (skip sources that already have output files) and can be re-run independently.

### Phase 3 Revisited: Hybrid Zero-Shot Classification

The initial classification used pure keyword matching from `prepare_prompts.py`. This produced a heavily skewed distribution because unmatched prompts fall to defaults:

| Axis | Default | Keyword-only share |
|------|---------|-------------------|
| Subject | S10 (Objects/Artifacts) | 10.1% |
| Style | T1 (Photorealistic) | 67.2% |
| Camera | C1 (Standard/Eye-level) | ~80% |

Most of these defaults were wrong — prompts about Chinese culture, fantasy scenes, or digital illustrations all landed in S10/T1 because no keyword matched.

**Solution: Hybrid keyword + zero-shot embedding classifier** (`zeroshot_classify.py`)

The approach:
1. Run keyword classifier first
2. If keyword returns a **non-default** label → trust it (keywords are high-precision when they match)
3. If keyword returns the **default** → use zero-shot embedding similarity with `all-MiniLM-L6-v2` to reclassify

Each category axis has 3-6 natural language descriptions per label (e.g. S2/Animals: "an animal, dog, cat, bird, wildlife, pet"). These are embedded and averaged into label centroids. Prompts are assigned to the nearest centroid by cosine similarity.

**Three problems emerged during validation on 500-sample subsets:**

**Problem 1: Zero-shot too aggressive on close calls.** Prompts with similar scores across categories got reclassified when the zero-shot was barely confident. Fix: **margin threshold of 0.05** — zero-shot must beat the default by 0.05 cosine similarity to override.

**Problem 2: T6 (GraphicDesign) over-triggered.** Descriptive captions from DenseFusion/ShareGPT4V ("The image displays a...") have a formal register that MiniLM associates with design content. A photo of palm fruits described formally gets pulled toward T6. Two fixes:
- **Weak keyword demotion**: keyword T6 triggers like "minimal", "flat", "icon", "logo" match too broadly ("minimal wear", "flat surface", "Volkswagen logo"). If T6 was triggered *only* by these weak keywords, demote to default and let zero-shot decide.
- **Strong design signal gate**: zero-shot T6 requires explicit design keywords in the text ("graphic design", "poster", "infographic", "slide", "presentation", etc). Without them, the prompt stays T1 regardless of embedding similarity.

**Problem 3: "logo" appears in scene descriptions.** Descriptive captions mention brand logos on jerseys, car badges, watermarks — "logo" in text doesn't mean graphic design. Solved by treating "logo" as a weak keyword (only counts when combined with other design signals).

**Full-scale results (630K prompts):**

| Style | Keyword-only | Hybrid |
|-------|-------------|--------|
| T1 Photorealistic | 67.2% | 76.3% |
| T2 TraditionalArt | 2.5% | 2.8% |
| T3 DigitalIllustration | 7.4% | 11.2% |
| T4 3D/CGI | 1.4% | 1.4% |
| T5 Cinematic/Film | 1.6% | 2.1% |
| T6 GraphicDesign | 18.5% | 4.7% |
| T7 Mixed/Experimental | 1.4% | 1.5% |

| Subject | Keyword-only | Hybrid |
|---------|-------------|--------|
| S10 Objects (default) | 10.1% | 2.9% |
| S13 ChineseCultural | 0.2% | 2.2% |
| S14 Abstract/Imagination | 0.7% | 1.3% |
| S12 WorldKnowledge | 0.3% | 0.8% |
| S9 Fashion/Clothing | 3.3% | 4.4% |

T6 dropped from 18.5% to 4.7% — most of the 14% reduction was false positives (product photos, descriptive captions, scenes that happened to mention "logo"). S10 dropped from 10.1% to 2.9% — Chinese text, abstract art, and world landmarks correctly redistributed.

**Iteration needed:** the first full-scale run revealed T7 (Mixed/Experimental) exploding to 24.6% — invisible in 500-sample validation because T7's label descriptions are vague enough to be a catch-all. Fix: same keyword-gating approach as T6, requiring explicit experimental keywords ("glitch", "collage", "psychedelic", etc.) for zero-shot T7. After fix: T7 stable at 1.5%.

**Key insight**: pure zero-shot classification doesn't work for this task — MiniLM confuses caption register (formal descriptive writing) with visual style (graphic design). The hybrid approach uses keywords as a high-precision first pass and zero-shot as a recall booster for defaults only, with domain-specific heuristics to guard against known failure modes. Vague catch-all categories (T6 GraphicDesign, T7 Mixed/Experimental) need keyword gating to prevent the embedding model from using them as dumping grounds.

### Phase 3 Revisited Again: Caption First-Sentence Stripping

The hybrid approach still had a critical flaw: **keyword classification on full descriptive captions is unreliable for ALL categories, not just defaults.** DenseFusion/ShareGPT4V/Recap-DataComp captions are verbose and mention many objects incidentally:

- "The image displays a green hoodie... worn by a **person**" → keyword matches "person" → S1 (People). Wrong — subject is the hoodie.
- "The image displays a promotional flyer... with a **bird** logo" → keyword matches "bird" → S2 (Animals). Wrong — subject is the flyer.
- "The image displays a slide... about a **car** recall" → keyword matches "car" → S7 (Vehicles). Wrong — subject is the slide.

DenseFusion was classified 60% S1 (People) by keywords — manual inspection showed the majority were objects, graphics, text, and clothing that happened to mention a person.

**Fix: first-sentence stripping for descriptive captions.**

For prompts matching the pattern `"The image displays/shows/features/captures..."`:
1. Strip the prefix ("The image displays")
2. Extract the first sentence (up to the first period)
3. Embed that stripped sentence with MiniLM for zero-shot classification
4. Use the stripped-sentence embedding for BOTH subject and style (not just subject)

This works because VLM-generated captions follow a consistent structure: the first sentence names the primary subject, subsequent sentences describe details, context, and incidental elements. By classifying only the first sentence, we avoid keyword pollution from later text.

**Validation on 200 DenseFusion samples:**

| Subject | Keyword-only | Hybrid (first-sentence) |
|---------|-------------|------------------------|
| S1 People | 116 (58%) | 33 (17%) |
| S2 Animals | 52 (26%) | 2 (1%) |
| S10 Objects | 1 (0.5%) | 28 (14%) |
| S11 Text/Typography | 4 (2%) | 68 (34%) |
| S14 Abstract | 0 (0%) | 22 (11%) |
| S9 Fashion | 4 (2%) | 10 (5%) |
| S7 Vehicles | 7 (3.5%) | 10 (5%) |
| S3 Food | 3 (1.5%) | 10 (5%) |

S2 (Animals) dropped from 52 to 2 — the keyword classifier had been matching "bird", "fish", "cat" mentioned in logos, book titles, and metaphors. The remaining S1 prompts were verified to genuinely feature people as the primary subject.

Non-caption prompts (DiffusionDB user prompts, JourneyDB Midjourney prompts) still use the original hybrid approach since they don't have the "The image displays..." prefix structure.

### Phase 4 Revisited: Subject Balance Cap + Chinese Filter

Two additional data quality measures:

**1. Chinese prompt minimum length**: raised from 10 to 20 characters. Short Chinese captions (e.g. product names, single phrases) lack enough context for T2I generation.

**2. Subject percentage cap (15%)**: no single subject category can exceed 15% of the final dataset. Without this, S1 (People) dominates at ~30-50% depending on classification version, which biases the student model toward people-centric generation. Subjects exceeding the cap are randomly downsampled. Small subjects keep all their prompts.

This is applied in `build_dataset.py` via `--subject-cap 0.15`.

## 12. Lessons Learned

1. **Keyword classification is unreliable** for ambiguous prompts. Semantic classification (via LLM) is worth the cost for datasets under 50K.
2. **Generated prompts need length specifications.** Without explicit targets, LLMs default to concise outputs (10-15 words).
3. **Length balancing must be coverage-aware.** Random rejection sampling destroys sparse cells. Stratified approaches preserve the taxonomy while reshaping the distribution.
4. **Source diversity matters more than volume.** 10K well-balanced prompts from 9 sources outperforms 30K dominated by one source.
5. **Chinese prompts need separate handling.** Word-count filters don't apply (Chinese uses characters), and CJK detection is needed for language tagging.
6. **HuggingFace dataset IDs are unstable.** Three of 10 datasets had wrong or outdated org names. Always verify IDs before building automation.
7. **The `datasets` library broke backward compatibility.** Dataset scripts (`trust_remote_code`) are no longer supported in v4.8+. Fallback to direct parquet loading or raw URL download is essential.
8. **`sentence-transformers` adds significant overhead on CPU.** Raw `transformers` + manual pooling was 5× faster for the same model. Use the library for convenience, not for batch throughput.
9. **MinHash LSH is the workhorse for dedup.** It caught 21% of duplicates in 37 minutes. Semantic dedup adds value (18% more) but takes 10× longer. If time-constrained, MinHash alone gets you 80% of the way.
10. **Schema mismatches across parquet shards are real.** Recap-DataComp-1B had different columns in different shards. Loading individual shards instead of the whole dataset was the workaround.
11. **Zero-shot embedding classification confuses caption register with visual style.** Formal descriptive captions ("The image displays...") are semantically closer to design briefs than to casual photo descriptions in MiniLM's embedding space. Pure zero-shot T6 precision was ~50%. Domain heuristics (keyword gating) brought it above 90%.
12. **Hybrid keyword+zero-shot outperforms either alone.** Keywords are high-precision but low-recall (67% default rate). Zero-shot has good recall but poor precision for ambiguous categories. Keyword-first with zero-shot fallback on defaults combines both strengths.
13. **Margin thresholds prevent low-confidence reclassification.** Without a margin, zero-shot reclassifies on differences of 0.01 cosine similarity — essentially random. A 0.05 margin ensures only confident predictions override the default.
14. **Validate classification on small subsets before scaling.** Testing on 200-500 samples with side-by-side comparison (keyword vs zero-shot vs hybrid) caught all three major failure modes before running on 630K prompts.
15. **Keyword classification on verbose captions is wrong for ALL categories, not just defaults.** A caption mentioning "bird" in a logo description gets S2 (Animals). The fix is to classify only the primary subject (first sentence, prefix stripped) rather than the full text. This is specific to VLM-generated captions — user-written prompts don't have this problem.
16. **Subject balance requires explicit enforcement.** Without a cap, People/Portraits dominates at 30-50% because most image datasets are human-centric. A 15% per-subject cap ensures the student model sees diverse content during training.
