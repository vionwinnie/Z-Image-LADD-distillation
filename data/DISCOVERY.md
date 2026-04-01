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

## 11. Lessons Learned

1. **Keyword classification is unreliable** for ambiguous prompts. Semantic classification (via LLM) is worth the cost for datasets under 50K.
2. **Generated prompts need length specifications.** Without explicit targets, LLMs default to concise outputs (10-15 words).
3. **Length balancing must be coverage-aware.** Random rejection sampling destroys sparse cells. Stratified approaches preserve the taxonomy while reshaping the distribution.
4. **Source diversity matters more than volume.** 10K well-balanced prompts from 9 sources outperforms 30K dominated by one source.
5. **Chinese prompts need separate handling.** Word-count filters don't apply (Chinese uses characters), and CJK detection is needed for language tagging.

1. **Keyword classification is unreliable** for ambiguous prompts. Semantic classification (via LLM) is worth the cost for datasets under 50K.
2. **Generated prompts need length specifications.** Without explicit targets, LLMs default to concise outputs (10-15 words).
3. **Length balancing must be coverage-aware.** Random rejection sampling destroys sparse cells. Stratified approaches preserve the taxonomy while reshaping the distribution.
4. **Source diversity matters more than volume.** 10K well-balanced prompts from 9 sources outperforms 30K dominated by one source.
5. **Chinese prompts need separate handling.** Word-count filters don't apply (Chinese uses characters), and CJK detection is needed for language tagging.
