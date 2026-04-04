# Training Data

## Quick Start

`data/train/metadata.json` (501K prompts), `data/val/metadata.json` (13K), and `data/test/metadata.json` (13K) are stored with **Git LFS**. After cloning:

```bash
# Install git-lfs if you don't have it
# macOS: brew install git-lfs
# Ubuntu: apt-get install git-lfs

git lfs install
git lfs pull
```

If you cloned before LFS was set up, or the file shows as a small pointer:

```bash
git lfs fetch
git lfs checkout
```

Verify it downloaded correctly:

```bash
python3 -c "import json; d=json.load(open('data/train/metadata.json')); print(f'{len(d)} prompts')"
# Expected: 500569 prompts
```

## Data Format

Text-only prompts stored as a JSON array:

```json
[
  {
    "text": "A red sports car on a cobblestone street",
    "subject": "S7",
    "style": "T1",
    "camera": "C1",
    "source": "diffusiondb",
    "lang": "en"
  }
]
```

The training script only uses the `text` field. Other fields (`subject`, `style`, `camera`, `source`, `lang`) are for taxonomy tracking.

## Why Text-Only (No Images)

LADD distillation does not use real images. The teacher model generates synthetic latents from prompts during training. The student learns to match the teacher's distribution through adversarial feedback, not by training on real image-text pairs.

## Data Sources (527K prompts before split)

Harvested from 9 datasets via `scripts/harvest.py`, deduplicated, and classified:

| Source | Count | HuggingFace Dataset | Notes |
|--------|-------|---------------------|-------|
| DiffusionDB | 118,771 | `poloclub/diffusiondb` | Real SD user prompts, CC0 |
| Recap-DataComp-1B | 153,436 | `UCSC-VLAA/Recap-DataComp-1B` | LLaVA recaptions, CC-BY-4.0 |
| DenseFusion-1M | 197,624 | `BAAI/DenseFusion-1M` | Multi-VLM fused captions, Apache-2.0 |
| ShareGPT4V-PT | 76,221 | `Lin-Chen/ShareGPT4V` | GPT-4V captions, Apache-2.0 |
| JourneyDB | 29,964 | `JourneyDB/JourneyDB` | Midjourney prompts, CC-BY-NC-SA |
| SAM-LLaVA-10M | 17,942 | `PixArt-alpha/SAM-LLaVA-Captions10M` | LLaVA on SAM images |
| DOCCI | 13,622 | `google/docci` | Human-written dense captions, CC-BY-4.0 |
| Wukong | 13,282 | `wanng/wukong100m` | Chinese captions, Apache-2.0 |
| Existing pool | 8,581 | — | Prior benchmark prompts |

## Deduplication

Two-stage dedup removed 35% of raw prompts:

1. **MinHash LSH** (Jaccard ≥ 0.7): 970K → 768K — catches near-exact duplicates
2. **Semantic FAISS** (cosine > 0.90): 768K → 630K — catches paraphrases

## Classification

Hybrid keyword + zero-shot embedding classifier (`scripts/zeroshot_classify.py`):

- Keywords first (high precision when they match)
- Zero-shot fallback with `all-MiniLM-L6-v2` embeddings for unmatched prompts
- Margin threshold (0.05) prevents low-confidence reclassification
- T6/T7 keyword gating to prevent false positives from descriptive captions

Each prompt is classified on 3 axes:

- **Subject** (14 categories): S1 People, S2 Animals, S3 Food, ..., S14 Abstract
- **Style** (7 categories): T1 Photorealistic, T2 TraditionalArt, ..., T7 Mixed
- **Camera** (8 categories): C1 Standard, C2 Macro, ..., C8 DramaticLighting

All 98 Subject × Style cells are populated.

## Directory Structure

```
data/
├── README.md                       # This file
├── DISCOVERY.md                    # Design decisions and lessons learned
├── PLAN.md                         # Original scaling plan
├── train/
│   └── metadata.json               # 501K training prompts (Git LFS)
├── val/
│   └── metadata.json               # 13K validation prompts (Git LFS)
├── test/
│   └── metadata.json               # 13K test prompts (Git LFS)
├── debug/
│   └── metadata.json               # 98 prompts (1 per Subject×Style cell)
└── scripts/
    ├── harvest.py                  # Phase 1: Download & filter from HuggingFace
    ├── dedup_minhash.py            # Phase 2.1: MinHash LSH dedup
    ├── dedup_semantic.py           # Phase 2.2-2.3: Embedding + FAISS semantic dedup
    ├── zeroshot_classify.py        # Hybrid classification
    ├── classify_and_sample.py      # Phase 3: Taxonomy classification + sampling
    ├── build_dataset.py            # Phase 4: Length filter, validation, assembly
    ├── split_dataset.py            # Phase 5: Stratified train/val/test split
    ├── prepare_prompts.py          # Original benchmark downloader
    ├── generate_prompts.py         # Gap-fill via Claude API
    ├── merge_prompts.py            # Merge benchmark + gap-fill prompts
    └── visualize_tsne.py           # t-SNE embedding visualization
```

Plots are stored in the top-level `assets/` directory (`tsne_subject.png`, `tsne_style.png`).

## How to Use

**Training:**
```bash
python train.py --train_data_meta=data/train/metadata.json
```

**Validation / Test:**
```bash
python train.py --train_data_meta=data/val/metadata.json   # validation split
python train.py --train_data_meta=data/test/metadata.json   # test split
```

**Debug (smoke test):**
```bash
python train.py --train_data_meta=data/debug/metadata.json
```

## Rebuilding from Scratch

The full pipeline can be re-run if needed (requires `HF_TOKEN` for gated datasets):

```bash
export HF_TOKEN=your_token_here

# Phase 1: Download (~2-4 hours)
python3.13 data/scripts/harvest.py

# Phase 2.1: MinHash dedup (~40 min)
python3.13 data/scripts/dedup_minhash.py

# Phase 2.2-2.3: Semantic dedup (~3 hours on CPU)
python3.13 data/scripts/dedup_semantic.py

# Phase 3: Classify (hybrid, ~2.5 hours on CPU)
python3.13 data/scripts/zeroshot_classify.py

# Phase 4: Assemble final dataset
python3.13 data/scripts/build_dataset.py --max-words 300

# Phase 5: Stratified train/val/test split (95/2.5/2.5)
python3.13 data/scripts/split_dataset.py
```

## Train / Val / Test Split

`scripts/split_dataset.py` creates a stratified 95 / 2.5 / 2.5 split from the
full dataset. Stratification is on the **Subject × Style** cell (98 cells) so
each split preserves the original category distribution.

```bash
# Default: 95% train, 2.5% val, 2.5% test, seed=42
python data/scripts/split_dataset.py

# Custom fractions
python data/scripts/split_dataset.py --val-frac 0.05 --test-frac 0.05

# Custom input path
python data/scripts/split_dataset.py --input data/train/metadata.json
```

| Split | Prompts | % |
|-------|--------:|------:|
| Train | 500,569 | 95.0% |
| Val | 13,173 | 2.5% |
| Test | 13,173 | 2.5% |

The split is deterministic (seed 42) and reproducible.

## Pipeline Scripts

| Script | Input | Output | Time (16-core CPU) |
|--------|-------|--------|-------------------|
| `scripts/harvest.py` | HuggingFace datasets | `data/raw/*.jsonl`, `raw_merged.jsonl` | ~2-4 hours |
| `scripts/dedup_minhash.py` | `raw_merged.jsonl` | `deduped_stage1.jsonl` | ~40 min |
| `scripts/dedup_semantic.py` | `deduped_stage1.jsonl` | `deduped_stage2.jsonl` | ~3 hours |
| `scripts/zeroshot_classify.py` | `deduped_stage2.jsonl` | `full_batch.jsonl` | ~2.5 hours |
| `scripts/build_dataset.py` | `full_batch.jsonl` | `train/metadata.json` | ~2 min |
| `scripts/split_dataset.py` | `train/metadata.json` | `train/`, `val/`, `test/metadata.json` | ~10 sec |
