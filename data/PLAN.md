# Scaling Prompt Dataset to 1M: Execution Plan

This plan is designed for a **remote CPU instance with 32GB RAM, no GPU**.
It covers harvesting candidate datasets (DISCOVERY.md Section 9) and semantic deduplication (Section 10).

---

## Phase 1: Harvest & Download Datasets

Download each dataset, extract text prompts, normalize format, and save as JSONL.
All outputs go to `data/raw/{source_name}.jsonl` with schema: `{"text": "...", "source": "..."}`.

### 1.1 Install Dependencies

```bash
pip install datasets huggingface_hub datasketch sentence-transformers onnxruntime optimum[onnxruntime] faiss-cpu numpy scipy tqdm
```

### 1.2 Download & Extract Prompts

Process each source independently. Can be parallelized across datasets.

| # | Source | HuggingFace ID | Extract Field | Target Count | Notes |
|---|--------|----------------|---------------|-------------|-------|
| 1 | DiffusionDB | `poloclub/diffusiondb` (`2m_text_only` config) | `prompt` | 300K (sample after dedup) | **Use `2m_text_only` config** (not `2m_first_1m` which downloads images). Strip negative prompts. Filter NSFW: drop rows where `prompt_nsfw > 0.5`. |
| 2 | Recap-DataComp-1B | `UCSC-VLAA/Recap-DataComp-1B` | `re_caption` | 250K (sample) | Stream — do NOT download full 1.3B. Use `datasets` streaming mode, sample first ~500K. If streaming fails due to schema mismatches, load specific parquet shards directly via `load_dataset("parquet", data_files=...)`. |
| 3 | DenseFusion-1M | `DenseFusion/DenseFusion-1M` | `caption` | 200K (sample) | **Gated dataset** — requires HF token with accepted terms. Check actual field name on download. |
| 4 | ShareGPT4V-PT | `Lin-Chen/ShareGPT4V` (`ShareGPT4V-PT` config) | extract from conversations | 100K | **Must specify config `"ShareGPT4V-PT"`** for the 1.2M pre-training set (default loads the 102K instruct set). Extract assistant's first reply as the caption. |
| 5 | JourneyDB | `JourneyDB/JourneyDB` | `prompt` | 50K | **Gated** — requires HF login + terms acceptance. Strip MJ params (`--ar`, `--v`, `--s`, etc.). NC license — flag for review. |
| 6 | AnyText-3M | `modelscope/AnyText-3M` | `caption` / `text` | 30K | Bilingual EN+ZH. Good for S11 (Typography). |
| 7 | SAM-LLaVA-10M | `PixArt-alpha/SAM-LLaVA-Captions10M` | `txt` | 30K | Stream, sample 30K. Field is `txt`, not `caption`. |
| 8 | Wukong | `noah-wukong/wukong` | `caption` | 15K | **Gated** — requires HF token. Chinese only. Filter for length > 10 chars. |
| 9 | DOCCI | `google/docci` | `description` | 15K | Small dataset, take all. Highest quality. |
| 10 | Existing pool | `data/train/metadata.json` | `text` | 16K | Already classified. |

**Streaming strategy for large datasets** (Recap-DataComp, SAM-LLaVA, Wukong):
```python
from datasets import load_dataset
ds = load_dataset("UCSC-VLAA/Recap-DataComp-1B", split="train", streaming=True)
# Iterate and collect first N prompts passing quality filters
```

### 1.3 Per-Source Quality Filters

Apply before saving to JSONL:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Min word count (EN) | >= 8 words | Too short = too vague for T2I |
| Min char count (ZH) | >= 15 chars | Equivalent threshold for Chinese |
| Max word count | <= 200 words | Extremely long prompts add noise |
| Dedup exact text | exact string match | Remove verbatim copies within source |
| Drop URL-heavy | < 2 URLs in text | Web-crawled captions often contain URLs |
| Drop boilerplate | reject "stock photo", "getty images", "shutterstock" | Watermark/source artifacts |
| ASCII-art / code | reject if > 30% non-alpha chars | Corrupted entries |
| NSFW (DiffusionDB) | drop if `prompt_nsfw > 0.5` | Explicit content not suitable for training |

**Cross-source exact dedup**: After all sources are downloaded, run a global exact string match dedup across all JSONL files before proceeding to Phase 2. This is O(n) with a set and costs nothing.

### 1.4 Format Normalization

Each prompt saved as:
```json
{"text": "the prompt text", "source": "diffusiondb", "lang": "en"}
```

Language detection: use CJK character ratio (`re.search(r'[\u4e00-\u9fff]', text)`) — if > 30% CJK chars, tag as `zh`.

**Expected output**: `data/raw/` directory with ~1.1M raw prompts across 10 JSONL files.

---

## Phase 2: Deduplication Pipeline

Two-stage dedup: fast surface-level pass, then semantic pass. Designed for 32GB RAM CPU.

### 2.1 Stage 1 — MinHash LSH (Surface-Level Dedup)

Catches near-exact duplicates: copy-paste, minor rephrasing, prompt+param variants.

**Tool**: `datasketch` MinHash LSH
**Memory**: ~5-8 GB for 1M prompts
**Time**: ~1-2 hours

```python
import re
from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.7, num_perm=128)  # Jaccard threshold 0.7

def tokenize(text, lang):
    """Tokenize for MinHash: word split for EN, character 3-grams for ZH."""
    if lang == "zh":
        # Chinese has no spaces — use character-level n-grams
        chars = re.sub(r'\s+', '', text)
        return [chars[i:i+3] for i in range(max(1, len(chars) - 2))]
    else:
        return text.lower().split()

for idx, (prompt, lang) in enumerate(all_prompts):
    mh = MinHash(num_perm=128)
    for token in tokenize(prompt, lang):
        mh.update(token.encode('utf-8'))
    # Query for near-duplicates before inserting
    result = lsh.query(mh)
    if not result:
        lsh.insert(str(idx), mh)
        keep.add(idx)
    else:
        # Keep the longer prompt among duplicates
        ...
```

**Config**:
- `num_perm=128` — good precision/speed tradeoff
- `threshold=0.7` — catches "a cute cat in cinema" vs "a cute cat sitting in a cinema" but not semantic rephrases
- **Chinese tokenization**: use character-level 3-grams since Chinese text has no word boundaries. Process EN and ZH through the same LSH but with different tokenizers.

**Expected removal**: 10-30% of prompts (mainly DiffusionDB internal duplication)
**Output**: `data/deduped_stage1.jsonl` (~700K-900K prompts)

### 2.2 Stage 2 — Semantic Embedding

Embed all surviving prompts into dense vectors for semantic dedup.

**Model choice for CPU + bilingual**:

| Model | Dim | ONNX Speed (CPU) | Memory | EN+ZH | Recommended |
|-------|-----|-------------------|--------|-------|-------------|
| `all-MiniLM-L6-v2` | 384 | ~1000/sec | 90MB | No | EN-only workloads |
| **`multilingual-e5-small`** | 384 | ~500/sec | 471MB | **Yes** | **Best for our case** |
| `BGE-M3` | 1024 | ~70/sec | 2.2GB | Yes | Too slow, too large for 32GB |

**Selected: `multilingual-e5-small`** (384d) with ONNX Runtime

- **Time**: ~800K prompts / 500 per sec = ~27 min. For 1M = ~33 min (with ONNX)
- **Memory**: model ~500MB + batch overhead ~500MB = ~1GB during embedding
- **Output**: embeddings saved to disk as numpy memmap

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np

# NOTE: First run with export=True is slow (converts PyTorch → ONNX).
# After first run, the ONNX model is cached locally.
model = ORTModelForFeatureExtraction.from_pretrained(
    "intfloat/multilingual-e5-small", export=True
)
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")

# Pre-allocate memmap file on disk
n_prompts = len(prompts)
embeddings = np.memmap("data/embeddings.mmap", dtype="float32",
                        mode="w+", shape=(n_prompts, 384))

batch_size = 256
for i in range(0, n_prompts, batch_size):
    batch = prompts[i:i+batch_size]
    # E5 models require "query: " prefix for encoding
    batch = ["query: " + p for p in batch]
    inputs = tokenizer(batch, padding=True, truncation=True,
                       max_length=128, return_tensors="np")  # np, not pt — ONNX Runtime
    outputs = model(**inputs)
    # Mean pooling (outputs are already numpy arrays)
    emb = outputs.last_hidden_state.mean(axis=1)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # L2 normalize
    embeddings[i:i+batch_size] = emb

embeddings.flush()
```

**Memory profile**: embeddings memmap = 800K x 384 x 4 bytes = ~1.2 GB on disk (memory-mapped, OS manages paging). Model + batch in RAM = ~1.5 GB. **Total: ~3 GB peak**.

**Alternative — OpenAI API** (if data privacy is acceptable):
- `text-embedding-3-small` at dim=384: **~$8-12 for 10M prompts**, ~1-2 hours with parallel batches
- Much faster, no local compute needed, but data leaves your machine

### 2.3 Stage 3 — FAISS Clustering & Semantic Dedup

**Memory budget** for FAISS at 800K x 384d float32: ~1.2 GB. Comfortable.

#### Step A: Build FAISS Index

```python
import faiss
import numpy as np

embeddings = np.memmap("data/embeddings.mmap", dtype="float32",
                        mode="r", shape=(n_prompts, 384))

# K-means clustering
n_clusters = 2000  # sqrt(800K) ~ 894, round up for finer granularity
kmeans = faiss.Kmeans(384, n_clusters, niter=20, verbose=True)
# Train on random subset for speed — must be contiguous array for FAISS
train_idx = np.random.choice(n_prompts, 100_000, replace=False)
train_sample = np.ascontiguousarray(embeddings[train_idx])
kmeans.train(train_sample)

# Assign all prompts to nearest cluster (process in chunks for memmap compatibility)
assignments = np.empty(n_prompts, dtype=np.int64)
chunk_size = 100_000
for i in range(0, n_prompts, chunk_size):
    chunk = np.ascontiguousarray(embeddings[i:i+chunk_size])
    _, chunk_assignments = kmeans.index.search(chunk, 1)
    assignments[i:i+chunk_size] = chunk_assignments.squeeze()
```

**Time**: ~5-10 min for training, ~2 min for assignment
**Memory**: ~2 GB (embeddings memmap + centroids + assignment array)

#### Step B: Pairwise Dedup Within Clusters

```python
from scipy.spatial.distance import cdist

threshold = 0.90  # cosine similarity
duplicates = set()

for cluster_id in range(n_clusters):
    mask = (assignments == cluster_id)
    cluster_embs = embeddings[mask]  # avg ~400 prompts per cluster
    cluster_indices = np.where(mask)[0]

    if len(cluster_embs) < 2:
        continue

    # Pairwise cosine similarity
    sim_matrix = cluster_embs @ cluster_embs.T  # already L2-normalized

    for i in range(len(cluster_embs)):
        if cluster_indices[i] in duplicates:
            continue
        for j in range(i + 1, len(cluster_embs)):
            if cluster_indices[j] in duplicates:
                continue
            if sim_matrix[i, j] > threshold:
                # Keep higher quality prompt, mark other as duplicate
                keep, remove = quality_compare(
                    cluster_indices[i], cluster_indices[j]
                )
                duplicates.add(remove)
```

**Time**: ~10-30 min (800K prompts, 2000 clusters, avg 400 per cluster)
**Memory**: largest cluster might be ~5000 prompts -> 5000x384 matrix = ~7.3 MB. Negligible.

#### Step C: Quality Scoring Function

```python
def quality_score(text, source):
    """Higher = better quality. Used to pick winner among duplicates."""
    score = 0.0

    # Word count sweet spot (15-60 words)
    wc = len(text.split())
    if 15 <= wc <= 60:
        score += 0.3
    elif 10 <= wc <= 80:
        score += 0.15

    # Source quality prior
    source_scores = {
        "docci": 1.0, "sharegpt4v": 0.8, "densefusion": 0.75,
        "diffusiondb": 0.6, "journeydb": 0.6, "anytext": 0.5,
        "recap_datacomp": 0.5, "sam_llava": 0.4, "wukong": 0.3,
        "existing": 0.7,
    }
    score += 0.2 * source_scores.get(source, 0.3)

    # Visual detail: count color/material/lighting keywords
    visual_kw = {"red", "blue", "green", "golden", "marble", "wooden",
                 "glass", "dramatic", "soft", "ambient", "neon", "sunset",
                 "shadow", "reflection", "texture", "metallic", "matte"}
    visual_count = sum(1 for w in text.lower().split() if w in visual_kw)
    score += 0.25 * min(visual_count / 5.0, 1.0)

    # Specificity: named entities, numbers, concrete nouns
    import re
    specifics = len(re.findall(r'\b[A-Z][a-z]+\b', text))  # capitalized words
    specifics += len(re.findall(r'\b\d+\b', text))  # numbers
    score += 0.25 * min(specifics / 4.0, 1.0)

    return score
```

**Expected removal**: 5-15% of remaining prompts
**Output**: `data/deduped_stage2.jsonl` (~600K-850K prompts)

---

## Phase 3: Classification & Balanced Sampling

### 3.1 Classify All Prompts

Run all deduped prompts through the existing keyword classifier from `prepare_prompts.py`:

```python
from prepare_prompts import classify_subject, classify_style, classify_camera

for prompt in deduped_prompts:
    prompt["subject"] = classify_subject(prompt["text"])
    prompt["style"] = classify_style(prompt["text"])
    prompt["camera"] = classify_camera(prompt["text"])
```

**Time**: minutes (keyword matching is fast)
**Note**: Keyword classifier has ~25% fallback rate to defaults (S10, T1, C1). For datasets >50K, this is acceptable — LLM-based classification would cost hundreds of dollars at this scale.

### 3.2 Analyze Coverage

Compute the Subject x Style matrix (14 x 7 = 98 cells). Identify:
- Cells with surplus (can sample down)
- Cells with deficit (may need LLM generation to fill)

### 3.3 Diversity-Aware Sampling to 1M

Use **Maximal Marginal Relevance (MMR)** within each taxonomy cell to select the most diverse subset.

**Scalability note**: MMR is O(n*k) per cell. If a cell has >50K candidates, **pre-sample to 5x target** randomly before running MMR to keep runtime manageable.

```python
def mmr_sample(embeddings, indices, target_k, lambda_param=0.5, max_candidates=None):
    """Select target_k most diverse items from candidates using MMR.

    Args:
        max_candidates: If set, randomly pre-sample candidates to this size
                        before running MMR (prevents O(n*k) blowup).
    """
    n = len(indices)

    # Pre-sample if candidate pool is too large
    if max_candidates and n > max_candidates:
        sample_idx = np.random.choice(n, max_candidates, replace=False)
        embeddings = embeddings[sample_idx]
        indices = [indices[i] for i in sample_idx]
        n = max_candidates

    selected = []
    candidates = list(range(n))

    # Start with the prompt closest to cluster centroid
    centroid = embeddings.mean(axis=0)
    first = np.argmax(embeddings @ centroid)
    selected.append(candidates.pop(first))

    for _ in range(target_k - 1):
        if not candidates:
            break
        selected_embs = embeddings[selected]
        candidate_embs = embeddings[candidates]

        # Relevance: similarity to centroid (diversity of concept)
        relevance = candidate_embs @ centroid

        # Redundancy: max similarity to any already-selected item
        redundancy = (candidate_embs @ selected_embs.T).max(axis=1)

        # MMR score
        mmr_scores = lambda_param * relevance - (1 - lambda_param) * redundancy
        best = np.argmax(mmr_scores)
        selected.append(candidates.pop(best))

    return [indices[i] for i in selected]
```

**Usage**: Call with `max_candidates=5*target_k` to cap runtime at ~25x target_k^2 operations per cell.

**Sampling targets** (two options):

**Option A — Uniform across cells** (maximizes diversity):
- 1M / 98 cells = ~10,204 per cell
- May require generation for sparse cells (Abstract/3D CGI, etc.)

**Option B — Weighted to match natural distribution** (recommended):
- Keep current proportional mix but with minimum floor per cell
- Floor: 2,000 per cell (196K reserved)
- Remaining 804K distributed proportionally to candidate pool size

### 3.4 Gap-Fill with LLM Generation

For cells below the minimum floor after sampling:

```bash
python data/generate_prompts.py --target-per-cell 2000 --model claude-haiku
```

- Use Haiku for bulk generation (~$0.25/M input tokens)
- Estimated cost for 50K gap-fill prompts: ~$2-5
- Apply the same prompt expansion step as the original pipeline (30-40 words target)

---

## Phase 4: Final Assembly

### 4.1 Merge & Build

```bash
python data/build_dataset.py \
    --input data/sampled_1m.jsonl \
    --balance-lengths \
    --min-words 8 \
    --max-words 150 \
    --output data/train/metadata.json
```

### 4.2 Validation Checks

| Check | Target |
|-------|--------|
| Total prompts | 1,000,000 +/- 5% |
| All 98 Subject x Style cells populated | >= 2,000 per cell |
| Language split | ~85% EN, ~15% ZH |
| Mean word count (EN) | 30-45 words |
| No exact duplicates | 0 |
| Semantic duplicate rate (spot check 1K random pairs) | < 5% above 0.90 cosine sim |
| Source diversity | No single source > 35% |

### 4.3 Generate Debug Split

```bash
# 1 prompt per Subject x Style cell for smoke testing
python data/build_dataset.py --debug --min-per-cell 1 --output data/debug/metadata.json
```

---

## Resource Estimates

| Phase | Time | Peak RAM | Disk | Cost |
|-------|------|----------|------|------|
| 1. Download & filter | 2-4 hours | ~8 GB | ~5 GB (JSONL) | Free |
| 2.1 MinHash LSH dedup | 1-2 hours | ~5-8 GB | ~2 GB | Free |
| 2.2 Embedding (ONNX, multilingual-e5-small) | ~30-60 min | ~3 GB | ~1.2 GB (memmap) | Free |
| 2.3 FAISS clustering + dedup | ~30 min | ~3 GB | — | Free |
| 3. Classify + MMR sampling | ~30 min | ~4 GB | — | Free |
| 3.4 Gap-fill (Haiku) | ~1 hour | ~1 GB | — | ~$2-5 |
| 4. Assembly + validation | ~15 min | ~4 GB | ~500 MB (final JSON) | Free |
| **Total** | **~6-9 hours** | **~8 GB peak** | **~9 GB** | **~$2-5** |

---

## File Structure After Completion

```
data/
  raw/                          # Phase 1 outputs
    diffusiondb.jsonl
    recap_datacomp.jsonl
    densefusion.jsonl
    sharegpt4v.jsonl
    journeydb.jsonl
    anytext.jsonl
    sam_llava.jsonl
    wukong.jsonl
    docci.jsonl
    existing.jsonl
  embeddings.mmap               # Phase 2.2 output (delete after dedup)
  deduped_stage1.jsonl          # Phase 2.1 output
  deduped_stage2.jsonl          # Phase 2.3 output
  sampled_1m.jsonl              # Phase 3.3 output
  train/metadata.json           # Phase 4 final output (1M prompts)
  debug/metadata.json           # Phase 4.3 debug split
```

---

## Execution Checklist

- [ ] **Phase 1**: Download datasets (parallelize across sources)
  - [ ] DiffusionDB (300K sample)
  - [ ] Recap-DataComp-1B (250K streamed sample)
  - [ ] DenseFusion-1M (200K sample)
  - [ ] ShareGPT4V-PT (100K)
  - [ ] JourneyDB (50K) — flag NC license
  - [ ] AnyText-3M (30K)
  - [ ] SAM-LLaVA-10M (30K streamed)
  - [ ] Wukong (15K, Chinese)
  - [ ] DOCCI (15K, all)
  - [ ] Existing pool (16K)
- [ ] **Phase 2.1**: MinHash LSH dedup
- [ ] **Phase 2.2**: Embed with multilingual-e5-small (ONNX)
- [ ] **Phase 2.3**: FAISS k-means + pairwise semantic dedup (threshold=0.90)
- [ ] **Phase 3.1**: Classify into (S, T, C) taxonomy
- [ ] **Phase 3.2**: Analyze coverage matrix
- [ ] **Phase 3.3**: MMR diversity sampling to 1M
- [ ] **Phase 3.4**: Gap-fill sparse cells with Haiku
- [ ] **Phase 4.1**: Assemble final dataset
- [ ] **Phase 4.2**: Run validation checks
- [ ] **Phase 4.3**: Generate debug split
