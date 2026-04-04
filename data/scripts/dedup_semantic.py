#!/usr/bin/env python3
"""
Phase 2.2-2.3: Semantic embedding + FAISS clustering + pairwise dedup.

Input:  data/deduped_stage1.jsonl
Output: data/deduped_stage2.jsonl

Uses multilingual-e5-small for embeddings (CPU, ONNX Runtime or PyTorch).
FAISS k-means for clustering, pairwise cosine dedup at 0.90 threshold.
"""

import json
import logging
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import faiss
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
INPUT_PATH = DATA_DIR / "deduped_stage1.jsonl"
OUTPUT_PATH = DATA_DIR / "deduped_stage2.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.mmap"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 512
MAX_LENGTH = 128
COSINE_THRESHOLD = 0.90
N_CLUSTERS = 2000


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_embeddings(prompts: list[str], n_prompts: int):
    """Compute embeddings using PyTorch transformers directly (fast on CPU)."""
    from transformers import AutoTokenizer, AutoModel
    import torch

    logger.info(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    logger.info("Model loaded. Using PyTorch CPU inference.")

    # Pre-allocate memmap
    embeddings = np.memmap(
        str(EMBEDDINGS_PATH), dtype="float32",
        mode="w+", shape=(n_prompts, EMBED_DIM)
    )

    for i in tqdm(range(0, n_prompts, BATCH_SIZE), desc="Embedding"):
        batch = prompts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        emb = outputs.last_hidden_state.mean(dim=1).numpy()
        # L2 normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb = emb / norms
        embeddings[i:i+len(batch)] = emb

    embeddings.flush()
    logger.info(f"Embeddings saved to {EMBEDDINGS_PATH}")
    return embeddings


def quality_score(text: str, source: str) -> float:
    """Higher = better quality. Used to pick winner among duplicates."""
    score = 0.0

    wc = len(text.split())
    if 15 <= wc <= 60:
        score += 0.3
    elif 10 <= wc <= 80:
        score += 0.15

    source_scores = {
        "docci": 1.0, "sharegpt4v": 0.8, "densefusion": 0.75,
        "diffusiondb": 0.6, "journeydb": 0.6, "anytext": 0.5,
        "recap_datacomp": 0.5, "sam_llava": 0.4, "wukong": 0.3,
        "existing": 0.7,
    }
    score += 0.2 * source_scores.get(source, 0.3)

    visual_kw = {"red", "blue", "green", "golden", "marble", "wooden",
                 "glass", "dramatic", "soft", "ambient", "neon", "sunset",
                 "shadow", "reflection", "texture", "metallic", "matte"}
    visual_count = sum(1 for w in text.lower().split() if w in visual_kw)
    score += 0.25 * min(visual_count / 5.0, 1.0)

    specifics = len(re.findall(r'\b[A-Z][a-z]+\b', text))
    specifics += len(re.findall(r'\b\d+\b', text))
    score += 0.25 * min(specifics / 4.0, 1.0)

    return score


def faiss_cluster_dedup(records: list[dict], embeddings: np.ndarray) -> list[dict]:
    """FAISS k-means clustering + pairwise semantic dedup."""
    n_prompts = len(records)
    n_clusters = min(N_CLUSTERS, max(10, int(np.sqrt(n_prompts))))
    logger.info(f"FAISS k-means with {n_clusters} clusters on {n_prompts} prompts...")

    # Train k-means on subset
    train_size = min(100_000, n_prompts)
    train_idx = np.random.choice(n_prompts, train_size, replace=False)
    train_sample = np.ascontiguousarray(embeddings[train_idx])

    kmeans = faiss.Kmeans(EMBED_DIM, n_clusters, niter=20, verbose=True)
    kmeans.train(train_sample)

    # Assign all prompts to clusters
    logger.info("Assigning prompts to clusters...")
    assignments = np.empty(n_prompts, dtype=np.int64)
    chunk_size = 100_000
    for i in range(0, n_prompts, chunk_size):
        chunk = np.ascontiguousarray(embeddings[i:min(i+chunk_size, n_prompts)])
        _, chunk_assignments = kmeans.index.search(chunk, 1)
        assignments[i:i+len(chunk)] = chunk_assignments.squeeze()

    # Pairwise dedup within clusters
    logger.info(f"Pairwise dedup within clusters (threshold={COSINE_THRESHOLD})...")
    duplicates = set()

    # Pre-compute quality scores
    scores = [quality_score(r["text"], r["source"]) for r in records]

    cluster_sizes = []
    for cluster_id in tqdm(range(n_clusters), desc="Cluster dedup"):
        mask = (assignments == cluster_id)
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) < 2:
            continue

        cluster_sizes.append(len(cluster_indices))
        cluster_embs = np.ascontiguousarray(embeddings[cluster_indices])

        # Pairwise cosine similarity (embeddings are L2-normalized)
        sim_matrix = cluster_embs @ cluster_embs.T

        for i in range(len(cluster_indices)):
            if cluster_indices[i] in duplicates:
                continue
            for j in range(i + 1, len(cluster_indices)):
                if cluster_indices[j] in duplicates:
                    continue
                if sim_matrix[i, j] > COSINE_THRESHOLD:
                    idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                    # Keep higher quality
                    if scores[idx_i] >= scores[idx_j]:
                        duplicates.add(idx_j)
                    else:
                        duplicates.add(idx_i)

    if cluster_sizes:
        logger.info(f"Cluster size stats: mean={np.mean(cluster_sizes):.0f}, "
                     f"max={max(cluster_sizes)}, median={np.median(cluster_sizes):.0f}")

    kept = [r for i, r in enumerate(records) if i not in duplicates]
    logger.info(f"Semantic dedup: {n_prompts} -> {len(kept)} ({len(duplicates)} removed)")
    return kept


def main():
    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    records = load_records(INPUT_PATH)
    logger.info(f"Loaded {len(records)} prompts from {INPUT_PATH}")

    n_prompts = len(records)
    prompts = [r["text"] for r in records]

    # Phase 2.2: Compute embeddings
    if EMBEDDINGS_PATH.exists():
        logger.info(f"Loading existing embeddings from {EMBEDDINGS_PATH}")
        embeddings = np.memmap(
            str(EMBEDDINGS_PATH), dtype="float32",
            mode="r", shape=(n_prompts, EMBED_DIM)
        )
    else:
        embeddings = compute_embeddings(prompts, n_prompts)

    # Phase 2.3: FAISS cluster + dedup
    kept = faiss_cluster_dedup(records, embeddings)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(kept)} prompts to {OUTPUT_PATH}")

    # Stats
    source_counts = Counter(r["source"] for r in kept)
    logger.info("By source:")
    for src, cnt in source_counts.most_common():
        logger.info(f"  {src}: {cnt}")

    # Cleanup hint
    logger.info(f"\nEmbeddings file at {EMBEDDINGS_PATH} can be deleted after this step.")


if __name__ == "__main__":
    main()
