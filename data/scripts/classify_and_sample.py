#!/usr/bin/env python3
"""
Phase 3: Classification & Balanced MMR Sampling.

Input:  data/deduped_stage2.jsonl + data/embeddings.mmap
Output: data/full_batch.jsonl

Classifies all prompts into (Subject, Style, Camera) taxonomy,
analyzes coverage, applies MMR diversity sampling.
"""

import json
import logging
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
INPUT_PATH = DATA_DIR / "deduped_stage2.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.mmap"
STAGE1_PATH = DATA_DIR / "deduped_stage1.jsonl"
OUTPUT_PATH = DATA_DIR / "full_batch.jsonl"

EMBED_DIM = 384
TARGET_TOTAL = 1_000_000
MIN_PER_CELL = 2_000

# Import classifiers from prepare_prompts
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_prompts import (
    classify_subject, classify_style, classify_camera,
    SUBJECTS, STYLES, CAMERAS,
)


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def classify_all(records: list[dict]) -> list[dict]:
    """Classify all prompts into (Subject, Style, Camera) taxonomy."""
    logger.info("Classifying prompts...")
    for r in tqdm(records, desc="Classifying"):
        text = r["text"]
        r["subject"] = classify_subject(text, r.get("original_category", ""), r["source"])
        r["style"] = classify_style(text)
        r["camera"] = classify_camera(text)
    return records


def print_coverage(records: list[dict]):
    """Print Subject x Style coverage matrix."""
    matrix = defaultdict(lambda: defaultdict(int))
    for r in records:
        matrix[r["subject"]][r["style"]] += 1

    style_codes = sorted(STYLES.keys(), key=lambda x: int(x[1:]))
    subject_codes = sorted(SUBJECTS.keys(), key=lambda x: int(x[1:]))

    header = f"{'':>6}" + "".join(f"{s:>8}" for s in style_codes) + f"{'TOTAL':>8}"
    print("\n" + "=" * len(header))
    print("COVERAGE MATRIX: Subject x Style")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    grand_total = 0
    for subj in subject_codes:
        row_total = 0
        row_str = f"{subj:>6}"
        for sty in style_codes:
            count = matrix[subj][sty]
            row_total += count
            row_str += f"{count:>8}"
        row_str += f"{row_total:>8}"
        grand_total += row_total
        print(row_str)

    print("-" * len(header))
    totals_str = f"{'TOTAL':>6}"
    for sty in style_codes:
        col_total = sum(matrix[s][sty] for s in subject_codes)
        totals_str += f"{col_total:>8}"
    totals_str += f"{grand_total:>8}"
    print(totals_str)
    print("=" * len(header))
    print(f"\nTotal: {grand_total}")
    return matrix


def mmr_sample(embeddings: np.ndarray, indices: list[int], target_k: int,
               lambda_param: float = 0.5, max_candidates: int = None) -> list[int]:
    """Select target_k most diverse items using Maximal Marginal Relevance."""
    n = len(indices)
    if n <= target_k:
        return indices

    # Pre-sample if too large
    if max_candidates and n > max_candidates:
        sample_idx = np.random.choice(n, max_candidates, replace=False)
        embeddings = embeddings[sample_idx]
        indices = [indices[i] for i in sample_idx]
        n = max_candidates

    embs = np.ascontiguousarray(embeddings)
    centroid = embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    # Start with prompt closest to centroid
    scores_to_centroid = embs @ centroid
    first = int(np.argmax(scores_to_centroid))

    selected = [first]
    remaining = set(range(n)) - {first}

    for _ in range(target_k - 1):
        if not remaining:
            break

        remaining_list = list(remaining)
        cand_embs = embs[remaining_list]

        # Relevance: similarity to centroid
        relevance = cand_embs @ centroid

        # Redundancy: max similarity to any already-selected item
        selected_embs = embs[selected]
        sim_to_selected = cand_embs @ selected_embs.T
        redundancy = sim_to_selected.max(axis=1)

        # MMR score
        mmr_scores = lambda_param * relevance - (1 - lambda_param) * redundancy
        best_local = int(np.argmax(mmr_scores))
        best_global = remaining_list[best_local]

        selected.append(best_global)
        remaining.remove(best_global)

    return [indices[i] for i in selected]


def weighted_sampling(records: list[dict], embeddings_map: dict = None) -> list[dict]:
    """
    Option B: Weighted sampling with minimum floor per cell.
    Floor: MIN_PER_CELL per cell, remaining distributed proportionally.
    """
    # Group by (subject, style) cell
    cells = defaultdict(list)
    for i, r in enumerate(records):
        key = (r["subject"], r["style"])
        cells[key].append(i)

    n_cells = len(SUBJECTS) * len(STYLES)  # 14 * 7 = 98
    logger.info(f"Populated cells: {len(cells)} / {n_cells}")

    total_available = len(records)
    target = min(TARGET_TOTAL, total_available)

    # If total pool <= target, take everything (no sampling needed)
    if total_available <= target:
        logger.info(f"Pool ({total_available}) <= target ({target}), taking all prompts")
        return records

    # Calculate allocations (only needed if we have > 1M)
    reserved = min(MIN_PER_CELL * n_cells, target)
    remaining_budget = target - reserved

    total_pool = sum(len(v) for v in cells.values())
    allocations = {}

    for key, indices in cells.items():
        base = min(MIN_PER_CELL, len(indices))
        if total_pool > 0:
            prop = remaining_budget * len(indices) / total_pool
        else:
            prop = 0
        alloc = int(base + prop)
        alloc = min(alloc, len(indices))
        allocations[key] = alloc

    logger.info(f"Target total: {target}, sum of allocations: {sum(allocations.values())}")

    # Sample from each cell — use random sampling for speed, MMR only for small diverse cells
    sampled_indices = []
    for key, indices in tqdm(cells.items(), desc="Sampling cells"):
        target_k = allocations[key]
        if target_k <= 0:
            continue

        if len(indices) <= target_k:
            sampled_indices.extend(indices)
        elif embeddings_map and target_k <= 10000 and len(indices) > target_k:
            # Use MMR for small-to-medium cells where diversity matters
            cell_embs = np.array([embeddings_map[i] for i in indices])
            selected = mmr_sample(
                cell_embs, indices, target_k,
                lambda_param=0.5,
                max_candidates=min(5 * target_k, len(indices))
            )
            sampled_indices.extend(selected)
        else:
            # Random sample for very large cells
            chosen = np.random.choice(indices, target_k, replace=False)
            sampled_indices.extend(chosen.tolist())

    sampled = [records[i] for i in sampled_indices]
    logger.info(f"Sampled {len(sampled)} prompts")
    return sampled


def main():
    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    records = load_records(INPUT_PATH)
    logger.info(f"Loaded {len(records)} prompts")

    # Classify
    records = classify_all(records)

    # Coverage analysis
    print_coverage(records)

    # Load embeddings if available for MMR sampling
    embeddings_map = None
    # We need to map stage2 indices back to embeddings
    # Since stage2 is a subset of stage1, we need the original indices
    # For simplicity, if embeddings exist with matching count, use them
    if EMBEDDINGS_PATH.exists():
        try:
            # Try to load stage1 to get the mapping
            stage1 = load_records(STAGE1_PATH)
            stage1_texts = {r["text"]: i for i, r in enumerate(stage1)}
            emb_shape = (len(stage1), EMBED_DIM)
            all_embeddings = np.memmap(
                str(EMBEDDINGS_PATH), dtype="float32",
                mode="r", shape=emb_shape
            )
            embeddings_map = {}
            mapped = 0
            for i, r in enumerate(records):
                if r["text"] in stage1_texts:
                    orig_idx = stage1_texts[r["text"]]
                    embeddings_map[i] = all_embeddings[orig_idx]
                    mapped += 1
            logger.info(f"Mapped {mapped}/{len(records)} embeddings for MMR sampling")
            if mapped < len(records) * 0.8:
                logger.warning("Low embedding coverage, falling back to random sampling")
                embeddings_map = None
        except Exception as e:
            logger.warning(f"Could not load embeddings: {e}")
            embeddings_map = None

    # Sample
    sampled = weighted_sampling(records, embeddings_map)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(sampled)} prompts to {OUTPUT_PATH}")

    # Final coverage
    print("\n=== SAMPLED COVERAGE ===")
    print_coverage(sampled)


if __name__ == "__main__":
    main()
