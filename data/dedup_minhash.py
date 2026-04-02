#!/usr/bin/env python3
"""
Phase 2.1: MinHash LSH surface-level deduplication.

Input:  data/raw_merged.jsonl
Output: data/deduped_stage1.jsonl

Uses datasketch MinHash LSH with Jaccard threshold 0.7, 128 permutations.
Handles both English (word tokenization) and Chinese (character 3-grams).
"""

import json
import logging
import re
import sys
from pathlib import Path
from collections import Counter

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "raw_merged.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "deduped_stage1.jsonl"

NUM_PERM = 128
THRESHOLD = 0.7


def tokenize(text: str, lang: str) -> list[str]:
    """Tokenize for MinHash: word split for EN, character 3-grams for ZH."""
    if lang == "zh":
        chars = re.sub(r'\s+', '', text)
        return [chars[i:i+3] for i in range(max(1, len(chars) - 2))]
    else:
        return text.lower().split()


def make_minhash(tokens: list[str]) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for t in tokens:
        mh.update(t.encode('utf-8'))
    return mh


def main():
    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    # Load all prompts
    logger.info(f"Loading prompts from {INPUT_PATH}...")
    records = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} prompts")

    # MinHash LSH dedup
    logger.info(f"Running MinHash LSH (threshold={THRESHOLD}, num_perm={NUM_PERM})...")
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    keep_indices = []
    duplicates = 0

    for idx, rec in enumerate(tqdm(records, desc="MinHash LSH")):
        tokens = tokenize(rec["text"], rec.get("lang", "en"))
        if not tokens:
            keep_indices.append(idx)  # Keep empty-ish ones for now
            continue

        mh = make_minhash(tokens)

        # Query for near-duplicates before inserting
        result = lsh.query(mh)
        if not result:
            lsh.insert(str(idx), mh)
            keep_indices.append(idx)
        else:
            # Among duplicates, keep the longer prompt
            existing_idx = int(result[0])
            if len(rec["text"]) > len(records[existing_idx]["text"]):
                # Replace: remove old, insert new
                try:
                    lsh.remove(str(existing_idx))
                except:
                    pass
                lsh.insert(str(idx), mh)
                keep_indices = [i for i in keep_indices if i != existing_idx]
                keep_indices.append(idx)
            duplicates += 1

    kept = [records[i] for i in keep_indices]
    logger.info(f"MinHash LSH complete: {len(records)} -> {len(kept)} ({duplicates} duplicates removed)")

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved to {OUTPUT_PATH}")

    # Stats
    source_counts = Counter(r["source"] for r in kept)
    logger.info("By source:")
    for src, cnt in source_counts.most_common():
        logger.info(f"  {src}: {cnt}")


if __name__ == "__main__":
    main()
