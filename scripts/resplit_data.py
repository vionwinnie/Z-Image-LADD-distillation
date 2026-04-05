#!/usr/bin/env python3
"""Resplit val/test to 1000 balanced samples each, move leftovers to train.

Balanced sampling: stratified by (subject, style, camera) to preserve
distribution of each category. Uses proportional allocation with rounding,
then top-up from the largest strata if needed.

Usage:
    python scripts/resplit_data.py [--dry-run]
"""

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict

SEED = 42
TARGET_SIZE = 1000

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_META = os.path.join(DATA_DIR, "train", "metadata.json")
VAL_META = os.path.join(DATA_DIR, "val", "metadata.json")
TEST_META = os.path.join(DATA_DIR, "test", "metadata.json")


def stratified_sample(records, n, seed):
    """Sample n records with balanced (subject, style, camera) stratification."""
    rng = random.Random(seed)

    # Group by stratification key
    strata = defaultdict(list)
    for r in records:
        key = (r["subject"], r["style"], r["camera"])
        strata[key].append(r)

    # Proportional allocation
    total = len(records)
    allocation = {}
    for key, items in strata.items():
        allocation[key] = max(1, round(len(items) / total * n))

    # Adjust to hit exactly n
    allocated = sum(allocation.values())
    # Sort strata by size (largest first) for adjustment
    sorted_keys = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)

    if allocated > n:
        # Remove from largest strata
        for key in sorted_keys:
            if allocated <= n:
                break
            if allocation[key] > 1:
                allocation[key] -= 1
                allocated -= 1
    elif allocated < n:
        # Add to largest strata
        for key in sorted_keys:
            if allocated >= n:
                break
            if allocation[key] < len(strata[key]):
                allocation[key] += 1
                allocated += 1

    # Sample from each stratum
    selected = []
    for key, items in strata.items():
        k = min(allocation.get(key, 0), len(items))
        selected.extend(rng.sample(items, k))

    # Final adjustment if rounding left us off
    rng.shuffle(selected)
    if len(selected) > n:
        selected = selected[:n]
    elif len(selected) < n:
        selected_set = set(id(r) for r in selected)
        remaining = [r for r in records if id(r) not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    print("Loading metadata...")
    with open(TRAIN_META) as f:
        train = json.load(f)
    with open(VAL_META) as f:
        val = json.load(f)
    with open(TEST_META) as f:
        test = json.load(f)

    print(f"Before: train={len(train)}, val={len(val)}, test={len(test)}")

    # Sample 1000 balanced from val and test
    val_keep = stratified_sample(val, TARGET_SIZE, SEED)
    test_keep = stratified_sample(test, TARGET_SIZE, SEED + 1)

    # Leftovers go to train
    val_keep_set = set(id(r) for r in val_keep)
    test_keep_set = set(id(r) for r in test_keep)
    val_leftover = [r for r in val if id(r) not in val_keep_set]
    test_leftover = [r for r in test if id(r) not in test_keep_set]

    train_new = train + val_leftover + test_leftover

    print(f"After:  train={len(train_new)}, val={len(val_keep)}, test={len(test_keep)}")
    print(f"  Added to train: {len(val_leftover)} from val + {len(test_leftover)} from test")

    # Print distribution of new val/test
    from collections import Counter
    for name, data in [("val", val_keep), ("test", test_keep)]:
        print(f"\n  {name} distribution:")
        for key in ["subject", "style", "camera"]:
            counts = Counter(r[key] for r in data)
            print(f"    {key}: {dict(sorted(counts.items()))}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Backup originals
    for path in [TRAIN_META, VAL_META, TEST_META]:
        backup = path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"  Backed up {path} -> {backup}")

    # Write new metadata
    with open(VAL_META, "w") as f:
        json.dump(val_keep, f, indent=2)
    with open(TEST_META, "w") as f:
        json.dump(test_keep, f, indent=2)
    with open(TRAIN_META, "w") as f:
        json.dump(train_new, f, indent=2)

    print("\nDone. Old files backed up as *.bak")

    # Clean up val embeddings (no longer valid for new split)
    val_emb = os.path.join(DATA_DIR, "val", "embeddings")
    if os.path.exists(val_emb):
        shutil.rmtree(val_emb)
        print(f"Removed stale {val_emb}")
    val_fid = os.path.join(DATA_DIR, "val", "fid_reference_stats.npz")
    if os.path.exists(val_fid):
        os.remove(val_fid)
        print(f"Removed stale {val_fid}")


if __name__ == "__main__":
    main()
