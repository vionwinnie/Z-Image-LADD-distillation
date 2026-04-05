#!/usr/bin/env python3
"""Create a stratified subsample of training prompts for precompute.

Samples ~10K prompts from the full training set using the same stratified
sampling approach as resplit_data.py (balanced by subject, style, camera).

Usage:
    python scripts/subsample_prompts.py [--n 10000] [--dry-run]
"""

import argparse
import json
import os
import random
from collections import defaultdict

SEED = 42
DEFAULT_N = 10000

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_META = os.path.join(DATA_DIR, "train", "metadata.json")
OUTPUT_META = os.path.join(DATA_DIR, "train", "metadata_subsample.json")


def stratified_sample(records, n, seed):
    """Sample n records with balanced (subject, style, camera) stratification."""
    rng = random.Random(seed)

    # Group by stratification key
    strata = defaultdict(list)
    for r in records:
        key = (r.get("subject", ""), r.get("style", ""), r.get("camera", ""))
        strata[key].append(r)

    # Proportional allocation
    total = len(records)
    allocation = {}
    for key, items in strata.items():
        allocation[key] = max(1, round(len(items) / total * n))

    # Adjust to hit exactly n
    allocated = sum(allocation.values())
    sorted_keys = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)

    if allocated > n:
        for key in sorted_keys:
            if allocated <= n:
                break
            if allocation[key] > 1:
                allocation[key] -= 1
                allocated -= 1
    elif allocated < n:
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

    # Final adjustment
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
    parser = argparse.ArgumentParser(description="Stratified subsample of training prompts.")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of prompts to sample (default: {DEFAULT_N}).")
    parser.add_argument("--input", type=str, default=TRAIN_META,
                        help="Input metadata JSON file.")
    parser.add_argument("--output", type=str, default=OUTPUT_META,
                        help="Output metadata JSON file.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing.")
    args = parser.parse_args()

    with open(args.input) as f:
        train = json.load(f)

    print(f"Full training set: {len(train)} prompts")

    n = min(args.n, len(train))
    selected = stratified_sample(train, n, SEED)

    print(f"Subsampled: {len(selected)} prompts")

    # Print distribution
    from collections import Counter
    for key in ["subject", "style", "camera"]:
        if key in selected[0]:
            counts = Counter(r[key] for r in selected)
            top5 = counts.most_common(5)
            print(f"  {key}: {len(counts)} unique — top 5: {top5}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
