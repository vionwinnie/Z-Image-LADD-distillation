#!/usr/bin/env python3
"""
Create stratified train / val / test split.

Stratifies on (subject, style) cells so each split preserves the
original distribution across all 98 Subject×Style groups.

Usage:
    python data/scripts/split_dataset.py                    # defaults
    python data/scripts/split_dataset.py --val-frac 0.05 --test-frac 0.05
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent


def stratified_split(
    records: list[dict],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records so every (subject, style) cell keeps its proportions."""
    rng = np.random.default_rng(seed)

    # Group by stratification key
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["subject"], r["style"])
        buckets[key].append(r)

    train, val, test = [], [], []

    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)

        n = len(pool)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        # Ensure we don't over-allocate for very small cells
        if n_test + n_val >= n:
            n_test = min(n_test, max(1, n // 3))
            n_val = min(n_val, max(1, n // 3))

        test.extend(pool[:n_test])
        val.extend(pool[n_test : n_test + n_val])
        train.extend(pool[n_test + n_val :])

    return train, val, test


def report(name: str, records: list[dict]) -> None:
    """Log distribution summary for a split."""
    n = len(records)
    subj_counts = defaultdict(int)
    style_counts = defaultdict(int)
    for r in records:
        subj_counts[r["subject"]] += 1
        style_counts[r["style"]] += 1

    logger.info(f"\n--- {name} ({n:,} prompts) ---")
    logger.info("  Subject distribution:")
    for s in sorted(subj_counts, key=lambda x: int(x[1:])):
        logger.info(f"    {s}: {subj_counts[s]:>7,}  ({subj_counts[s]/n*100:5.2f}%)")
    logger.info("  Style distribution:")
    for s in sorted(style_counts, key=lambda x: int(x[1:])):
        logger.info(f"    {s}: {style_counts[s]:>7,}  ({style_counts[s]/n*100:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val/test split")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_DIR / "train" / "metadata.json"),
        help="Path to the full dataset JSON",
    )
    parser.add_argument("--val-frac", type=float, default=0.025)
    parser.add_argument("--test-frac", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    logger.info(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records):,} prompts")

    train, val, test = stratified_split(
        records, args.val_frac, args.test_frac, args.seed
    )

    logger.info(
        f"\nSplit: train={len(train):,} ({len(train)/len(records)*100:.1f}%)  "
        f"val={len(val):,} ({len(val)/len(records)*100:.1f}%)  "
        f"test={len(test):,} ({len(test)/len(records)*100:.1f}%)"
    )

    # Detailed reports
    report("TRAIN", train)
    report("VAL", val)
    report("TEST", test)

    # Write splits
    out_dir = input_path.parent.parent  # data/
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = out_dir / name / "metadata.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(split):,} prompts -> {out_path}")


if __name__ == "__main__":
    main()
