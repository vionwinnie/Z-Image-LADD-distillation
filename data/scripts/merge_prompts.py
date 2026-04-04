#!/usr/bin/env python3
"""Merge benchmark prompts with generated gap-fill prompts into the final training dataset.

Usage:
    python data/scripts/merge_prompts.py
"""

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
BENCHMARK_PATH = DATA_DIR / "all_classified_prompts.json"
GENERATED_DIR = DATA_DIR / "generated"
TRAIN_DIR = DATA_DIR / "train"
TRAIN_META = TRAIN_DIR / "metadata.json"
DEBUG_DIR = DATA_DIR / "debug"
DEBUG_META = DEBUG_DIR / "metadata.json"

SUBJECTS = {
    "S1": "People/Portraits", "S2": "Animals", "S3": "Food&Beverage",
    "S4": "IndoorScenes", "S5": "Outdoor/Landscape", "S6": "Architecture/Urban",
    "S7": "Vehicles", "S8": "Plants/Nature", "S9": "Fashion/Clothing",
    "S10": "Objects/Artifacts", "S11": "Text/Typography", "S12": "WorldKnowledge",
    "S13": "ChineseCultural", "S14": "Abstract/Imagination",
}
STYLES = {
    "T1": "Photorealistic", "T2": "TraditionalArt", "T3": "DigitalIllustration",
    "T4": "3D/CGI", "T5": "Cinematic/Film", "T6": "GraphicDesign",
    "T7": "Mixed/Experimental",
}


def deduplicate(records):
    """Deduplicate by exact text match (case-insensitive)."""
    seen = set()
    unique = []
    for r in records:
        key = r["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def print_coverage(records):
    """Print coverage matrix."""
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
            count = matrix.get(subj, {}).get(sty, 0)
            row_total += count
            row_str += f"{count:>8}"
        row_str += f"{row_total:>8}"
        grand_total += row_total
        print(row_str)

    print("-" * len(header))
    totals_str = f"{'TOTAL':>6}"
    for sty in style_codes:
        col_total = sum(matrix.get(s, {}).get(sty, 0) for s in subject_codes)
        totals_str += f"{col_total:>8}"
    totals_str += f"{grand_total:>8}"
    print(totals_str)
    print("=" * len(header))
    print(f"\nTotal prompts: {grand_total}")


def build_debug_split(classified):
    """Pick 1 prompt per (subject x style) cell."""
    seen = set()
    debug = []
    for r in classified:
        key = (r["subject"], r["style"])
        if key not in seen:
            seen.add(key)
            debug.append(r)
    return debug


def main():
    logger.info("=== Merging benchmark + generated prompts ===\n")

    # Load benchmark prompts
    all_records = []
    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH) as f:
            benchmark = json.load(f)
        logger.info(f"Loaded {len(benchmark)} benchmark prompts")
        all_records.extend(benchmark)
    else:
        logger.warning(f"Benchmark file not found: {BENCHMARK_PATH}")

    # Load all generated prompt files
    if GENERATED_DIR.exists():
        for gen_file in sorted(GENERATED_DIR.glob("*.json")):
            with open(gen_file) as f:
                generated = json.load(f)
            logger.info(f"Loaded {len(generated)} generated prompts from {gen_file.name}")
            all_records.extend(generated)
    else:
        logger.warning(f"Generated directory not found: {GENERATED_DIR}")

    logger.info(f"\nTotal before dedup: {len(all_records)}")

    # Deduplicate
    all_records = deduplicate(all_records)
    logger.info(f"After deduplication: {len(all_records)}")

    # Print coverage
    print_coverage(all_records)

    # Summary by source
    source_counts = Counter(r.get("source", "unknown") for r in all_records)
    print("\nPrompts by source:")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")

    lang_counts = Counter(r.get("language", "en") for r in all_records)
    print("\nPrompts by language:")
    for lang, count in lang_counts.most_common():
        print(f"  {lang}: {count}")

    # Save training set
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_META, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(all_records)} prompts to {TRAIN_META}")

    # Save debug split
    debug = build_debug_split(all_records)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_META, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(debug)} debug prompts to {DEBUG_META}")


if __name__ == "__main__":
    main()
