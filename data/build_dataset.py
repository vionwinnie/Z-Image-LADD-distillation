#!/usr/bin/env python3
"""
Phase 4: Final Assembly & Validation.

Input:  data/full_batch.jsonl
Output: data/train/metadata.json, data/debug/metadata.json

Runs validation checks and generates debug split.
"""

import json
import logging
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "full_batch.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "train" / "metadata.json"
DEBUG_OUTPUT = SCRIPT_DIR / "debug" / "metadata.json"

sys.path.insert(0, str(SCRIPT_DIR))
from prepare_prompts import SUBJECTS, STYLES, CAMERAS


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate(records: list[dict]) -> bool:
    """Run validation checks per PLAN.md Phase 4.2."""
    logger.info("\n=== VALIDATION ===")
    all_pass = True

    # Total count
    n = len(records)
    target_low, target_high = 950_000, 1_050_000
    status = "PASS" if target_low <= n <= target_high else "WARN"
    if status == "WARN":
        all_pass = False
    logger.info(f"[{status}] Total prompts: {n} (target: 1M +/- 5%)")

    # Coverage: all 98 cells populated
    cell_counts = defaultdict(int)
    for r in records:
        cell_counts[(r["subject"], r["style"])] += 1
    n_cells = len(SUBJECTS) * len(STYLES)
    populated = sum(1 for v in cell_counts.values() if v > 0)
    min_cell = min(cell_counts.values()) if cell_counts else 0
    below_floor = sum(1 for v in cell_counts.values() if v < 2000)
    status = "PASS" if populated == n_cells else "WARN"
    logger.info(f"[{status}] Populated cells: {populated}/{n_cells}, min={min_cell}, below 2K floor: {below_floor}")

    # Language split
    lang_counts = Counter(r.get("lang", "en") for r in records)
    en_pct = lang_counts.get("en", 0) / max(n, 1) * 100
    zh_pct = lang_counts.get("zh", 0) / max(n, 1) * 100
    logger.info(f"[INFO] Language: EN={en_pct:.1f}%, ZH={zh_pct:.1f}%")

    # Mean word count (EN)
    en_wcs = [len(r["text"].split()) for r in records if r.get("lang", "en") == "en"]
    if en_wcs:
        mean_wc = np.mean(en_wcs)
        status = "PASS" if 30 <= mean_wc <= 45 else "INFO"
        logger.info(f"[{status}] Mean EN word count: {mean_wc:.1f} (target: 30-45)")

    # Exact duplicates
    texts = [r["text"].strip().lower() for r in records]
    n_unique = len(set(texts))
    n_dupes = n - n_unique
    status = "PASS" if n_dupes == 0 else "WARN"
    logger.info(f"[{status}] Exact duplicates: {n_dupes}")

    # Source diversity
    source_counts = Counter(r["source"] for r in records)
    max_source_pct = max(source_counts.values()) / max(n, 1) * 100
    max_source_name = source_counts.most_common(1)[0][0] if source_counts else "N/A"
    status = "PASS" if max_source_pct <= 35 else "WARN"
    logger.info(f"[{status}] Max source: {max_source_name} = {max_source_pct:.1f}% (limit: 35%)")

    logger.info("\nSource distribution:")
    for src, cnt in source_counts.most_common():
        pct = cnt / max(n, 1) * 100
        logger.info(f"  {src}: {cnt} ({pct:.1f}%)")

    return all_pass


def build_debug_split(records: list[dict]) -> list[dict]:
    """Pick 1 prompt per (Subject x Style) cell."""
    seen = set()
    debug = []
    for r in records:
        key = (r["subject"], r["style"])
        if key not in seen:
            seen.add(key)
            debug.append(r)
    return debug


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=150)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    records = load_records(input_path)
    logger.info(f"Loaded {len(records)} prompts from {input_path}")

    # Length filtering
    before = len(records)
    filtered = []
    for r in records:
        wc = len(r["text"].split())
        if r.get("lang") == "zh":
            if len(r["text"]) >= 10:
                filtered.append(r)
        else:
            if args.min_words <= wc <= args.max_words:
                filtered.append(r)
    records = filtered
    logger.info(f"After length filter: {len(records)} (removed {before - len(records)})")

    # Validation
    validate(records)

    # Save final dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(records)} prompts to {output_path}")

    # Debug split
    debug = build_debug_split(records)
    debug_path = Path(args.output).parent.parent / "debug" / "metadata.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(debug)} debug prompts to {debug_path}")


if __name__ == "__main__":
    main()
