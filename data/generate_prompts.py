#!/usr/bin/env python3
"""
Read the coverage matrix from prepare_prompts.py output, identify gaps,
and generate prompts to fill them using Claude API.

Usage:
    python data/generate_prompts.py              # generate prompts (skip if already done)
    python data/generate_prompts.py --force      # regenerate even if output exists
    python data/generate_prompts.py --dry-run    # just print coverage gaps
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "all_classified_prompts.json"
TRAIN_DIR = SCRIPT_DIR / "train"
TRAIN_META = TRAIN_DIR / "metadata.json"
DEBUG_DIR = SCRIPT_DIR / "debug"
DEBUG_META = DEBUG_DIR / "metadata.json"
COVERAGE_CSV = SCRIPT_DIR / "coverage_matrix.csv"

# ---------------------------------------------------------------------------
# Taxonomy (mirrored from prepare_prompts.py)
# ---------------------------------------------------------------------------
SUBJECTS = {
    "S1": "People/Portraits",
    "S2": "Animals",
    "S3": "Food&Beverage",
    "S4": "IndoorScenes",
    "S5": "Outdoor/Landscape",
    "S6": "Architecture/Urban",
    "S7": "Vehicles",
    "S8": "Plants/Nature",
    "S9": "Fashion/Clothing",
    "S10": "Objects/Artifacts",
    "S11": "Text/Typography",
    "S12": "WorldKnowledge",
    "S13": "ChineseCultural",
    "S14": "Abstract/Imagination",
}

STYLES = {
    "T1": "Photorealistic",
    "T2": "TraditionalArt",
    "T3": "DigitalIllustration",
    "T4": "3D/CGI",
    "T5": "Cinematic/Film",
    "T6": "GraphicDesign",
    "T7": "Mixed/Experimental",
}

CAMERAS = {
    "C1": "Standard/Eye-level",
    "C2": "Macro/Close-up",
    "C3": "WideAngle/Panoramic",
    "C4": "Aerial/BirdsEye",
    "C5": "LowAngle/WormsEye",
    "C6": "Bokeh/ShallowDOF",
    "C7": "LongExposure/Motion",
    "C8": "DramaticLighting",
}

TARGET_PER_CELL = 100
TOTAL_TARGET = len(SUBJECTS) * len(STYLES) * TARGET_PER_CELL  # 9,800
CHINESE_TARGET = 500
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
BATCH_SIZE = 25  # prompts to generate per API call


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------
def compute_coverage(prompts: list[dict]) -> dict[str, dict[str, int]]:
    """Count per (subject x style) cell."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in prompts:
        matrix[p["subject"]][p["style"]] += 1
    return matrix


def compute_gaps(matrix: dict[str, dict[str, int]]) -> list[tuple[str, str, int]]:
    """Return list of (subject, style, needed) for cells below target."""
    gaps = []
    for subj in sorted(SUBJECTS.keys(), key=lambda x: int(x[1:])):
        for sty in sorted(STYLES.keys(), key=lambda x: int(x[1:])):
            current = matrix.get(subj, {}).get(sty, 0)
            if current < TARGET_PER_CELL:
                gaps.append((subj, sty, TARGET_PER_CELL - current))
    return gaps


def assign_camera() -> str:
    """Assign camera technique: 70% C1, 30% random from C2-C8."""
    if random.random() < 0.7:
        return "C1"
    return random.choice(["C2", "C3", "C4", "C5", "C6", "C7", "C8"])


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def print_coverage(matrix: dict[str, dict[str, int]]) -> None:
    """Print coverage matrix summary."""
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


def print_gaps(gaps: list[tuple[str, str, int]]) -> None:
    """Print gap summary."""
    total_needed = sum(n for _, _, n in gaps)
    print(f"\nCoverage gaps: {len(gaps)} cells need {total_needed} total prompts")
    print(f"Target: {TARGET_PER_CELL} per cell, {TOTAL_TARGET} total")
    print(f"\nTop 20 largest gaps:")
    for subj, sty, needed in sorted(gaps, key=lambda x: -x[2])[:20]:
        print(f"  {subj} ({SUBJECTS[subj]}) x {sty} ({STYLES[sty]}): need {needed}")


# ---------------------------------------------------------------------------
# Claude API prompt generation
# ---------------------------------------------------------------------------
def generate_prompts_for_cell(
    client,
    subject_code: str,
    style_code: str,
    n: int,
    language: str = "en",
) -> list[str]:
    """Generate n prompts for a given (subject, style) cell using Claude."""
    subject_name = SUBJECTS[subject_code]
    style_name = STYLES[style_code]

    lang_instruction = ""
    if language == "zh":
        lang_instruction = (
            "IMPORTANT: Write all prompts in Chinese (Simplified Chinese). "
            "The prompts should be natural Chinese descriptions, not translations."
        )

    user_message = f"""Generate exactly {n} diverse text-to-image prompts for the following category:

Subject: {subject_code} = {subject_name}
Style: {style_code} = {style_name}
{lang_instruction}

Requirements:
- Each prompt should be 10-40 words (or equivalent in Chinese)
- Prompts must be diverse in content, composition, and detail
- Prompts should naturally fit both the subject category and the style
- Include specific details (colors, textures, lighting, mood, composition)
- Do NOT number the prompts
- Output ONLY the prompts, one per line, with no other text or explanation

Generate {n} prompts now:"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text.strip()
            # Parse: one prompt per line, skip empty lines
            lines = [
                line.strip()
                for line in text.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            # Remove numbering if present (e.g., "1. ", "1) ")
            cleaned = []
            for line in lines:
                line = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
                if line:
                    cleaned.append(line)
            return cleaned
        except Exception as e:
            logger.warning(
                f"  API error (attempt {attempt + 1}/{MAX_RETRIES}) for "
                f"{subject_code}x{style_code}: {e}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    logger.error(f"  Failed to generate for {subject_code}x{style_code} after {MAX_RETRIES} retries")
    return []


def generate_chinese_prompts(client, n: int = CHINESE_TARGET) -> list[dict]:
    """Generate Chinese-language prompts focused on S12 and S13."""
    results = []
    # Split: ~250 for S12 (WorldKnowledge), ~250 for S13 (ChineseCultural)
    for subject_code, count in [("S12", n // 2), ("S13", n - n // 2)]:
        remaining = count
        while remaining > 0:
            batch = min(remaining, BATCH_SIZE)
            # Pick a random style for variety
            style_code = random.choice(list(STYLES.keys()))
            logger.info(
                f"  Generating {batch} Chinese prompts for {subject_code}x{style_code}..."
            )
            prompts = generate_prompts_for_cell(
                client, subject_code, style_code, batch, language="zh"
            )
            for p in prompts:
                camera = assign_camera()
                results.append({
                    "text": p,
                    "subject": subject_code,
                    "style": style_code,
                    "camera": camera,
                    "source": "generated_claude",
                    "language": "zh",
                })
            remaining -= len(prompts)
            if not prompts:
                break  # avoid infinite loop on repeated failure
    return results


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_coverage_csv(matrix: dict[str, dict[str, int]]) -> None:
    """Save coverage matrix as CSV."""
    style_codes = sorted(STYLES.keys(), key=lambda x: int(x[1:]))
    subject_codes = sorted(SUBJECTS.keys(), key=lambda x: int(x[1:]))

    lines = ["subject," + ",".join(style_codes)]
    for subj in subject_codes:
        values = [str(matrix.get(subj, {}).get(sty, 0)) for sty in style_codes]
        lines.append(f"{subj}," + ",".join(values))

    COVERAGE_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Saved coverage matrix to {COVERAGE_CSV}")


def build_debug_split(prompts: list[dict]) -> list[dict]:
    """Pick 1 prompt per (subject x style) cell."""
    seen: set[tuple[str, str]] = set()
    debug = []
    for p in prompts:
        key = (p["subject"], p["style"])
        if key not in seen:
            seen.add(key)
            debug.append(p)
    return debug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate prompts to fill coverage gaps")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    parser.add_argument("--dry-run", action="store_true", help="Just print gaps, don't generate")
    args = parser.parse_args()

    # Check idempotency
    if TRAIN_META.exists() and not args.force and not args.dry_run:
        logger.info(
            f"{TRAIN_META} already exists. Use --force to regenerate, "
            f"or --dry-run to inspect coverage."
        )
        return

    # Load existing prompts
    if not INPUT_PATH.exists():
        logger.error(
            f"{INPUT_PATH} not found. Run 'python data/prepare_prompts.py' first."
        )
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    logger.info(f"Loaded {len(prompts)} prompts from {INPUT_PATH}")

    # Compute coverage
    matrix = compute_coverage(prompts)
    print_coverage(matrix)

    gaps = compute_gaps(matrix)
    print_gaps(gaps)

    if args.dry_run:
        logger.info("Dry run complete. No prompts generated.")
        return

    # Initialize Claude client
    try:
        import anthropic
    except ImportError:
        logger.error("The 'anthropic' package is required. Install with: pip install anthropic")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Generate prompts for each gap
    total_generated = 0
    for subj, sty, needed in gaps:
        remaining = needed
        while remaining > 0:
            batch = min(remaining, BATCH_SIZE)
            logger.info(
                f"Generating {batch} prompts for {subj} ({SUBJECTS[subj]}) x "
                f"{sty} ({STYLES[sty]})... ({remaining} remaining)"
            )
            new_prompts = generate_prompts_for_cell(client, subj, sty, batch)
            for p in new_prompts:
                camera = assign_camera()
                prompts.append({
                    "text": p,
                    "subject": subj,
                    "style": sty,
                    "camera": camera,
                    "source": "generated_claude",
                    "language": "en",
                })
                total_generated += 1
            remaining -= len(new_prompts)
            if not new_prompts:
                logger.warning(f"  Stopping generation for {subj}x{sty} due to failures")
                break

    logger.info(f"\nGenerated {total_generated} English prompts")

    # Generate Chinese prompts
    logger.info(f"\nGenerating ~{CHINESE_TARGET} Chinese-language prompts...")
    chinese_prompts = generate_chinese_prompts(client, CHINESE_TARGET)
    prompts.extend(chinese_prompts)
    logger.info(f"Generated {len(chinese_prompts)} Chinese prompts")

    # Final coverage
    final_matrix = compute_coverage(prompts)
    print("\n--- FINAL COVERAGE ---")
    print_coverage(final_matrix)

    # Save outputs
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_META, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(prompts)} prompts to {TRAIN_META}")

    # Debug split
    debug = build_debug_split(prompts)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_META, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(debug)} debug prompts to {DEBUG_META}")

    # Coverage CSV
    save_coverage_csv(final_matrix)

    # Summary
    total = len(prompts)
    en_count = sum(1 for p in prompts if p.get("language") == "en")
    zh_count = sum(1 for p in prompts if p.get("language") == "zh")
    logger.info(f"\nFinal dataset: {total} prompts ({en_count} English, {zh_count} Chinese)")


if __name__ == "__main__":
    main()
