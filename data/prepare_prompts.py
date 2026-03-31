#!/usr/bin/env python3
"""
Download prompts from T2I benchmarks, deduplicate, filter, and classify
into a MECE taxonomy (Subject x Style x Camera).

Usage:
    python data/prepare_prompts.py
"""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "all_classified_prompts.json"
DEBUG_DIR = SCRIPT_DIR / "debug"
DEBUG_META = DEBUG_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# MECE Taxonomy
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

# ---------------------------------------------------------------------------
# Category mapping: source categories -> our Subject axis
# ---------------------------------------------------------------------------
# PartiPrompts categories (lowercase normalised) -> Subject code
PARTI_CATEGORY_MAP = {
    "people": "S1",
    "portrait": "S1",
    "portraits": "S1",
    "human": "S1",
    "person": "S1",
    "animal": "S2",
    "animals": "S2",
    "food": "S3",
    "food & beverage": "S3",
    "food and beverage": "S3",
    "indoor": "S4",
    "indoor scene": "S4",
    "indoor scenes": "S4",
    "outdoor": "S5",
    "outdoor scene": "S5",
    "outdoor scenes": "S5",
    "landscape": "S5",
    "architecture": "S6",
    "building": "S6",
    "buildings": "S6",
    "urban": "S6",
    "vehicle": "S7",
    "vehicles": "S7",
    "transport": "S7",
    "plant": "S8",
    "plants": "S8",
    "nature": "S8",
    "flowers": "S8",
    "fashion": "S9",
    "clothing": "S9",
    "object": "S10",
    "objects": "S10",
    "artifacts": "S10",
    "arts": "S10",
    "produce": "S3",
    "illustration": "S14",
    "illustrations": "S14",
    "abstract": "S14",
    "world knowledge": "S12",
    "text": "S11",
    "text rendering": "S11",
}

# MJHQ-30K categories (lowercase normalised) -> Subject code
MJHQ_CATEGORY_MAP = {
    "people": "S1",
    "portrait": "S1",
    "animal": "S2",
    "animals": "S2",
    "food": "S3",
    "indoor": "S4",
    "outdoor": "S5",
    "landscape": "S5",
    "logo": "S6",
    "architecture": "S6",
    "vehicle": "S7",
    "nature": "S8",
    "plant": "S8",
    "fashion": "S9",
    "object": "S10",
    "abstract": "S14",
    "art": "S14",
    "science_fiction": "S14",
    "sci-fi": "S14",
    "fantasy": "S14",
    "concept-art": "S14",
    "texture": "S10",
    "vibrant_color": "S14",
}

# ---------------------------------------------------------------------------
# Keyword classifiers for Style and Camera axes
# ---------------------------------------------------------------------------
STYLE_KEYWORDS: dict[str, list[str]] = {
    "T2": [
        "oil painting", "watercolor", "watercolour", "acrylic", "ink wash",
        "charcoal", "pencil drawing", "sketch", "pastel", "gouache",
        "impressionist", "baroque", "renaissance", "ukiyo-e", "fresco",
        "etching", "lithograph", "woodcut", "calligraphy",
    ],
    "T3": [
        "digital art", "digital illustration", "vector", "pixel art",
        "anime", "manga", "cartoon", "comic", "cel shaded", "flat design",
        "concept art", "artstation", "deviantart",
    ],
    "T4": [
        "3d render", "3d rendering", "cgi", "blender", "unreal engine",
        "octane render", "isometric", "voxel", "low poly", "clay render",
    ],
    "T5": [
        "cinematic", "film still", "movie scene", "anamorphic",
        "film grain", "35mm", "70mm", "imax", "technicolor",
        "noir", "film noir",
    ],
    "T6": [
        "graphic design", "poster", "logo", "infographic", "typography",
        "minimal", "flat", "icon", "badge", "emblem", "sticker",
    ],
    "T7": [
        "mixed media", "collage", "surreal", "glitch", "vaporwave",
        "psychedelic", "abstract", "experimental", "generative",
        "fractal", "kaleidoscope",
    ],
}

CAMERA_KEYWORDS: dict[str, list[str]] = {
    "C2": [
        "macro", "close-up", "closeup", "close up", "extreme close",
        "detail shot", "micro",
    ],
    "C3": [
        "wide angle", "wide-angle", "panoramic", "panorama", "ultra wide",
        "fisheye", "fish-eye",
    ],
    "C4": [
        "aerial", "bird's eye", "birds eye", "bird-eye", "drone",
        "satellite", "overhead", "top-down", "top down",
    ],
    "C5": [
        "low angle", "low-angle", "worm's eye", "worms eye", "worm-eye",
        "looking up", "from below",
    ],
    "C6": [
        "bokeh", "shallow depth of field", "shallow dof", "blurred background",
        "portrait lens", "85mm", "f/1.4", "f/1.8", "f1.4", "f1.8",
    ],
    "C7": [
        "long exposure", "motion blur", "light trails", "light painting",
        "slow shutter", "silk water", "milky water",
    ],
    "C8": [
        "dramatic lighting", "chiaroscuro", "rim light", "rim lighting",
        "backlit", "backlighting", "golden hour", "blue hour",
        "neon light", "neon glow", "volumetric light", "god rays",
        "spotlight", "moody lighting", "studio lighting",
    ],
}

# Subject keyword fallback (when no category mapping is available)
SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "S1": [
        "person", "people", "man", "woman", "boy", "girl", "child",
        "portrait", "face", "human", "couple", "family", "baby",
        "old man", "old woman", "elderly", "teenager", "selfie",
    ],
    "S2": [
        "animal", "dog", "cat", "bird", "fish", "horse", "lion",
        "tiger", "elephant", "bear", "rabbit", "deer", "wolf",
        "fox", "butterfly", "insect", "snake", "whale", "dolphin",
        "monkey", "owl", "eagle", "penguin", "turtle", "frog",
    ],
    "S3": [
        "food", "meal", "dish", "cake", "pizza", "sushi", "fruit",
        "coffee", "tea", "wine", "beer", "cocktail", "bread",
        "chocolate", "dessert", "salad", "soup", "burger", "ice cream",
        "beverage", "drink", "juice", "smoothie",
    ],
    "S4": [
        "indoor", "room", "kitchen", "bedroom", "bathroom", "living room",
        "office", "library", "museum", "restaurant", "cafe", "bar",
        "studio", "classroom", "hallway", "interior",
    ],
    "S5": [
        "landscape", "mountain", "ocean", "sea", "river", "lake",
        "forest", "desert", "beach", "valley", "waterfall", "sunset",
        "sunrise", "horizon", "field", "meadow", "countryside", "cliff",
        "hill", "glacier", "canyon",
    ],
    "S6": [
        "architecture", "building", "city", "skyline", "skyscraper",
        "bridge", "tower", "castle", "church", "cathedral", "temple",
        "mosque", "street", "alley", "urban",
    ],
    "S7": [
        "car", "truck", "bus", "train", "airplane", "plane", "boat",
        "ship", "bicycle", "motorcycle", "helicopter", "rocket",
        "vehicle", "spaceship", "submarine",
    ],
    "S8": [
        "plant", "tree", "flower", "garden", "leaf", "rose", "tulip",
        "sunflower", "cactus", "mushroom", "moss", "fern", "vine",
        "bamboo", "bonsai", "lotus", "cherry blossom", "orchid",
    ],
    "S9": [
        "fashion", "clothing", "dress", "suit", "hat", "shoes",
        "jewelry", "accessory", "handbag", "watch", "glasses",
        "runway", "model wearing",
    ],
    "S10": [
        "object", "artifact", "tool", "instrument", "clock", "lamp",
        "book", "bottle", "vase", "sculpture", "statue", "toy",
        "gadget", "device", "furniture", "chair", "table",
    ],
    "S11": [
        "text", "typography", "letter", "word", "sign", "banner",
        "calligraphy", "font", "handwriting", "graffiti", "neon sign",
    ],
    "S12": [
        "world", "globe", "map", "country", "landmark", "monument",
        "historical", "ancient", "civilization", "culture",
        "famous", "iconic", "heritage",
    ],
    "S13": [
        "chinese", "china", "dragon", "phoenix", "pagoda", "lantern",
        "calligraphy", "silk", "jade", "porcelain", "dynasty",
        "great wall", "forbidden city", "panda",
    ],
    "S14": [
        "abstract", "surreal", "fantasy", "dream", "imagination",
        "futuristic", "sci-fi", "science fiction", "alien",
        "mythical", "magical", "ethereal", "cosmic", "galaxy",
        "nebula", "psychedelic", "geometric", "fractal",
    ],
}


# ---------------------------------------------------------------------------
# Benchmark downloaders
# ---------------------------------------------------------------------------
def _try_import_datasets():
    """Import HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required. Install with: pip install datasets"
        )
        raise


def download_parti_prompts() -> list[dict]:
    """Download PartiPrompts from HuggingFace."""
    load_dataset = _try_import_datasets()
    try:
        logger.info("Downloading PartiPrompts...")
        ds = load_dataset("nateraw/parti-prompts", split="train")
        results = []
        for row in ds:
            prompt = row.get("Prompt") or row.get("prompt", "")
            category = row.get("Category") or row.get("category", "")
            if prompt:
                results.append({
                    "text": prompt.strip(),
                    "original_category": str(category).strip().lower(),
                    "source": "parti_prompts",
                })
        logger.info(f"  PartiPrompts: {len(results)} prompts loaded")
        return results
    except Exception as e:
        logger.warning(f"Failed to download PartiPrompts: {e}")
        return []


def download_genai_bench() -> list[dict]:
    """Download GenAI-Bench image_generation prompts from HuggingFace."""
    load_dataset = _try_import_datasets()
    try:
        logger.info("Downloading GenAI-Bench (image_generation)...")
        ds = load_dataset("TIGER-Lab/GenAI-Bench", "image_generation", split="test")
        seen = set()
        results = []
        for row in ds:
            prompt = row.get("prompt") or row.get("text", "")
            if prompt and prompt not in seen:
                seen.add(prompt)
                results.append({
                    "text": prompt.strip(),
                    "original_category": "",
                    "source": "genai_bench",
                })
        logger.info(f"  GenAI-Bench: {len(results)} unique prompts loaded")
        return results
    except Exception as e:
        logger.warning(f"Failed to download GenAI-Bench: {e}")
        return []


def download_drawbench() -> list[dict]:
    """Download DrawBench prompts."""
    load_dataset = _try_import_datasets()
    try:
        logger.info("Downloading DrawBench...")
        ds = load_dataset("sayakpaul/drawbench", split="train")
        results = []
        for row in ds:
            prompt = row.get("Prompts") or row.get("prompts") or row.get("prompt") or row.get("text", "")
            category = row.get("Category") or row.get("category", "")
            if prompt:
                results.append({
                    "text": prompt.strip(),
                    "original_category": str(category).strip().lower(),
                    "source": "drawbench",
                })
        logger.info(f"  DrawBench: {len(results)} prompts loaded")
        return results
    except Exception as e:
        logger.warning(f"Failed to download DrawBench: {e}")
        return []


def download_geneval() -> list[dict]:
    """Download GenEval prompts from the GitHub repo (JSONL, not an HF dataset).

    The benchmark lives at https://github.com/djghosh13/geneval and stores its
    evaluation metadata as ``prompts/evaluation_metadata.jsonl``.  Each line is
    a JSON object with at least ``prompt`` and ``tag`` keys.
    """
    import urllib.request

    urls = [
        "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl",
        "https://raw.githubusercontent.com/djghosh13/geneval/master/prompts/evaluation_metadata.jsonl",
    ]
    for url in urls:
        try:
            logger.info(f"  Trying GenEval: {url}")
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            results = []
            for line in raw.strip().splitlines():
                row = json.loads(line)
                prompt = row.get("prompt", "")
                tag = row.get("tag", "")
                if prompt:
                    results.append({
                        "text": prompt.strip(),
                        "original_category": tag.strip().lower(),
                        "source": "geneval",
                    })
            logger.info(f"  GenEval: {len(results)} prompts loaded")
            return results
        except Exception as e:
            logger.info(f"    Failed: {e}")
            continue
    logger.warning("  GenEval: all attempts failed, skipping")
    return []


def download_dpg_bench() -> list[dict]:
    """Download DPG-Bench dense prompts from HuggingFace."""
    load_dataset = _try_import_datasets()
    attempts = [
        ("Jialuo21/DPG-Bench", None, "train"),
        ("Jialuo21/DPG-Bench", None, "test"),
        ("Jialuo21/DPG-Bench", "default", "train"),
    ]
    for dataset_id, config, split in attempts:
        try:
            logger.info(f"  Trying DPG-Bench: {dataset_id} config={config} split={split}")
            if config:
                ds = load_dataset(dataset_id, config, split=split)
            else:
                ds = load_dataset(dataset_id, split=split)
            results = []
            for row in ds:
                prompt = row.get("prompt") or row.get("text", "")
                if prompt:
                    results.append({
                        "text": prompt.strip(),
                        "original_category": "",
                        "source": "dpg_bench",
                    })
            logger.info(f"  DPG-Bench: {len(results)} prompts loaded")
            return results
        except Exception as e:
            logger.info(f"    Failed: {e}")
            continue
    logger.warning("  DPG-Bench: all attempts failed, skipping")
    return []


def download_oneig_zh() -> list[dict]:
    """Download OneIG-Bench Chinese prompts from HuggingFace.

    The dataset ``OneIG-Bench/OneIG-Bench`` has two configs:
      - ``OneIG-Bench``    (English, column ``prompt_en``)
      - ``OneIG-Bench-ZH`` (Chinese, column ``prompt_cn``)
    Both only have a ``train`` split.
    """
    load_dataset = _try_import_datasets()
    attempts = [
        ("OneIG-Bench/OneIG-Bench", "OneIG-Bench-ZH", "train"),
    ]
    for dataset_id, config, split in attempts:
        try:
            logger.info(f"  Trying OneIG-ZH: {dataset_id} config={config} split={split}")
            ds = load_dataset(dataset_id, config, split=split)
            results = []
            for row in ds:
                # ZH config uses prompt_cn for Chinese prompts
                prompt = (
                    row.get("prompt_cn")
                    or row.get("prompt")
                    or row.get("text", "")
                )
                category = row.get("category") or row.get("class", "")
                if prompt:
                    results.append({
                        "text": prompt.strip(),
                        "original_category": str(category).strip().lower(),
                        "source": "oneig_zh",
                        "language": "zh",
                    })
            logger.info(f"  OneIG-ZH: {len(results)} prompts loaded")
            return results
        except Exception as e:
            logger.info(f"    Failed: {e}")
            continue
    logger.warning("  OneIG-ZH: all attempts failed, skipping")
    return []


def download_cvtg_2k() -> list[dict]:
    """Download CVTG-2K text rendering prompts from HuggingFace.

    The dataset ``dnkdnk/CVTG-2K`` is a zip archive (not a standard HF
    dataset).  Inside is ``CVTG-2K/CVTG/`` and ``CVTG-2K/CVTG-style/``
    containing JSON files like ``2.json`` ... ``5.json`` (fine-grained) and
    ``2_combined.json`` ... ``5_combined.json`` (simplified).  Each JSON has a
    ``data_list`` array of objects with a ``prompt`` field.

    We use the ``_combined.json`` files from the ``CVTG/`` folder (no style
    attributes) for the standalone prompt text.
    """
    import io
    import urllib.request
    import zipfile

    zip_url = (
        "https://huggingface.co/datasets/dnkdnk/CVTG-2K/resolve/main/CVTG-2K.zip"
    )
    try:
        logger.info(f"  Downloading CVTG-2K zip from {zip_url}")
        with urllib.request.urlopen(zip_url, timeout=60) as resp:
            zip_bytes = resp.read()

        results = []
        seen = set()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Prefer _combined.json from both CVTG and CVTG-style folders
            json_names = sorted(
                n for n in zf.namelist()
                if n.endswith(".json")
            )
            for name in json_names:
                try:
                    data = json.loads(zf.read(name).decode("utf-8"))
                    items = data if isinstance(data, list) else data.get("data_list", [])
                    for entry in items:
                        prompt = ""
                        if isinstance(entry, dict):
                            prompt = entry.get("prompt", "") or entry.get("combined_prompt", "")
                        elif isinstance(entry, str):
                            prompt = entry
                        if prompt and prompt not in seen:
                            seen.add(prompt)
                            results.append({
                                "text": prompt.strip(),
                                "original_category": "text rendering",
                                "source": "cvtg_2k",
                            })
                except Exception:
                    continue
        logger.info(f"  CVTG-2K: {len(results)} unique prompts loaded")
        return results
    except Exception as e:
        logger.warning(f"  CVTG-2K: download failed: {e}")
        return []


def _contains_chinese(text: str) -> bool:
    """Return True if *text* contains CJK Unified Ideograph characters."""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def download_longtext_bench() -> list[dict]:
    """Download LongText-Bench prompts from HuggingFace.

    The dataset has 320 rows (160 EN + 160 ZH) in a single ``train`` split
    with columns: ``category``, ``length``, ``prompt``, ``text`` (list),
    ``text_length``, ``prompt_id``.  There is no explicit language column, so
    we detect Chinese prompts by checking for CJK characters.
    """
    load_dataset = _try_import_datasets()
    attempts = [
        ("X-Omni/LongText-Bench", None, "train"),
        ("X-Omni/LongText-Bench", None, "test"),
        ("X-Omni/LongText-Bench", "default", "train"),
    ]
    for dataset_id, config, split in attempts:
        try:
            logger.info(f"  Trying LongText-Bench: {dataset_id} config={config} split={split}")
            if config:
                ds = load_dataset(dataset_id, config, split=split)
            else:
                ds = load_dataset(dataset_id, split=split)
            results = []
            for row in ds:
                prompt = row.get("prompt", "")
                if prompt:
                    lang = "zh" if _contains_chinese(prompt) else "en"
                    results.append({
                        "text": prompt.strip(),
                        "original_category": "text rendering",
                        "source": "longtext_bench",
                        "language": lang,
                    })
            logger.info(f"  LongText-Bench: {len(results)} prompts loaded")
            return results
        except Exception as e:
            logger.info(f"    Failed: {e}")
            continue
    logger.warning("  LongText-Bench: all attempts failed, skipping")
    return []


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _match_keywords(text_lower: str, keyword_map: dict[str, list[str]]) -> str | None:
    """Return the first matching key from keyword_map, or None."""
    for code, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text_lower:
                return code
    return None


def classify_subject(prompt: str, original_category: str, source: str) -> str:
    """Classify a prompt into a Subject axis code."""
    cat = original_category.strip().lower()

    # Try source-specific category mapping first
    if source == "parti_prompts" and cat:
        mapped = PARTI_CATEGORY_MAP.get(cat)
        if mapped:
            return mapped
    elif source == "mjhq_30k" and cat:
        mapped = MJHQ_CATEGORY_MAP.get(cat)
        if mapped:
            return mapped

    # Fallback: keyword matching on the prompt text
    text_lower = prompt.lower()
    matched = _match_keywords(text_lower, SUBJECT_KEYWORDS)
    if matched:
        return matched

    # Default to Objects/Artifacts
    return "S10"


def classify_style(prompt: str) -> str:
    """Classify a prompt into a Style axis code."""
    text_lower = prompt.lower()
    matched = _match_keywords(text_lower, STYLE_KEYWORDS)
    return matched or "T1"  # default: photorealistic


def classify_camera(prompt: str) -> str:
    """Classify a prompt into a Camera axis code."""
    text_lower = prompt.lower()
    matched = _match_keywords(text_lower, CAMERA_KEYWORDS)
    return matched or "C1"  # default: standard/eye-level


def classify_prompt(record: dict) -> dict:
    """Classify a single prompt record into (subject, style, camera)."""
    text = record["text"]
    language = record.get("language", "en")
    return {
        "text": text,
        "subject": classify_subject(text, record.get("original_category", ""), record["source"]),
        "style": classify_style(text),
        "camera": classify_camera(text),
        "source": record["source"],
        "language": language,
    }


# ---------------------------------------------------------------------------
# Deduplication & filtering
# ---------------------------------------------------------------------------
def deduplicate(records: list[dict]) -> list[dict]:
    """Deduplicate by exact string match (case-insensitive)."""
    seen: set[str] = set()
    unique = []
    for r in records:
        key = r["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def filter_short(records: list[dict], min_words: int = 5, min_chars_zh: int = 4) -> list[dict]:
    """Remove prompts with fewer than min_words words (or min_chars_zh chars for Chinese)."""
    result = []
    for r in records:
        if r.get("language") == "zh":
            # Chinese text uses characters, not whitespace-separated words
            if len(r["text"].strip()) >= min_chars_zh:
                result.append(r)
        else:
            if len(r["text"].split()) >= min_words:
                result.append(r)
    return result


# ---------------------------------------------------------------------------
# Coverage matrix
# ---------------------------------------------------------------------------
def compute_coverage(classified: list[dict]) -> dict[str, dict[str, int]]:
    """Count per (subject x style) cell."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in classified:
        matrix[r["subject"]][r["style"]] += 1
    return matrix


def print_coverage(matrix: dict[str, dict[str, int]]) -> None:
    """Print coverage matrix summary to stdout."""
    style_codes = sorted(STYLES.keys(), key=lambda x: int(x[1:]))
    subject_codes = sorted(SUBJECTS.keys(), key=lambda x: int(x[1:]))

    # Header
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

    # Totals row
    print("-" * len(header))
    totals_str = f"{'TOTAL':>6}"
    for sty in style_codes:
        col_total = sum(matrix.get(s, {}).get(sty, 0) for s in subject_codes)
        totals_str += f"{col_total:>8}"
    totals_str += f"{grand_total:>8}"
    print(totals_str)
    print("=" * len(header))
    print(f"\nTotal classified prompts: {grand_total}")


def build_debug_split(classified: list[dict]) -> list[dict]:
    """Pick 1 prompt per (subject x style) cell where available."""
    seen: set[tuple[str, str]] = set()
    debug = []
    for r in classified:
        key = (r["subject"], r["style"])
        if key not in seen:
            seen.add(key)
            debug.append(r)
    return debug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=== Prompt Preparation Pipeline ===\n")

    # Step 1: Download all benchmarks
    all_records: list[dict] = []
    for downloader in [
        download_parti_prompts,
        download_genai_bench,
        download_drawbench,
        download_geneval,
        download_dpg_bench,
        download_oneig_zh,
        download_cvtg_2k,
        download_longtext_bench,
    ]:
        records = downloader()
        all_records.extend(records)

    logger.info(f"\nTotal raw prompts: {len(all_records)}")

    # Step 2: Deduplicate
    all_records = deduplicate(all_records)
    logger.info(f"After deduplication: {len(all_records)}")

    # Step 3: Filter short prompts
    all_records = filter_short(all_records)
    logger.info(f"After filtering (<5 words removed): {len(all_records)}")

    # Step 4: Classify
    classified = [classify_prompt(r) for r in all_records]
    logger.info(f"Classified: {len(classified)} prompts")

    # Step 5: Coverage matrix
    matrix = compute_coverage(classified)
    print_coverage(matrix)

    # Step 6: Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(classified, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(classified)} prompts to {OUTPUT_PATH}")

    # Step 7: Debug split
    debug = build_debug_split(classified)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_META, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(debug)} debug prompts to {DEBUG_META}")

    # Summary by source
    source_counts = Counter(r["source"] for r in classified)
    print("\nPrompts by source:")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")

    # Summary by language
    lang_counts = Counter(r["language"] for r in classified)
    print("\nPrompts by language:")
    for lang, count in lang_counts.most_common():
        print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
