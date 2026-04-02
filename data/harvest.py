#!/usr/bin/env python3
"""
Phase 1: Harvest & Download Datasets for 1M prompt scaling.

Downloads each dataset, extracts text prompts, applies quality filters,
normalizes format, and saves as JSONL to data/raw/{source_name}.jsonl.

Schema: {"text": "...", "source": "...", "lang": "en"|"zh"}

Usage:
    python3.13 data/harvest.py [--source SOURCE_NAME] [--all]
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# ---------------------------------------------------------------------------
# Quality filters (applied per-prompt before saving)
# ---------------------------------------------------------------------------
BOILERPLATE = {"stock photo", "getty images", "shutterstock", "alamy", "dreamstime",
               "istock", "adobe stock", "123rf"}


def _is_cjk(text: str) -> bool:
    """Check if text is predominantly CJK."""
    cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cjk / max(len(text), 1) > 0.3


def detect_lang(text: str) -> str:
    return "zh" if _is_cjk(text) else "en"


def quality_filter(text: str, lang: str) -> bool:
    """Return True if the prompt passes quality filters."""
    if not text or not text.strip():
        return False

    text = text.strip()

    # Min length
    if lang == "zh":
        if len(text) < 15:
            return False
    else:
        wc = len(text.split())
        if wc < 8:
            return False
        if wc > 200:
            return False

    # URL-heavy
    urls = len(re.findall(r'https?://', text))
    if urls >= 2:
        return False

    # Boilerplate
    text_lower = text.lower()
    for bp in BOILERPLATE:
        if bp in text_lower:
            return False

    # ASCII-art / code (>30% non-alpha)
    alpha = sum(1 for ch in text if ch.isalpha() or ch.isspace())
    if alpha / max(len(text), 1) < 0.7:
        return False

    return True


def save_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} prompts to {path}")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Source harvesters
# ---------------------------------------------------------------------------

def harvest_diffusiondb():
    """DiffusionDB 2m_text_only: 300K sample after filter."""
    from datasets import load_dataset

    output = RAW_DIR / "diffusiondb.jsonl"
    if output.exists():
        logger.info(f"Skipping diffusiondb (already exists: {output})")
        return

    logger.info("Downloading DiffusionDB metadata-large.parquet...")
    ds = None
    for attempt_name, attempt_fn in [
        ("metadata-large.parquet", lambda: load_dataset(
            "parquet",
            data_files="hf://datasets/poloclub/diffusiondb/metadata-large.parquet",
            split="train")),
        ("metadata.parquet", lambda: load_dataset(
            "parquet",
            data_files="hf://datasets/poloclub/diffusiondb/metadata.parquet",
            split="train")),
    ]:
        try:
            logger.info(f"  Trying: {attempt_name}")
            ds = attempt_fn()
            logger.info(f"  Success! columns: {ds.column_names}, rows: {len(ds)}")
            break
        except Exception as e:
            logger.warning(f"  Failed ({attempt_name}): {e}")
            ds = None
    if ds is None:
        logger.error("All attempts to load DiffusionDB failed")
        return
    logger.info(f"  Loaded {len(ds)} rows")
    logger.info(f"  Loaded {len(ds)} rows")

    seen = set()
    records = []
    for row in ds:
        prompt = row.get("prompt", "").strip()
        nsfw_score = row.get("prompt_nsfw", 0)
        if nsfw_score and nsfw_score > 0.5:
            continue
        if not prompt or prompt.lower() in seen:
            continue
        seen.add(prompt.lower())
        lang = detect_lang(prompt)
        if quality_filter(prompt, lang):
            records.append({"text": prompt, "source": "diffusiondb", "lang": lang})
            if len(records) >= 300_000:
                break

    save_jsonl(records, output)


def harvest_recap_datacomp():
    """Recap-DataComp-1B: 250K streamed sample."""
    from datasets import load_dataset

    output = RAW_DIR / "recap_datacomp.jsonl"
    if output.exists():
        logger.info(f"Skipping recap_datacomp (already exists: {output})")
        return

    logger.info("Loading Recap-DataComp-1B via individual parquet shards...")
    # Schema mismatch between shards prevents standard loading.
    # Load individual shards with pyarrow directly.
    from huggingface_hub import HfApi
    import pyarrow.parquet as pq
    from io import BytesIO

    api = HfApi()
    try:
        files = list(api.list_repo_tree(
            "UCSC-VLAA/Recap-DataComp-1B", repo_type="dataset",
            path_in_repo="data/train_data"))
        parquet_files = [f.path for f in files if f.path.endswith(".parquet")]
        logger.info(f"  Found {len(parquet_files)} parquet shards")
    except Exception as e:
        logger.error(f"Failed to list Recap-DataComp-1B files: {e}")
        return

    seen = set()
    records = []
    # Process shards until we have enough
    for shard_idx, shard_path in enumerate(parquet_files):
        if len(records) >= 250_000:
            break
        try:
            # Load single shard ignoring schema differences
            shard_ds = load_dataset(
                "parquet",
                data_files=f"hf://datasets/UCSC-VLAA/Recap-DataComp-1B/{shard_path}",
                split="train"
            )
            for row in shard_ds:
                caption = row.get("re_caption") or row.get("caption", "")
                if not caption:
                    continue
                text = caption.strip()
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                lang = detect_lang(text)
                if quality_filter(text, lang):
                    records.append({"text": text, "source": "recap_datacomp", "lang": lang})
                    if len(records) >= 250_000:
                        break
            if shard_idx % 50 == 0:
                logger.info(f"  Processed {shard_idx+1} shards, kept {len(records)}")
        except Exception as e:
            logger.warning(f"  Shard {shard_path} failed: {e}")
            continue

    save_jsonl(records, output)


def harvest_densefusion():
    """DenseFusion-1M: 200K sample (gated, needs HF token)."""
    from datasets import load_dataset

    output = RAW_DIR / "densefusion.jsonl"
    if output.exists():
        logger.info(f"Skipping densefusion (already exists: {output})")
        return

    logger.info("Downloading DenseFusion-1M...")
    try:
        ds = load_dataset("BAAI/DenseFusion-1M", "DenseFusion-1M", split="train",
                          streaming=True, token=HF_TOKEN if HF_TOKEN else None)
    except Exception as e:
        logger.error(f"Failed to download DenseFusion: {e}")
        return

    seen = set()
    records = []
    count = 0

    for row in ds:
        count += 1
        if count % 100_000 == 0:
            logger.info(f"  Processed {count} rows, kept {len(records)}")
        text = row.get("caption", "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        lang = detect_lang(text)
        if quality_filter(text, lang):
            records.append({"text": text, "source": "densefusion", "lang": lang})
            if len(records) >= 200_000:
                break
        if count >= 1_500_000:
            break

    save_jsonl(records, output)


def harvest_sharegpt4v():
    """ShareGPT4V-PT: 100K sample."""
    from datasets import load_dataset

    output = RAW_DIR / "sharegpt4v.jsonl"
    if output.exists():
        logger.info(f"Skipping sharegpt4v (already exists: {output})")
        return

    logger.info("Downloading ShareGPT4V (ShareGPT4V-PT config)...")
    try:
        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V-PT", split="train")
    except Exception:
        logger.info("  Config failed, trying default...")
        try:
            ds = load_dataset("Lin-Chen/ShareGPT4V", split="train")
        except Exception as e:
            logger.error(f"Failed to download ShareGPT4V: {e}")
            return

    seen = set()
    records = []
    cols = ds.column_names
    logger.info(f"  Columns: {cols}")

    for row in ds:
        # Extract caption from conversations (assistant's first reply)
        text = ""
        conversations = row.get("conversations", [])
        if conversations:
            for turn in conversations:
                if turn.get("from") == "gpt" or turn.get("role") == "assistant":
                    text = turn.get("value", "") or turn.get("content", "")
                    break
        if not text:
            text = row.get("caption", "") or row.get("text", "")

        text = text.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        lang = detect_lang(text)
        if quality_filter(text, lang):
            records.append({"text": text, "source": "sharegpt4v", "lang": lang})
            if len(records) >= 100_000:
                break

    save_jsonl(records, output)


def harvest_journeydb():
    """JourneyDB: 50K (gated, NC license)."""
    from datasets import load_dataset

    output = RAW_DIR / "journeydb.jsonl"
    if output.exists():
        logger.info(f"Skipping journeydb (already exists: {output})")
        return

    if not HF_TOKEN:
        logger.warning("Skipping JourneyDB (no HF_TOKEN for gated dataset)")
        return

    logger.info("Downloading JourneyDB (non-streaming due to tar archives)...")
    try:
        ds = load_dataset("JourneyDB/JourneyDB", split="train",
                          token=HF_TOKEN, cache_dir="/workspace/.cache/huggingface/datasets")
    except Exception as e:
        logger.error(f"Failed to download JourneyDB: {e}")
        return

    # Strip MJ params
    mj_param_re = re.compile(r'\s*--\w+\s+\S*', re.IGNORECASE)

    seen = set()
    records = []
    for row in ds:
        text = row.get("prompt", "").strip()
        if not text:
            continue
        # Strip MJ params
        text = mj_param_re.sub("", text).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        lang = detect_lang(text)
        if quality_filter(text, lang):
            records.append({"text": text, "source": "journeydb", "lang": lang})
            if len(records) >= 50_000:
                break

    save_jsonl(records, output)


def harvest_anytext():
    """AnyText-3M: 30K sample (bilingual)."""
    from datasets import load_dataset

    output = RAW_DIR / "anytext.jsonl"
    if output.exists():
        logger.info(f"Skipping anytext (already exists: {output})")
        return

    logger.info("Streaming AnyText-3M...")
    try:
        ds = load_dataset("modelscope/AnyText-3M", split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed: {e}")
        return

    seen = set()
    records = []
    count = 0
    for row in ds:
        count += 1
        if count % 100_000 == 0:
            logger.info(f"  Processed {count}, kept {len(records)}")
        text = row.get("caption") or row.get("text", "")
        if not text:
            continue
        text = text.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        lang = detect_lang(text)
        if quality_filter(text, lang):
            records.append({"text": text, "source": "anytext", "lang": lang})
            if len(records) >= 30_000:
                break
        if count >= 500_000:
            break

    save_jsonl(records, output)


def harvest_sam_llava():
    """SAM-LLaVA-10M: 30K streamed sample."""
    from datasets import load_dataset

    output = RAW_DIR / "sam_llava.jsonl"
    if output.exists():
        logger.info(f"Skipping sam_llava (already exists: {output})")
        return

    logger.info("Streaming SAM-LLaVA-Captions10M...")
    try:
        ds = load_dataset("PixArt-alpha/SAM-LLaVA-Captions10M", split="train",
                          streaming=True)
    except Exception as e:
        logger.error(f"Failed: {e}")
        return

    seen = set()
    records = []
    count = 0
    for row in ds:
        count += 1
        if count % 100_000 == 0:
            logger.info(f"  Processed {count}, kept {len(records)}")
        text = row.get("txt") or row.get("caption") or row.get("text", "")
        if not text:
            continue
        text = text.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        lang = detect_lang(text)
        if quality_filter(text, lang):
            records.append({"text": text, "source": "sam_llava", "lang": lang})
            if len(records) >= 30_000:
                break
        if count >= 500_000:
            break

    save_jsonl(records, output)


def harvest_wukong():
    """Wukong: 15K Chinese (gated)."""
    from datasets import load_dataset

    output = RAW_DIR / "wukong.jsonl"
    if output.exists():
        logger.info(f"Skipping wukong (already exists: {output})")
        return

    logger.info("Streaming Wukong (wanng/wukong100m)...")
    try:
        ds = load_dataset("wanng/wukong100m", split="train", streaming=True,
                          token=HF_TOKEN if HF_TOKEN else None)
    except Exception as e:
        logger.error(f"Failed: {e}")
        return

    seen = set()
    records = []
    count = 0
    for row in ds:
        count += 1
        if count % 100_000 == 0:
            logger.info(f"  Processed {count}, kept {len(records)}")
        text = row.get("caption", "").strip()
        if not text or len(text) < 10:
            continue
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        # Wukong is all Chinese — use relaxed filter
        lang = "zh"
        # Skip URL-heavy and very short, but allow Chinese chars
        urls = len(re.findall(r'https?://', text))
        if urls >= 2:
            continue
        records.append({"text": text, "source": "wukong", "lang": lang})
        if len(records) >= 15_000:
            break
        if count >= 500_000:
            break

    save_jsonl(records, output)


def harvest_docci():
    """DOCCI: Take all (~15K). Highest quality."""
    from datasets import load_dataset

    output = RAW_DIR / "docci.jsonl"
    if output.exists():
        logger.info(f"Skipping docci (already exists: {output})")
        return

    logger.info("Downloading DOCCI descriptions directly...")
    import urllib.request
    url = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"
    try:
        logger.info(f"  Fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        lines = [json.loads(l.strip()) for l in raw.strip().split("\n") if l.strip()]
        logger.info(f"  Loaded {len(lines)} DOCCI entries")
    except Exception as e:
        logger.error(f"Failed to download DOCCI: {e}")
        return

    seen = set()
    records = []
    for entry in lines:
        text = entry.get("description", "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        lang = detect_lang(text)
        if quality_filter(text, lang):
            records.append({"text": text, "source": "docci", "lang": lang})

    save_jsonl(records, output)


def harvest_existing():
    """Existing pool from data/train/metadata.json."""
    output = RAW_DIR / "existing.jsonl"
    if output.exists():
        logger.info(f"Skipping existing (already exists: {output})")
        return

    meta_path = SCRIPT_DIR / "train" / "metadata.json"
    if not meta_path.exists():
        logger.warning(f"No existing metadata at {meta_path}")
        return

    logger.info("Loading existing prompts...")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    seen = set()
    for item in data:
        text = item.get("text", "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        lang = detect_lang(text)
        records.append({"text": text, "source": "existing", "lang": lang})

    save_jsonl(records, output)


# ---------------------------------------------------------------------------
# Cross-source exact dedup
# ---------------------------------------------------------------------------
def cross_source_dedup():
    """Deduplicate across all source JSONL files."""
    logger.info("\n=== Cross-source exact dedup ===")
    all_records = []
    for jsonl_file in sorted(RAW_DIR.glob("*.jsonl")):
        records = load_jsonl(jsonl_file)
        all_records.extend(records)
        logger.info(f"  {jsonl_file.name}: {len(records)}")

    logger.info(f"Total before dedup: {len(all_records)}")

    seen = set()
    unique = []
    for r in all_records:
        key = r["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    logger.info(f"After cross-source dedup: {len(unique)} ({len(all_records) - len(unique)} removed)")

    # Save merged output
    output = SCRIPT_DIR / "raw_merged.jsonl"
    save_jsonl(unique, output)

    # Stats
    source_counts = Counter(r["source"] for r in unique)
    lang_counts = Counter(r["lang"] for r in unique)
    logger.info("By source:")
    for src, cnt in source_counts.most_common():
        logger.info(f"  {src}: {cnt}")
    logger.info("By language:")
    for lang, cnt in lang_counts.most_common():
        logger.info(f"  {lang}: {cnt}")

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_HARVESTERS = {
    "existing": harvest_existing,
    "diffusiondb": harvest_diffusiondb,
    "recap_datacomp": harvest_recap_datacomp,
    "densefusion": harvest_densefusion,
    "sharegpt4v": harvest_sharegpt4v,
    "journeydb": harvest_journeydb,
    "anytext": harvest_anytext,
    "sam_llava": harvest_sam_llava,
    "wukong": harvest_wukong,
    "docci": harvest_docci,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Run only this source")
    parser.add_argument("--skip-gated", action="store_true",
                        help="Skip gated datasets even if token available")
    parser.add_argument("--dedup-only", action="store_true",
                        help="Only run cross-source dedup")
    args = parser.parse_args()

    if args.dedup_only:
        cross_source_dedup()
        return

    gated = {"densefusion", "journeydb", "wukong"}

    if args.source:
        if args.source in ALL_HARVESTERS:
            ALL_HARVESTERS[args.source]()
        else:
            logger.error(f"Unknown source: {args.source}. Available: {list(ALL_HARVESTERS.keys())}")
            sys.exit(1)
    else:
        for name, harvester in ALL_HARVESTERS.items():
            if args.skip_gated and name in gated:
                logger.info(f"Skipping gated dataset: {name}")
                continue
            try:
                harvester()
            except Exception as e:
                logger.error(f"Failed to harvest {name}: {e}")
                import traceback
                traceback.print_exc()

    # Run cross-source dedup
    cross_source_dedup()


if __name__ == "__main__":
    main()
