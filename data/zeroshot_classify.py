#!/usr/bin/env python3
"""
Zero-shot classification using embedding similarity.

Embeds category label descriptions, then assigns each prompt to the
nearest category by cosine similarity. Uses the same all-MiniLM-L6-v2
model from the dedup pipeline.

Input:  data/deduped_stage2.jsonl
Output: data/full_batch.jsonl (overwritten with new classifications)
"""

import json
import logging
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "deduped_stage2.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "full_batch.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 512
MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# Category descriptions for zero-shot — multiple phrasings per category
# to create a richer embedding centroid
# ---------------------------------------------------------------------------

SUBJECT_DESCRIPTIONS = {
    "S1": [
        "a portrait of a person, human face, man, woman, child",
        "people interacting, group of humans, family portrait",
        "selfie, facial expression, human body, elderly person",
        "a photo of a person standing, sitting, walking",
    ],
    "S2": [
        "an animal, dog, cat, bird, wildlife, pet",
        "a lion in the savanna, fish swimming, horse galloping",
        "butterfly, insect, snake, marine life, deer in forest",
        "cute puppy, exotic bird, endangered species",
    ],
    "S3": [
        "food and beverage, meal on a plate, delicious dish",
        "coffee cup, wine glass, sushi platter, fresh fruit",
        "a cake with frosting, pizza slice, gourmet cooking",
        "restaurant table setting, breakfast, dessert",
    ],
    "S4": [
        "interior of a room, indoor scene, furniture",
        "living room with sofa, kitchen countertop, bedroom",
        "office workspace, library shelves, museum gallery",
        "restaurant interior, bathroom, cozy indoor space",
    ],
    "S5": [
        "outdoor landscape, mountain scenery, ocean view",
        "sunset over the sea, forest trail, desert dunes",
        "countryside meadow, river flowing, beach at dawn",
        "snowy peaks, rolling hills, dramatic sky over valley",
    ],
    "S6": [
        "architecture, building exterior, city skyline, urban street",
        "skyscraper, bridge, cathedral, ancient temple ruins",
        "modern glass building, old town alley, castle tower",
        "urban cityscape at night, suburban houses, monument",
    ],
    "S7": [
        "a vehicle, car on the road, airplane in the sky",
        "motorcycle, bicycle, train at station, ship at sea",
        "spaceship, helicopter, vintage automobile, truck",
        "racing car, sailboat, hot air balloon, subway",
    ],
    "S8": [
        "plants, flowers, tree, garden, natural vegetation",
        "a rose in bloom, sunflower field, tropical palm tree",
        "moss-covered rock, fern leaves, cherry blossom branch",
        "mushroom in forest, cactus in desert, autumn leaves",
    ],
    "S9": [
        "fashion, clothing, dress, outfit, accessories",
        "model on runway, designer handbag, stylish shoes",
        "jewelry, watch, hat, elegant evening gown",
        "streetwear outfit, vintage clothing, fashion photography",
    ],
    "S10": [
        "everyday object, tool, gadget, household item",
        "a clock on the wall, glass bottle, wooden chair",
        "musical instrument, sculpture, antique artifact",
        "electronics, kitchen utensil, decorative vase",
    ],
    "S11": [
        "text and typography, written words, lettering, sign",
        "neon sign with text, graffiti letters, book cover title",
        "calligraphy, handwritten note, poster with headline",
        "logo design with text, street sign, banner with words",
    ],
    "S12": [
        "world knowledge, famous landmark, historical event",
        "the Eiffel Tower, ancient Egyptian pyramids, globe",
        "map of the world, cultural heritage site, monument",
        "historical figure, iconic location, national flag",
    ],
    "S13": [
        "Chinese culture, dragon, pagoda, traditional Chinese art",
        "Chinese lantern festival, jade ornament, silk painting",
        "Great Wall of China, forbidden city, panda, calligraphy",
        "Chinese New Year celebration, porcelain, tea ceremony",
    ],
    "S14": [
        "abstract art, surreal dreamscape, fantasy scene",
        "geometric patterns, psychedelic colors, fractal design",
        "futuristic sci-fi scene, cosmic nebula, alien world",
        "ethereal magical landscape, impossible architecture, imaginative",
    ],
}

STYLE_DESCRIPTIONS = {
    "T1": [
        "photorealistic photograph taken with a camera, DSLR photo",
        "high resolution photography, natural real-world scene captured on camera",
        "photojournalism, candid photo, product photography on plain background",
        "realistic image of a real object or scene, not illustrated or designed",
        "The image displays a real photograph of an object or person or scene",
        "a detailed photograph showing a real-world subject against a background",
    ],
    "T2": [
        "oil painting, watercolor art, traditional brush strokes",
        "impressionist painting, renaissance artwork, ink wash",
        "charcoal sketch, pastel drawing, classical fine art",
    ],
    "T3": [
        "digital illustration, anime style, cartoon drawing",
        "vector art, pixel art, comic book style, cel shaded",
        "concept art, manga, digital painting, flat design",
    ],
    "T4": [
        "3D render, CGI, blender render, octane render",
        "isometric 3D, voxel art, low poly, clay render",
        "unreal engine, 3D modeling, computer generated imagery",
    ],
    "T5": [
        "cinematic film still, movie scene, anamorphic lens",
        "film grain, 35mm film, noir lighting, IMAX quality",
        "dramatic movie shot, technicolor, widescreen cinematic",
    ],
    "T6": [
        "graphic design poster with bold text and layout composition",
        "logo design, brand identity, flat vector icon illustration",
        "infographic with charts and diagrams, UI mockup design",
    ],
    "T7": [
        "mixed media collage combining different art techniques",
        "glitch art with digital distortion, vaporwave aesthetic",
        "generative algorithmic art, fractal pattern, kaleidoscope art",
    ],
}

CAMERA_DESCRIPTIONS = {
    "C1": [
        "standard eye-level shot, normal perspective",
        "straight-on view, neutral camera angle",
    ],
    "C2": [
        "macro close-up shot, extreme detail, micro photography",
        "tight crop on subject, detail shot, magnified view",
    ],
    "C3": [
        "wide angle panoramic shot, ultra wide lens, fisheye",
        "expansive vista, sweeping landscape view",
    ],
    "C4": [
        "aerial drone shot, bird's eye view, top-down overhead",
        "satellite view, looking straight down from above",
    ],
    "C5": [
        "low angle shot, looking up, worm's eye view",
        "shot from below, towering perspective, upward angle",
    ],
    "C6": [
        "shallow depth of field, bokeh background, blurred background",
        "portrait lens 85mm f/1.4, dreamy out of focus",
    ],
    "C7": [
        "long exposure, motion blur, light trails, silk water",
        "slow shutter speed, light painting, smooth water",
    ],
    "C8": [
        "dramatic lighting, chiaroscuro, rim light, golden hour",
        "neon glow, volumetric light, god rays, moody spotlight",
    ],
}


def load_model():
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def embed_texts(texts: list[str], tokenizer, model) -> np.ndarray:
    """Embed a list of texts, return L2-normalized embeddings."""
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def build_label_embeddings(descriptions: dict[str, list[str]], tokenizer, model) -> dict[str, np.ndarray]:
    """Embed all label descriptions and average them per category."""
    label_embs = {}
    for code, desc_list in descriptions.items():
        embs = embed_texts(desc_list, tokenizer, model)
        centroid = embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        label_embs[code] = centroid
    return label_embs


def classify_batch_with_sims(prompt_embs: np.ndarray, label_embs: dict[str, np.ndarray]):
    """Return (best_codes, similarity_matrices) for a batch."""
    codes = list(label_embs.keys())
    label_matrix = np.stack([label_embs[c] for c in codes])  # (n_labels, dim)
    sims = prompt_embs @ label_matrix.T  # (batch, n_labels)
    best_indices = sims.argmax(axis=1)
    best_codes = [codes[i] for i in best_indices]
    return best_codes, sims, codes


# ---------------------------------------------------------------------------
# Hybrid classification heuristics
# ---------------------------------------------------------------------------
MARGIN = 0.05  # Zero-shot must beat default by this much to override

# Weak T6 keywords that trigger false positives in descriptive captions
T6_WEAK_KEYWORDS = {'minimal', 'flat', 'icon', 'typography', 'sticker',
                     'logo', 'badge', 'emblem'}

# Strong design signals required for zero-shot T6 assignment
STRONG_DESIGN_SIGNALS = [
    'graphic design', 'logo design', 'poster', 'infographic',
    'banner design', 'flyer', 'brochure', 'advertisement',
    'brand identity', 'sticker design', 'ui design', 'mockup',
    'layout design', 'packaging design', 'slide', 'presentation',
    'diagram', 'chart', 'icon design', 'template', 'typography design',
]

# T7 keywords that confirm mixed/experimental style
STRONG_EXPERIMENTAL_SIGNALS = [
    'mixed media', 'collage', 'glitch', 'vaporwave', 'psychedelic',
    'generative', 'fractal', 'kaleidoscope', 'experimental',
    'abstract art', 'surreal', 'double exposure', 'datamosh',
]


def _kw_t6_is_weak(text: str) -> bool:
    """Check if keyword classifier's T6 was triggered only by weak keywords."""
    t = text.lower()
    from prepare_prompts import STYLE_KEYWORDS
    matched = [kw for kw in STYLE_KEYWORDS['T6'] if kw in t]
    return all(kw in T6_WEAK_KEYWORDS for kw in matched)


def _has_strong_design(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in STRONG_DESIGN_SIGNALS)


def _has_strong_experimental(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in STRONG_EXPERIMENTAL_SIGNALS)


def hybrid_classify_style(text: str, kw_style: str, zs_sims: np.ndarray,
                          style_codes: list[str],
                          tokenizer=None, model=None, style_matrix=None) -> str:
    """Hybrid style: for descriptive captions, use first-sentence embedding.
    For non-caption prompts, keyword + zero-shot with heuristics."""

    # For descriptive captions: use first-sentence embedding
    first_sent = extract_first_sentence_stripped(text)
    if first_sent and tokenizer is not None and model is not None and style_matrix is not None:
        sent_emb = embed_texts([first_sent], tokenizer, model)
        sent_sims = (sent_emb @ style_matrix.T)[0]
        best_idx = int(sent_sims.argmax())
        best_code = style_codes[best_idx]
        # Still apply T6/T7 keyword gating on first sentence
        if best_code == 'T6' and not _has_strong_design(first_sent):
            return 'T1'
        if best_code == 'T7' and not _has_strong_experimental(first_sent):
            return 'T1'
        return best_code

    # Non-caption: keyword matched T6 but only via weak keywords, demote
    if kw_style == 'T6' and _kw_t6_is_weak(text):
        kw_style = 'T1'

    # Trust non-default keyword matches
    if kw_style != 'T1':
        return kw_style

    # Zero-shot with margin
    best_idx = int(zs_sims.argmax())
    best_code = style_codes[best_idx]
    t1_idx = style_codes.index('T1')
    t1_sim = zs_sims[t1_idx]
    gap = zs_sims[best_idx] - t1_sim

    if best_code == 'T6' and not _has_strong_design(text):
        return 'T1'
    if best_code == 'T7' and not _has_strong_experimental(text):
        return 'T1'

    if best_code != 'T1' and gap > MARGIN:
        return best_code
    return 'T1'


_CAPTION_PREFIX_RE = re.compile(
    r'^(?:The|This)\s+image\s+(?:displays?|shows?|features?|captures?|depicts?|presents?|is)\s+',
    re.IGNORECASE
)


def extract_first_sentence_stripped(text: str) -> str | None:
    """For descriptive captions, strip 'The image displays' prefix and return first sentence.
    Returns None if text doesn't match the caption pattern."""
    m = _CAPTION_PREFIX_RE.match(text.strip())
    if not m:
        return None
    stripped = text.strip()[m.end():]
    # Take first sentence
    for delim in ['. ', '.\n']:
        idx = stripped.find(delim)
        if idx > 0:
            stripped = stripped[:idx]
            break
    return stripped.strip()


def hybrid_classify_subject(text: str, source: str, kw_subject: str,
                            zs_sims: np.ndarray, subj_codes: list[str],
                            tokenizer=None, model=None, subj_matrix=None) -> str:
    """Hybrid subject: for descriptive captions, always use stripped first-sentence
    embedding (keyword matching on full captions is unreliable — incidental word
    mentions like 'bird', 'bench', 'van' trigger wrong categories).
    For non-caption prompts, use keyword + zero-shot fallback."""

    # For descriptive captions: ALWAYS classify via first-sentence embedding
    first_sent = extract_first_sentence_stripped(text)
    if first_sent and tokenizer is not None and model is not None and subj_matrix is not None:
        sent_emb = embed_texts([first_sent], tokenizer, model)
        sent_sims = (sent_emb @ subj_matrix.T)[0]
        best_idx = int(sent_sims.argmax())
        return subj_codes[best_idx]

    # Non-caption prompts: trust non-default keyword matches
    if kw_subject != 'S10':
        return kw_subject

    # Zero-shot with margin on full text
    best_idx = int(zs_sims.argmax())
    best_code = subj_codes[best_idx]
    s10_idx = subj_codes.index('S10')
    s10_sim = zs_sims[s10_idx]
    gap = zs_sims[best_idx] - s10_sim

    if best_code != 'S10' and gap > MARGIN:
        return best_code
    return 'S10'


def hybrid_classify_camera(kw_camera: str, zs_sims: np.ndarray,
                           cam_codes: list[str]) -> str:
    """Hybrid camera: trust keyword if non-default, else zero-shot with margin."""
    if kw_camera != 'C1':
        return kw_camera

    best_idx = int(zs_sims.argmax())
    best_code = cam_codes[best_idx]
    c1_idx = cam_codes.index('C1')
    c1_sim = zs_sims[c1_idx]
    gap = zs_sims[best_idx] - c1_sim

    if best_code != 'C1' and gap > MARGIN:
        return best_code
    return 'C1'


def main():
    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    # Load records
    records = []
    with open(INPUT_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} prompts")

    # Import keyword classifiers
    from prepare_prompts import classify_subject as kw_classify_subject
    from prepare_prompts import classify_style as kw_classify_style
    from prepare_prompts import classify_camera as kw_classify_camera

    # Load model
    tokenizer, model = load_model()

    # Build label embeddings
    logger.info("Embedding category descriptions...")
    subj_label_embs = build_label_embeddings(SUBJECT_DESCRIPTIONS, tokenizer, model)
    style_label_embs = build_label_embeddings(STYLE_DESCRIPTIONS, tokenizer, model)
    camera_label_embs = build_label_embeddings(CAMERA_DESCRIPTIONS, tokenizer, model)

    subj_codes = list(subj_label_embs.keys())
    subj_matrix = np.stack([subj_label_embs[c] for c in subj_codes])
    style_codes = list(style_label_embs.keys())
    style_matrix = np.stack([style_label_embs[c] for c in style_codes])
    cam_codes = list(camera_label_embs.keys())
    cam_matrix = np.stack([camera_label_embs[c] for c in cam_codes])

    logger.info(f"  Subject: {len(subj_codes)} labels, Style: {len(style_codes)}, Camera: {len(cam_codes)}")

    # Classify all prompts in batches
    n = len(records)

    for i in tqdm(range(0, n, BATCH_SIZE), desc="Classifying"):
        batch = records[i:i+BATCH_SIZE]
        batch_texts = [r["text"] for r in batch]
        batch_embs = embed_texts(batch_texts, tokenizer, model)

        # Compute similarities for all axes
        subj_sims = batch_embs @ subj_matrix.T
        style_sims = batch_embs @ style_matrix.T
        cam_sims = batch_embs @ cam_matrix.T

        for j, r in enumerate(batch):
            text = r["text"]
            source = r.get("source", "")

            # Keyword classifications
            kw_s = kw_classify_subject(text, "", source)
            kw_t = kw_classify_style(text)
            kw_c = kw_classify_camera(text)

            # Hybrid
            r["subject"] = hybrid_classify_subject(text, source, kw_s, subj_sims[j], subj_codes,
                                                    tokenizer, model, subj_matrix)
            r["style"] = hybrid_classify_style(text, kw_t, style_sims[j], style_codes,
                                                   tokenizer, model, style_matrix)
            r["camera"] = hybrid_classify_camera(kw_c, cam_sims[j], cam_codes)

    # Print distribution
    subj_counts = Counter(r["subject"] for r in records)
    style_counts = Counter(r["style"] for r in records)

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

    print("\n=== Subject Distribution (Zero-Shot) ===")
    for k in sorted(SUBJECTS.keys(), key=lambda x: int(x[1:])):
        cnt = subj_counts.get(k, 0)
        print(f"  {k} {SUBJECTS[k]:>25s}: {cnt:>8} ({cnt/n*100:.1f}%)")

    print("\n=== Style Distribution (Zero-Shot) ===")
    for k in sorted(STYLES.keys(), key=lambda x: int(x[1:])):
        cnt = style_counts.get(k, 0)
        print(f"  {k} {STYLES[k]:>25s}: {cnt:>8} ({cnt/n*100:.1f}%)")

    # Coverage matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for r in records:
        matrix[r["subject"]][r["style"]] += 1

    print("\n=== Coverage Matrix: Subject x Style ===")
    style_codes = sorted(STYLES.keys(), key=lambda x: int(x[1:]))
    subj_codes = sorted(SUBJECTS.keys(), key=lambda x: int(x[1:]))
    header = f"{'':>6}" + "".join(f"{s:>8}" for s in style_codes) + f"{'TOTAL':>8}"
    print(header)
    for subj in subj_codes:
        row = f"{subj:>6}"
        row_total = 0
        for sty in style_codes:
            c = matrix[subj][sty]
            row_total += c
            row += f"{c:>8}"
        row += f"{row_total:>8}"
        print(row)
    totals = f"{'TOTAL':>6}"
    for sty in style_codes:
        totals += f"{sum(matrix[s][sty] for s in subj_codes):>8}"
    totals += f"{n:>8}"
    print(totals)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"\nSaved {len(records)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
