#!/usr/bin/env python3
"""
t-SNE visualization of prompt embeddings with category centroids.

Embeds a random subsample of prompts, runs t-SNE, and produces a 2D scatter
plot colored by Subject (or Style), with category centroid markers overlaid.

Usage:
    python3.13 data/scripts/visualize_tsne.py [--n-samples 30000] [--axis subject|style]
"""

import json
import logging
import argparse
import random
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from openTSNE import TSNE
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
REPO_ROOT = DATA_DIR.parent
ASSETS_DIR = REPO_ROOT / "assets"
INPUT_PATH = DATA_DIR / "full_batch.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 512
MAX_LENGTH = 128

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

# Distinct colors for categories
SUBJECT_COLORS = {
    "S1": "#e6194b", "S2": "#3cb44b", "S3": "#ffe119", "S4": "#4363d8",
    "S5": "#f58231", "S6": "#911eb4", "S7": "#42d4f4", "S8": "#f032e6",
    "S9": "#bfef45", "S10": "#fabed4", "S11": "#469990", "S12": "#dcbeff",
    "S13": "#9A6324", "S14": "#800000",
}

STYLE_COLORS = {
    "T1": "#e6194b", "T2": "#3cb44b", "T3": "#4363d8",
    "T4": "#f58231", "T5": "#911eb4", "T6": "#42d4f4",
    "T7": "#f032e6",
}


def load_model():
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--axis", choices=["subject", "style"], default="subject")
    parser.add_argument("--perplexity", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(ASSETS_DIR / f"tsne_{args.axis}.png")

    # Load data
    logger.info(f"Loading data from {INPUT_PATH}...")
    with open(INPUT_PATH) as f:
        records = [json.loads(l) for l in f if l.strip()]
    logger.info(f"Loaded {len(records)} prompts")

    # Balanced sampling: equal samples per category
    random.seed(args.seed)
    if args.axis == "subject":
        group_key = "subject"
        n_groups = len(SUBJECTS)
    else:
        group_key = "style"
        n_groups = len(STYLES)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r[group_key]].append(r)

    per_group = args.n_samples // n_groups
    sampled = []
    for code in sorted(groups.keys()):
        pool = groups[code]
        random.shuffle(pool)
        sampled.extend(pool[:per_group])
    # Fill remainder from largest groups
    remaining = args.n_samples - len(sampled)
    if remaining > 0:
        leftover = [r for g in groups.values() for r in g[per_group:]]
        random.shuffle(leftover)
        sampled.extend(leftover[:remaining])

    records = sampled
    logger.info(f"Using {len(records)} balanced samples for t-SNE ({per_group} per {group_key})")

    # Choose axis
    if args.axis == "subject":
        labels = [r["subject"] for r in records]
        label_names = SUBJECTS
        colors = SUBJECT_COLORS
    else:
        labels = [r["style"] for r in records]
        label_names = STYLES
        colors = STYLE_COLORS

    # Embed
    tokenizer, model = load_model()
    n = len(records)
    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)

    for i in tqdm(range(0, n, BATCH_SIZE), desc="Embedding"):
        batch = [r["text"] for r in records[i:i+BATCH_SIZE]]
        emb = embed_texts(batch, tokenizer, model)
        embeddings[i:i+len(batch)] = emb

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Compute category centroids in embedding space
    label_codes = sorted(label_names.keys(), key=lambda x: int(x[1:]))
    centroids = {}
    for code in label_codes:
        mask = [i for i, l in enumerate(labels) if l == code]
        if mask:
            centroids[code] = embeddings[mask].mean(axis=0)

    # Add centroids to the embedding matrix for joint t-SNE
    centroid_codes = list(centroids.keys())
    centroid_embs = np.stack([centroids[c] for c in centroid_codes])
    all_embs = np.vstack([embeddings, centroid_embs])

    # t-SNE (openTSNE — much faster than sklearn for large n)
    logger.info(f"Running openTSNE (n={all_embs.shape[0]}, perplexity={args.perplexity})...")
    tsne = TSNE(perplexity=args.perplexity, random_state=args.seed,
                n_iter=750, initialization="pca", n_jobs=-1)
    coords_2d = tsne.fit(all_embs)

    prompt_coords = np.array(coords_2d[:n])
    centroid_coords = np.array(coords_2d[n:])

    # Plot
    logger.info("Plotting...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Plot prompts
    for code in label_codes:
        mask = [i for i, l in enumerate(labels) if l == code]
        if not mask:
            continue
        ax.scatter(
            prompt_coords[mask, 0], prompt_coords[mask, 1],
            c=colors[code], s=3, alpha=0.3, label=f"{code} {label_names[code]}",
            rasterized=True,
        )

    # Plot centroids
    for i, code in enumerate(centroid_codes):
        ax.scatter(
            centroid_coords[i, 0], centroid_coords[i, 1],
            c=colors[code], s=300, marker="*", edgecolors="black",
            linewidths=1.5, zorder=10,
        )
        ax.annotate(
            f"{code}", (centroid_coords[i, 0], centroid_coords[i, 1]),
            fontsize=9, fontweight="bold", ha="center", va="bottom",
            xytext=(0, 8), textcoords="offset points",
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c],
               markersize=8, label=f"{c} {label_names[c]}")
        for c in label_codes if c in [l for l in labels]
    ]
    legend_elements.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markersize=15, markeredgecolor="black", label="Centroid")
    )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
              ncol=2 if args.axis == "subject" else 1, framealpha=0.8)

    ax.set_title(f"t-SNE of {n:,} Prompts — Colored by {args.axis.title()}\n"
                 f"(all-MiniLM-L6-v2 embeddings, perplexity={args.perplexity})",
                 fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    logger.info(f"Saved to {args.output}")

    # Print counts
    print(f"\n=== {args.axis.title()} Distribution in Sample ===")
    cnt = Counter(labels)
    for c in label_codes:
        print(f"  {c} {label_names[c]:>25s}: {cnt.get(c,0):>6} ({cnt.get(c,0)/n*100:.1f}%)")


if __name__ == "__main__":
    main()
