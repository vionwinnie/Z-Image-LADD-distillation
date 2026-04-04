"""Evaluation script for the LADD research loop — IMMUTABLE.

Loads a checkpoint from research/checkpoint, generates images, computes FID
and CLIP score, and prints a standardized metrics block.

Usage:
    python research/evaluate.py

The research agent must NOT modify this file.
"""

import json
import logging
import os
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [eval]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_project_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _project_root)

# Fixed evaluation parameters
MODEL_DIR = "models/Z-Image"
CHECKPOINT_DIR = "research/checkpoint"
VAL_DATA = "data/val/metadata.json"
FID_REFERENCE = "data/val/fid_reference_stats.npz"
NUM_EVAL_IMAGES = 20        # small for fast iteration; increase for final eval
NUM_INFERENCE_STEPS = 4
IMAGE_SIZE = 512
SEED = 42
DEVICE = "cuda"


def main():
    os.chdir(_project_root)

    from training.ladd_eval import generate_eval_images, compute_clip_score

    # Load val prompts
    if os.path.exists(VAL_DATA):
        with open(VAL_DATA) as f:
            val_records = json.load(f)
        all_prompts = [r["text"] for r in val_records]
    else:
        logger.warning(f"Val data not found at {VAL_DATA}, using default prompts")
        all_prompts = [
            "A serene mountain landscape at golden hour with dramatic clouds",
            "A fluffy orange cat sleeping on a windowsill in warm sunlight",
            "A cyberpunk cityscape at night with neon reflections on wet streets",
            "A photorealistic portrait of an elderly man with kind eyes",
            "A plate of sushi arranged artistically on a wooden board",
            "An ancient Chinese pagoda in a misty bamboo forest",
            "A macro photograph of a dewdrop on a red rose petal",
            "A futuristic spacecraft orbiting a ringed planet",
            "A cozy bookstore interior with warm lighting and shelves",
            "Chinese calligraphy of the character dragon in bold brush strokes",
        ]

    # Sample fixed subset
    import random
    rng = random.Random(SEED)
    if NUM_EVAL_IMAGES < len(all_prompts):
        indices = rng.sample(range(len(all_prompts)), NUM_EVAL_IMAGES)
        val_prompts = [all_prompts[i] for i in sorted(indices)]
    else:
        val_prompts = all_prompts[:NUM_EVAL_IMAGES]

    logger.info(f"Evaluating checkpoint: {CHECKPOINT_DIR}")
    logger.info(f"Generating {len(val_prompts)} images at {IMAGE_SIZE}x{IMAGE_SIZE}, "
                f"{NUM_INFERENCE_STEPS} steps")

    # Generate images
    image_paths = generate_eval_images(
        checkpoint_path=CHECKPOINT_DIR,
        model_dir=MODEL_DIR,
        prompts=val_prompts,
        output_dir="research",
        step=0,
        num_inference_steps=NUM_INFERENCE_STEPS,
        image_size=IMAGE_SIZE,
        device=DEVICE,
        seed=SEED,
    )
    logger.info(f"Generated {len(image_paths)} images")

    # CLIP score
    logger.info("Computing CLIP score...")
    clip_score = compute_clip_score(image_paths, val_prompts, device=DEVICE)
    logger.info(f"CLIP score = {clip_score:.2f}")

    # FID
    fid = -1.0
    if os.path.exists(FID_REFERENCE):
        logger.info("Computing FID...")
        from training.ladd_eval import compute_fid
        fid = compute_fid(image_paths, FID_REFERENCE, device=DEVICE)
        logger.info(f"FID = {fid:.2f}")
    else:
        logger.warning(f"FID reference stats not found at {FID_REFERENCE}, skipping FID")

    # Print standardized metrics block
    print("\n---")
    print(f"fid:                {fid:.4f}")
    print(f"clip_score:         {clip_score:.4f}")
    print(f"num_images:         {len(image_paths)}")
    print("---")


if __name__ == "__main__":
    main()
