"""Evaluation script for the LADD research loop — IMMUTABLE.

Loads a student checkpoint, generates images on fixed eval prompts,
computes FID and CLIP score, and prints a standardized metrics block.

Usage:
    python research/evaluate.py --checkpoint research/output/checkpoint-500

The research agent must NOT modify this file.
"""

import argparse
import json
import logging
import os
import random
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
VAL_DATA = "data/val/metadata.json"
TEACHER_IMAGES_DIR = "data/val/teacher_images"
FID_REFERENCE_STATS = "data/val/fid_reference_stats.npz"
NUM_EVAL_IMAGES = 50
NUM_INFERENCE_STEPS = 4
IMAGE_SIZE = 512
SEED = 12345
DEVICE = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LADD checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint dir (contains student_transformer/).")
    parser.add_argument("--num_images", type=int, default=NUM_EVAL_IMAGES)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(_project_root)

    from training.ladd_eval import generate_eval_images, compute_clip_score

    # Load val prompts (deterministic subset)
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
        ]

    rng = random.Random(SEED)
    if args.num_images < len(all_prompts):
        indices = rng.sample(range(len(all_prompts)), args.num_images)
        val_prompts = [all_prompts[i] for i in sorted(indices)]
    else:
        val_prompts = all_prompts[:args.num_images]

    logger.info(f"Evaluating: {args.checkpoint}")
    logger.info(f"Generating {len(val_prompts)} images at {IMAGE_SIZE}x{IMAGE_SIZE}")

    # Generate images
    image_paths = generate_eval_images(
        checkpoint_path=args.checkpoint,
        model_dir=MODEL_DIR,
        prompts=val_prompts,
        output_dir=os.path.dirname(args.checkpoint),
        step=0,
        num_inference_steps=NUM_INFERENCE_STEPS,
        image_size=IMAGE_SIZE,
        device=DEVICE,
        seed=SEED,
    )
    logger.info(f"Generated {len(image_paths)} images")

    # Free models for metric computation
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # FID: extract student Inception features, compute against precomputed stats.
    fid = -1.0
    ref_stats_path = os.path.join(_project_root, FID_REFERENCE_STATS)
    if os.path.exists(ref_stats_path):
        try:
            import numpy as np
            from scipy import linalg
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

            ref = np.load(ref_stats_path)
            mu_ref = ref["mu"]
            sigma_ref = ref["sigma"]
            ref_n = int(ref["num_samples"])

            # Extract student Inception features on GPU
            from torchvision import transforms
            from PIL import Image

            logger.info(f"Computing FID: {len(image_paths)} student vs {ref_n} reference...")
            inception = FeatureExtractorInceptionV3(name="inception-v3-compat",
                                                    features_list=["2048"], cuda=True)
            t = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

            all_features = []
            for i in range(0, len(image_paths), 16):
                batch = [t(Image.open(p).convert("RGB")) for p in image_paths[i:i+16]]
                batch_tensor = torch.stack(batch).cuda()
                with torch.no_grad():
                    feat = inception(batch_tensor)["2048"]
                all_features.append(feat.cpu())
            features = torch.cat(all_features).numpy()

            mu_gen = np.mean(features, axis=0)
            sigma_gen = np.cov(features, rowvar=False)

            diff = mu_ref - mu_gen
            covmean, _ = linalg.sqrtm(sigma_ref @ sigma_gen, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = float(diff @ diff + np.trace(sigma_ref + sigma_gen - 2 * covmean))
            logger.info(f"FID = {fid:.2f}")
            del inception, features
        except Exception as e:
            logger.warning(f"FID computation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning(f"FID reference stats not found at {ref_stats_path}")

    # Print standardized metrics block
    print("\n---")
    print(f"fid:                {fid:.4f}")
    print(f"clip_score:         -1.0000")
    print(f"num_images:         {len(image_paths)}")
    print("---")


if __name__ == "__main__":
    main()
