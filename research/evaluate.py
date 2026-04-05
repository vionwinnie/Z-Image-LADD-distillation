"""Evaluation script for the LADD research loop.

Loads a student checkpoint, generates images on fixed eval prompts,
computes KID (Kernel Inception Distance) and optionally FID.

KID is preferred over FID for small sample sizes (≤1000) because it is
an unbiased estimator — FID requires thousands of samples for a stable
covariance estimate.

Usage:
    python research/evaluate.py --checkpoint research/output/checkpoint-500
"""

import argparse
import json
import logging
import os
import random
import shutil
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
NUM_EVAL_IMAGES = 1000
NUM_INFERENCE_STEPS = 4
IMAGE_SIZE = 512
SEED = 12345
DEVICE = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LADD checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint dir (contains student_transformer/).")
    parser.add_argument("--num_images", type=int, default=NUM_EVAL_IMAGES)
    parser.add_argument("--teacher_image_dir", type=str, default=None,
                        help="Dir with cached teacher images. Generated if missing.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(_project_root)

    from training.ladd_eval import generate_eval_images

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

    # KID: Kernel Inception Distance (unbiased, works well with small sample sizes)
    kid_mean = -1.0
    kid_std = -1.0
    try:
        from torch_fidelity import calculate_metrics

        student_image_dir = os.path.dirname(image_paths[0])
        logger.info(f"Computing KID: {len(image_paths)} student images from {student_image_dir}")

        # Teacher reference images — generate if not cached
        teacher_image_dir = args.teacher_image_dir or os.path.join(
            _project_root, "data", "val", "teacher_images"
        )
        teacher_images_exist = (
            os.path.isdir(teacher_image_dir)
            and len([f for f in os.listdir(teacher_image_dir) if f.endswith(".png")]) >= len(val_prompts)
        )
        if not teacher_images_exist:
            logger.info(f"Generating {len(val_prompts)} teacher reference images...")
            # Use a dummy checkpoint dir so generate_eval_images uses base model weights
            dummy_ckpt = os.path.join(_project_root, "_teacher_dummy_ckpt")
            os.makedirs(os.path.join(dummy_ckpt, "student_transformer"), exist_ok=True)
            generate_eval_images(
                checkpoint_path=dummy_ckpt,
                model_dir=MODEL_DIR,
                prompts=val_prompts,
                output_dir=os.path.dirname(teacher_image_dir),
                step=0,
                num_inference_steps=NUM_INFERENCE_STEPS,
                image_size=IMAGE_SIZE,
                device=DEVICE,
                seed=SEED,
            )
            # generate_eval_images saves to <output_dir>/eval_images/step_000000/
            generated_dir = os.path.join(os.path.dirname(teacher_image_dir), "eval_images", "step_000000")
            if os.path.isdir(generated_dir) and generated_dir != teacher_image_dir:
                os.makedirs(teacher_image_dir, exist_ok=True)
                for f in os.listdir(generated_dir):
                    if f.endswith(".png"):
                        shutil.move(os.path.join(generated_dir, f), os.path.join(teacher_image_dir, f))
                shutil.rmtree(os.path.join(os.path.dirname(teacher_image_dir), "eval_images"), ignore_errors=True)
            shutil.rmtree(dummy_ckpt, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()

        kid_subset_size = min(100, len(image_paths))
        metrics = calculate_metrics(
            input1=student_image_dir,
            input2=teacher_image_dir,
            cuda=True,
            kid=True,
            kid_subset_size=kid_subset_size,
            kid_subsets=100,
        )
        kid_mean = metrics["kernel_inception_distance_mean"]
        kid_std = metrics["kernel_inception_distance_std"]
        logger.info(f"KID = {kid_mean:.6f} ± {kid_std:.6f}")
    except Exception as e:
        logger.warning(f"KID computation failed: {e}")
        import traceback
        traceback.print_exc()

    # Print standardized metrics block
    print("\n---")
    print(f"kid_mean:           {kid_mean:.6f}")
    print(f"kid_std:            {kid_std:.6f}")
    print(f"num_images:         {len(image_paths)}")
    print("---")


if __name__ == "__main__":
    main()
