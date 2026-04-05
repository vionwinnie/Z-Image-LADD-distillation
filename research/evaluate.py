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

    # FID via torch-fidelity (GPU-native, fast)
    fid = -1.0
    teacher_dir = os.path.join(_project_root, TEACHER_IMAGES_DIR)
    if os.path.isdir(teacher_dir):
        # Copy matching count of teacher images to temp dir (avoid corrupt partial files)
        import shutil, tempfile
        teacher_all = sorted([f for f in os.listdir(teacher_dir)
                              if f.endswith(".png") and os.path.getsize(os.path.join(teacher_dir, f)) > 0])
        n = min(len(image_paths), len(teacher_all))
        if n >= 10:
            tmpdir = tempfile.mkdtemp(prefix="fid_teacher_")
            student_tmpdir = tempfile.mkdtemp(prefix="fid_student_")
            try:
                for f in teacher_all[:n]:
                    shutil.copy2(os.path.join(teacher_dir, f), tmpdir)
                for p in image_paths[:n]:
                    shutil.copy2(p, student_tmpdir)
                logger.info(f"Computing FID: {n} student vs {n} teacher images...")
                import torch_fidelity
                metrics = torch_fidelity.calculate_metrics(
                    input1=student_tmpdir, input2=tmpdir,
                    cuda=True, fid=True, verbose=False,
                )
                fid = metrics["frechet_inception_distance"]
                logger.info(f"FID = {fid:.2f}")
            except Exception as e:
                logger.warning(f"FID computation failed: {e}")
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
                shutil.rmtree(student_tmpdir, ignore_errors=True)
        else:
            logger.warning(f"Only {n} valid image pairs, need >= 10 for FID")
    else:
        logger.warning(f"Teacher images not found at {teacher_dir}")

    # Print standardized metrics block
    print("\n---")
    print(f"fid:                {fid:.4f}")
    print(f"clip_score:         -1.0000")
    print(f"num_images:         {len(image_paths)}")
    print("---")


if __name__ == "__main__":
    main()
