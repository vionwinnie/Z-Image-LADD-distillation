#!/usr/bin/env python3
"""Pre-compute FID reference statistics from teacher-generated images.

Generates images for all val/test prompts using the teacher model (50 steps),
extracts Inception-v3 features, and saves (mu, sigma, num_samples) to a
.npz file for use during training eval.

This is a one-time cost (~6 hours on one GPU for 13k images per split).

Usage:
    # Val split (default):
    python scripts/precompute_fid_reference.py \
        --model_dir models/Z-Image --split val --device cuda:0

    # Test split:
    python scripts/precompute_fid_reference.py \
        --model_dir models/Z-Image --split test --device cuda:0

    # Resume from partial generation (images already on disk):
    python scripts/precompute_fid_reference.py \
        --model_dir models/Z-Image --split val \
        --skip_generation --image_dir data/val/teacher_images
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_project_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_teacher_images(
    model_dir: str,
    prompts: list[str],
    output_dir: str,
    num_inference_steps: int = 50,
    image_size: int = 512,
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 4,
    teacache_thresh: float = 0.0,
) -> list[str]:
    """Generate reference images using the teacher model."""
    from zimage.pipeline import generate
    from zimage.scheduler import FlowMatchEulerDiscreteScheduler
    from training.ladd_model_utils import load_transformer, load_vae
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16

    logger.info("Loading teacher model...")
    teacher = load_transformer(model_dir, dtype)
    teacher.to(device)
    teacher.eval()

    logger.info("Loading VAE...")
    vae = load_vae(model_dir)
    vae.to(device, dtype=torch.float32)
    vae.eval()

    logger.info("Loading text encoder...")
    text_encoder_dir = os.path.join(model_dir, "text_encoder")
    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, torch_dtype=dtype, trust_remote_code=True)
    text_encoder.to(device)
    text_encoder.eval()
    if os.path.exists(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True)

    scheduler_config_path = os.path.join(model_dir, "scheduler", "scheduler_config.json")
    if os.path.exists(scheduler_config_path):
        with open(scheduler_config_path) as f:
            sched_cfg = json.load(f)
    else:
        sched_cfg = {}
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        shift=sched_cfg.get("shift", 3.0),
        use_dynamic_shifting=sched_cfg.get("use_dynamic_shifting", False),
    )

    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    generator = torch.Generator(device).manual_seed(seed)

    # Check for existing images to enable resume
    existing = set(os.listdir(output_dir))

    logger.info(f"Generating {len(prompts)} teacher images ({num_inference_steps} steps)...")
    for start in range(0, len(prompts), batch_size):
        # Skip batches where all images already exist
        batch_indices = range(start, min(start + batch_size, len(prompts)))
        batch_names = [f"{i:05d}.png" for i in batch_indices]
        if all(name in existing for name in batch_names):
            image_paths.extend([os.path.join(output_dir, n) for n in batch_names])
            continue

        batch_prompts = prompts[start : start + batch_size]
        images = generate(
            transformer=teacher,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=batch_prompts,
            height=image_size,
            width=image_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=0,
            generator=generator,
            teacache_thresh=teacache_thresh,
        )
        for j, img in enumerate(images):
            path = os.path.join(output_dir, f"{start + j:05d}.png")
            img.save(path)
            image_paths.append(path)

        if (start // batch_size) % 25 == 0:
            logger.info(f"  Generated {start + len(batch_prompts)}/{len(prompts)} images")

    del teacher, vae, text_encoder, tokenizer, scheduler
    torch.cuda.empty_cache()

    return image_paths


class _ImageDirDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform):
        self.paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(".png") and os.path.getsize(os.path.join(image_dir, f)) > 0
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            return self.transform(Image.open(self.paths[idx]).convert("RGB"))
        except Exception:
            # Return a black image for corrupt files
            return torch.zeros(3, 299, 299)


def extract_inception_stats(
    image_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract Inception-v3 features and compute (mu, sigma)."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = _ImageDirDataset(image_dir, transform)
    logger.info(f"Extracting Inception features from {len(dataset)} images (batch={batch_size}, workers={num_workers})...")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, persistent_workers=True,
    )

    for i, imgs in enumerate(loader):
        fid.update(imgs.to(device, non_blocking=True), real=True)
        if i % 25 == 0:
            logger.info(f"  Processed {min((i + 1) * batch_size, len(dataset))}/{len(dataset)}")

    n = fid.real_features_num_samples.item()
    mu = (fid.real_features_sum / n).cpu().numpy()
    sigma = ((fid.real_features_cov_sum - n * torch.outer(
        fid.real_features_sum / n, fid.real_features_sum / n
    )) / (n - 1)).cpu().numpy()

    del fid
    torch.cuda.empty_cache()

    return mu, sigma, n


def main():
    parser = argparse.ArgumentParser(description="Pre-compute FID reference stats from teacher model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to pretrained Z-Image model directory")
    parser.add_argument("--split", type=str, default=None, choices=["val", "test"],
                        help="Convenience flag: sets --data_meta, --output, --image_dir "
                             "for the given split (data/{split}/...).")
    parser.add_argument("--data_meta", type=str, default=None,
                        help="Path to JSON annotation file (default: data/{split}/metadata.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz path (default: data/{split}/fid_reference_stats.npz)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory for teacher images (default: data/{split}/teacher_images)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--teacache_thresh", type=float, default=0.5,
                        help="TeaCache threshold for inference speedup (0=disabled, 0.5=~4.5x faster)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip image generation, only compute stats from existing images")
    args = parser.parse_args()

    # Resolve split-based defaults
    split = args.split or "val"
    data_meta = args.data_meta or f"data/{split}/metadata.json"
    output = args.output or f"data/{split}/fid_reference_stats.npz"
    image_dir = args.image_dir or f"data/{split}/teacher_images"

    # Load prompts
    with open(data_meta, "r") as f:
        records = json.load(f)
    prompts = [r["text"] for r in records]
    logger.info(f"Loaded {len(prompts)} {split} prompts from {data_meta}")

    # Generate teacher images
    if args.skip_generation:
        image_paths = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
        )
        logger.info(f"Using {len(image_paths)} existing images from {image_dir}")
    else:
        image_paths = generate_teacher_images(
            model_dir=args.model_dir,
            prompts=prompts,
            output_dir=image_dir,
            num_inference_steps=args.num_inference_steps,
            image_size=args.image_size,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            teacache_thresh=args.teacache_thresh,
        )

    # Extract Inception stats
    mu, sigma, n = extract_inception_stats(image_dir, device=args.device)

    # Save
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.savez(output, mu=mu, sigma=sigma, num_samples=n)
    logger.info(f"Saved FID reference stats ({n} samples) to {output}")


if __name__ == "__main__":
    main()
