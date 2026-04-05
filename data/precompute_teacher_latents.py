#!/usr/bin/env python3
"""Precompute teacher latents for LADD training.

Runs the teacher transformer through 50-step CFG=5 denoising to produce
clean latent predictions (x0). Supports batched generation and multi-GPU
sharding for parallel precompute across 8 GPUs.

Each GPU processes its shard of prompts independently (no communication).
Saves one .pt file per prompt containing the latent tensor.

Usage (single GPU):
    python data/precompute_teacher_latents.py \
        --model_dir models/Z-Image \
        --data_meta data/train/metadata_subsample.json \
        --output_dir data/train/teacher_latents \
        --batch_size 4

Usage (multi-GPU via launch script):
    bash scripts/precompute_launch.sh
"""

import argparse
import json
import os
import sys
import time

import torch

# Add src to path for local imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_project_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _project_root)

from zimage.pipeline import generate, calculate_shift, retrieve_timesteps
from zimage.scheduler import FlowMatchEulerDiscreteScheduler
from training.ladd_model_utils import load_transformer, load_vae


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute teacher latents for LADD training.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to Z-Image model directory.")
    parser.add_argument("--data_meta", type=str, required=True,
                        help="Path to JSON metadata file with text prompts.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save .pt latent files.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images to generate in parallel.")
    parser.add_argument("--rank", type=int, default=0,
                        help="GPU rank (0-indexed) for multi-GPU sharding.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of GPUs.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image resolution (height=width).")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Maximum token length for text encoder.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    # When launched with CUDA_VISIBLE_DEVICES=$rank, only 1 GPU is visible as device 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Load metadata and shard across GPUs
    with open(args.data_meta) as f:
        metadata = json.load(f)

    # Each GPU processes its shard: prompts[rank::world_size]
    my_indices = list(range(args.rank, len(metadata), args.world_size))
    my_metadata = [metadata[i] for i in my_indices]

    os.makedirs(args.output_dir, exist_ok=True)

    # Skip already-computed latents
    todo = []
    for idx, item in zip(my_indices, my_metadata):
        out_path = os.path.join(args.output_dir, f"{idx:06d}.pt")
        if not os.path.exists(out_path):
            todo.append((idx, item))

    print(f"[Rank {args.rank}] {len(todo)} prompts to process "
          f"({len(my_metadata) - len(todo)} already done)")

    if len(todo) == 0:
        return

    # Load models
    weight_dtype = torch.bfloat16
    print(f"[Rank {args.rank}] Loading teacher transformer...")
    teacher = load_transformer(args.model_dir, weight_dtype)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.to(device)

    # Text encoder + tokenizer
    from transformers import AutoModel, AutoTokenizer
    text_encoder_dir = os.path.join(args.model_dir, "text_encoder")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, dtype=weight_dtype, trust_remote_code=True)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(device)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_dir = os.path.join(args.model_dir, "tokenizer")
    if os.path.exists(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True)

    # Scheduler
    scheduler_dir = os.path.join(args.model_dir, "scheduler")
    scheduler_config_path = os.path.join(scheduler_dir, "scheduler_config.json")
    if os.path.exists(scheduler_config_path):
        with open(scheduler_config_path) as f:
            sched_cfg = json.load(f)
    else:
        sched_cfg = {}
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        shift=sched_cfg.get("shift", 3.0),
        use_dynamic_shifting=sched_cfg.get("use_dynamic_shifting", True),
    )

    # We don't need VAE — we output latents directly
    # Create a minimal mock so generate() can compute vae_scale_factor
    class _VaeMock:
        class config:
            block_out_channels = (128, 256, 512, 512)
            scaling_factor = 0.18215
            shift_factor = 0.0
    vae_mock = _VaeMock()

    # Seed
    generator = torch.Generator(device=device).manual_seed(args.seed + args.rank)

    # Process in batches
    t_start = time.time()
    processed = 0
    for batch_start in range(0, len(todo), args.batch_size):
        batch = todo[batch_start : batch_start + args.batch_size]
        indices = [item[0] for item in batch]
        prompts = [item[1]["text"] for item in batch]

        latents = generate(
            transformer=teacher,
            vae=vae_mock,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=prompts,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
            output_type="latent",
        )

        # Save individual latents
        for i, idx in enumerate(indices):
            out_path = os.path.join(args.output_dir, f"{idx:06d}.pt")
            torch.save(latents[i].cpu(), out_path)

        processed += len(batch)
        elapsed = time.time() - t_start
        rate = processed / elapsed
        remaining = (len(todo) - processed) / rate if rate > 0 else 0
        print(f"[Rank {args.rank}] {processed}/{len(todo)} "
              f"({rate:.1f} img/s, ~{remaining/60:.0f}min remaining)")

    elapsed = time.time() - t_start
    print(f"[Rank {args.rank}] Done. {processed} latents in {elapsed/60:.1f}min "
          f"({processed/elapsed:.2f} img/s)")


if __name__ == "__main__":
    main()
