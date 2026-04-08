"""Inference with LADD-distilled student model.

Loads student weights from either:
  - Accelerate FSDP checkpoint dir (pytorch_model_fsdp.bin from accelerator.save_state)
  - DCP sharded checkpoint (from FSDP training)
  - Safetensors file (consolidated)
  - PyTorch .bin file

Usage:
    # From Accelerate FSDP checkpoint directory:
    python scripts/inference_student.py \
        --checkpoint output/ladd/checkpoint-5000 \
        --model_dir models/Z-Image \
        --prompt "A cat sitting on a windowsill"

    # From a specific .bin file:
    python scripts/inference_student.py \
        --checkpoint output/ladd/checkpoint-5000/pytorch_model_fsdp.bin \
        --model_dir models/Z-Image \
        --prompt "A cat sitting on a windowsill"

    # From consolidated safetensors:
    python scripts/inference_student.py \
        --checkpoint output/ladd/checkpoint-20000/model.safetensors \
        --model_dir models/Z-Image \
        --prompt "A cat sitting on a windowsill"

    # Consolidate checkpoint to safetensors (one-time):
    python scripts/inference_student.py \
        --checkpoint output/ladd/checkpoint-5000 \
        --model_dir models/Z-Image \
        --consolidate_only
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ladd_model_utils import _build_transformer_from_config, load_transformer, load_vae
from zimage.pipeline import generate
from zimage.scheduler import FlowMatchEulerDiscreteScheduler


def load_student_from_fsdp_bin(path: str, model_dir: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load student from Accelerate FSDP checkpoint (pytorch_model_fsdp.bin)."""
    student = _build_transformer_from_config(model_dir, dtype, device="cpu")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    # Cast to target dtype if saved in fp32
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    student.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    return student


def load_student_from_dcp(checkpoint_dir: str, model_dir: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load student from DCP sharded checkpoint (requires torch.distributed init)."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")

    student = load_transformer(model_dir, dtype)
    state_dict = {"model": student.state_dict()}
    dcp.load(state_dict, checkpoint_id=checkpoint_dir)
    student.load_state_dict(state_dict["model"])
    del state_dict

    dist.destroy_process_group()
    return student


def load_student_from_safetensors(path: str, model_dir: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load student from a single safetensors file."""
    student = load_transformer(model_dir, dtype)
    student.load_state_dict(load_file(path), strict=False)
    return student


def consolidate_checkpoint(checkpoint_dir: str, model_dir: str, dtype: torch.dtype, output_path: str = None):
    """Convert FSDP/DCP checkpoint to a single safetensors file."""
    if output_path is None:
        output_path = os.path.join(checkpoint_dir, "model.safetensors")

    # Try Accelerate FSDP .bin first
    fsdp_bin = os.path.join(checkpoint_dir, "pytorch_model_fsdp.bin")
    if os.path.exists(fsdp_bin):
        print(f"Loading Accelerate FSDP checkpoint from {fsdp_bin}...")
        student = load_student_from_fsdp_bin(fsdp_bin, model_dir, dtype)
    else:
        print(f"Loading DCP sharded checkpoint from {checkpoint_dir}...")
        student = load_student_from_dcp(checkpoint_dir, model_dir, dtype)

    print(f"Saving consolidated weights to {output_path}...")
    save_file(
        {k: v.contiguous().cpu() for k, v in student.state_dict().items()},
        output_path,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done. Saved {size_mb:.0f} MB to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Inference with LADD student model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DCP checkpoint dir or safetensors file")
    parser.add_argument("--model_dir", type=str, default="models/Z-Image",
                        help="Path to pretrained Z-Image model (for architecture + VAE + text encoder)")
    parser.add_argument("--prompt", type=str,
                        default="A beautiful sunset over the ocean with golden clouds")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=4,
                        help="Student denoising steps (distilled model uses 1-4)")
    parser.add_argument("--guidance_scale", type=float, default=0.0,
                        help="CFG scale. 0.0 for distilled student (no CFG needed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts", nargs="+", type=str, default=None,
                        help="Multiple prompts (overrides --prompt)")
    parser.add_argument("--output", type=str, default="student_output.png")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for multi-prompt mode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true", help="torch.compile for faster inference")
    parser.add_argument("--consolidate_only", action="store_true",
                        help="Only consolidate DCP checkpoint to safetensors, then exit")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (e.g. 'ladd-eval'). Enables wandb logging.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Consolidate mode
    if args.consolidate_only:
        consolidate_checkpoint(args.checkpoint, args.model_dir, dtype)
        return

    # Determine checkpoint type and load student
    is_safetensors = args.checkpoint.endswith(".safetensors")
    is_bin = args.checkpoint.endswith(".bin")
    is_dir = os.path.isdir(args.checkpoint)

    if is_dir:
        # Check for consolidated safetensors first
        consolidated_path = os.path.join(args.checkpoint, "model.safetensors")
        fsdp_bin_path = os.path.join(args.checkpoint, "pytorch_model_fsdp.bin")
        dcp_metadata = os.path.join(args.checkpoint, ".metadata")

        if os.path.exists(consolidated_path):
            print(f"Found consolidated weights: {consolidated_path}")
            is_safetensors = True
            args.checkpoint = consolidated_path
        elif os.path.exists(fsdp_bin_path):
            print(f"Found Accelerate FSDP checkpoint: {fsdp_bin_path}")
            is_bin = True
            args.checkpoint = fsdp_bin_path
        elif os.path.exists(dcp_metadata):
            pass  # handled below as DCP
        else:
            raise ValueError(f"No recognized checkpoint format in {args.checkpoint}. "
                             f"Expected model.safetensors, pytorch_model_fsdp.bin, or .metadata")

    is_dcp = os.path.isdir(args.checkpoint) and os.path.exists(os.path.join(args.checkpoint, ".metadata"))

    print(f"Loading student from {args.checkpoint}...")
    if is_safetensors:
        student = load_student_from_safetensors(args.checkpoint, args.model_dir, dtype)
    elif is_bin:
        student = load_student_from_fsdp_bin(args.checkpoint, args.model_dir, dtype)
    elif is_dcp:
        student = load_student_from_dcp(args.checkpoint, args.model_dir, dtype)
    else:
        raise ValueError(f"Cannot determine checkpoint type for {args.checkpoint}. "
                         f"Expected .safetensors file, .bin file, or DCP dir with .metadata")

    student.to(args.device)
    student.eval()
    if args.compile:
        student = torch.compile(student)

    # Load shared components
    print("Loading VAE, text encoder, scheduler...")
    vae = load_vae(args.model_dir)
    vae.to(args.device, dtype=torch.float32)
    vae.eval()

    text_encoder_dir = os.path.join(args.model_dir, "text_encoder")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, dtype=dtype, trust_remote_code=True)
    text_encoder.to(args.device)
    text_encoder.eval()

    tokenizer_dir = os.path.join(args.model_dir, "tokenizer")
    if os.path.exists(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True)

    scheduler_config_path = os.path.join(args.model_dir, "scheduler", "scheduler_config.json")
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

    # Build prompt list
    prompts = args.prompts if args.prompts else [args.prompt]

    # Setup output directory for multi-prompt mode
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    wb_run = None
    if args.wandb_project:
        if not HAS_WANDB:
            raise ImportError("wandb is required for --wandb_project. Install with: pip install wandb")
        checkpoint_name = os.path.basename(args.checkpoint.rstrip("/").rstrip(".bin").rstrip("pytorch_model_fsdp"))
        if not checkpoint_name or checkpoint_name == "pytorch_model_fsdp":
            checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint))
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"student-{checkpoint_name}-{args.num_inference_steps}step",
            config={
                "checkpoint": args.checkpoint,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "num_prompts": len(prompts),
            },
        )
        print(f"W&B run: {wb_run.url}")

    # Generate with student
    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(prompts)}] Student: {args.num_inference_steps} steps, CFG={args.guidance_scale}")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'='*60}")

        generator = torch.Generator(args.device).manual_seed(args.seed + i)
        start = time.time()
        images = generate(
            transformer=student,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        elapsed = time.time() - start

        if args.output_dir:
            out_path = os.path.join(args.output_dir, f"student_{i:03d}.png")
        elif len(prompts) > 1:
            base, ext = os.path.splitext(args.output)
            out_path = f"{base}_{i:03d}{ext}"
        else:
            out_path = args.output

        images[0].save(out_path)
        print(f"Student image saved to {out_path} ({elapsed:.2f}s)")
        all_results.append({"prompt": prompt, "image": images[0], "path": out_path, "time": elapsed})

        # Log each image to wandb
        if wb_run:
            wb_run.log({
                f"student/{i:03d}": wandb.Image(images[0], caption=prompt[:200]),
                "latency": elapsed,
            }, step=i)

    # Log summary table to wandb
    if wb_run:
        columns = ["idx", "prompt", "student", "latency_s"]
        table = wandb.Table(columns=columns)
        for i, r in enumerate(all_results):
            table.add_data(i, r["prompt"], wandb.Image(r["image"]), f"{r['time']:.2f}")
        wb_run.log({"samples": table})
        wb_run.finish()
        print(f"\nW&B run finished: {wb_run.url}")

    # Summary
    mean_time = sum(r["time"] for r in all_results) / len(all_results)
    print(f"\n{'='*60}")
    print(f"  Generated {len(all_results)} images")
    print(f"  Mean latency: {mean_time:.2f}s")
    print(f"{'='*60}")
    print("\nDone.")


if __name__ == "__main__":
    main()
