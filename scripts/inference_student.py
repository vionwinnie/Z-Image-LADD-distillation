"""Inference with LADD-distilled student model.

Loads student weights from either:
  - DCP sharded checkpoint (from FSDP training)
  - Safetensors file (consolidated)

Usage:
    # From DCP sharded checkpoint:
    python scripts/inference_student.py \
        --checkpoint output/ladd/checkpoint-20000 \
        --model_dir models/Z-Image \
        --prompt "A cat sitting on a windowsill"

    # From consolidated safetensors:
    python scripts/inference_student.py \
        --checkpoint output/ladd/checkpoint-20000/model.safetensors \
        --model_dir models/Z-Image \
        --prompt "A cat sitting on a windowsill"

    # Consolidate DCP checkpoint to safetensors (one-time):
    python scripts/inference_student.py \
        --checkpoint output/ladd/checkpoint-20000 \
        --model_dir models/Z-Image \
        --consolidate_only
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ladd_model_utils import load_transformer, load_vae
from zimage.pipeline import generate
from zimage.scheduler import FlowMatchEulerDiscreteScheduler


def load_student_from_dcp(checkpoint_dir: str, model_dir: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load student from DCP sharded checkpoint (requires torch.distributed init)."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

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
    from safetensors.torch import load_file
    student = load_transformer(model_dir, dtype)
    student.load_state_dict(load_file(path), strict=False)
    return student


def consolidate_checkpoint(checkpoint_dir: str, model_dir: str, dtype: torch.dtype, output_path: str = None):
    """Convert DCP sharded checkpoint to a single safetensors file."""
    from safetensors.torch import save_file

    if output_path is None:
        output_path = os.path.join(checkpoint_dir, "model.safetensors")

    print(f"Loading sharded checkpoint from {checkpoint_dir}...")
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
    parser.add_argument("--output", type=str, default="student_output.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true", help="torch.compile for faster inference")
    parser.add_argument("--consolidate_only", action="store_true",
                        help="Only consolidate DCP checkpoint to safetensors, then exit")
    parser.add_argument("--compare_teacher", action="store_true",
                        help="Also generate with teacher model for side-by-side comparison")
    parser.add_argument("--teacher_steps", type=int, default=50,
                        help="Teacher denoising steps for comparison")
    parser.add_argument("--teacher_cfg", type=float, default=5.0,
                        help="Teacher CFG scale for comparison")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Consolidate mode
    if args.consolidate_only:
        consolidate_checkpoint(args.checkpoint, args.model_dir, dtype)
        return

    # Determine checkpoint type
    is_safetensors = args.checkpoint.endswith(".safetensors")
    is_dcp = os.path.isdir(args.checkpoint) and os.path.exists(os.path.join(args.checkpoint, ".metadata"))

    # Check for consolidated file inside DCP dir
    consolidated_path = os.path.join(args.checkpoint, "model.safetensors") if is_dcp else None
    if consolidated_path and os.path.exists(consolidated_path):
        print(f"Found consolidated weights: {consolidated_path}")
        is_safetensors = True
        args.checkpoint = consolidated_path

    # Load student
    print(f"Loading student from {args.checkpoint}...")
    if is_safetensors:
        student = load_student_from_safetensors(args.checkpoint, args.model_dir, dtype)
    elif is_dcp:
        student = load_student_from_dcp(args.checkpoint, args.model_dir, dtype)
    else:
        raise ValueError(f"Cannot determine checkpoint type for {args.checkpoint}. "
                         f"Expected .safetensors file or DCP dir with .metadata")

    student.to(args.device)
    student.eval()
    if args.compile:
        student = torch.compile(student)

    # Load shared components
    print("Loading VAE, text encoder, scheduler...")
    import json
    from transformers import AutoModel, AutoTokenizer

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

    # Generate with student
    print(f"\n{'='*60}")
    print(f"Student: {args.num_inference_steps} steps, CFG={args.guidance_scale}")
    print(f"Prompt: {args.prompt[:80]}...")
    print(f"{'='*60}")

    generator = torch.Generator(args.device).manual_seed(args.seed)
    start = time.time()
    images = generate(
        transformer=student,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    elapsed = time.time() - start
    images[0].save(args.output)
    print(f"Student image saved to {args.output} ({elapsed:.2f}s)")

    # Compare with teacher
    if args.compare_teacher:
        print(f"\n{'='*60}")
        print(f"Teacher: {args.teacher_steps} steps, CFG={args.teacher_cfg}")
        print(f"{'='*60}")

        # Load teacher (same architecture, original weights)
        teacher = load_transformer(args.model_dir, dtype)
        teacher.to(args.device)
        teacher.eval()

        generator = torch.Generator(args.device).manual_seed(args.seed)
        start = time.time()
        teacher_images = generate(
            transformer=teacher,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or "",
            height=args.height,
            width=args.width,
            num_inference_steps=args.teacher_steps,
            guidance_scale=args.teacher_cfg,
            generator=generator,
        )
        elapsed = time.time() - start
        teacher_output = args.output.replace(".png", "_teacher.png")
        teacher_images[0].save(teacher_output)
        print(f"Teacher image saved to {teacher_output} ({elapsed:.2f}s)")

        # Side-by-side comparison
        try:
            from PIL import Image
            s_img = images[0]
            t_img = teacher_images[0]
            combined = Image.new("RGB", (s_img.width + t_img.width + 20, max(s_img.height, t_img.height) + 40), "white")
            combined.paste(s_img, (0, 40))
            combined.paste(t_img, (s_img.width + 20, 40))
            comparison_path = args.output.replace(".png", "_comparison.png")
            combined.save(comparison_path)
            print(f"Comparison saved to {comparison_path}")
            print(f"  Left: Student ({args.num_inference_steps} steps)")
            print(f"  Right: Teacher ({args.teacher_steps} steps)")
        except Exception as e:
            print(f"Could not create comparison image: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
