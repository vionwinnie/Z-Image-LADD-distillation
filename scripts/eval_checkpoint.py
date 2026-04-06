"""Evaluate a student checkpoint: generate images, compute KID, log to wandb.

Designed to be called repeatedly with different checkpoints to build a learning
curve. Each call appends metrics at a new val_step and logs image-pair tables.

Usage:
    # Baseline (teacher weights = untrained student):
    python scripts/eval_checkpoint.py \
        --checkpoint baseline \
        --val_step 0

    # After training step 2000:
    python scripts/eval_checkpoint.py \
        --checkpoint output/ladd/checkpoint-2000/model.safetensors \
        --val_step 2000

    # Resume an existing wandb run (for learning curve continuity):
    python scripts/eval_checkpoint.py \
        --checkpoint output/ladd/checkpoint-4000/model.safetensors \
        --val_step 4000 \
        --wandb_run_id abc123xyz
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))
sys.path.insert(0, _project_root)

from training.ladd_model_utils import load_transformer, load_vae
from zimage.pipeline import generate
from zimage.scheduler import FlowMatchEulerDiscreteScheduler


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate student checkpoint and log to wandb")
    # Model / checkpoint
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .safetensors, DCP dir, or 'baseline' for teacher weights")
    p.add_argument("--model_dir", type=str, default="models/Z-Image",
                   help="Path to pretrained Z-Image model")
    p.add_argument("--val_step", type=int, required=True,
                   help="Training step this checkpoint corresponds to (x-axis of learning curve)")

    # Data
    p.add_argument("--val_metadata", type=str, default="data/val/metadata.json")
    p.add_argument("--teacher_image_dir", type=str, default="data/val/teacher_images")
    p.add_argument("--num_image_pairs", type=int, default=50,
                   help="Number of image pairs to log to wandb table")
    p.add_argument("--num_kid_images", type=int, default=1000,
                   help="Number of images to generate for KID computation")

    # Generation
    p.add_argument("--num_inference_steps", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # wandb
    p.add_argument("--wandb_project", type=str, default="ladd-eval")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="Resume an existing wandb run (for learning curve continuity)")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # Output
    p.add_argument("--output_dir", type=str, default="output/eval")
    return p.parse_args()


def consolidate_dcp_checkpoint(checkpoint_dir, model_dir, dtype, output_path):
    """Convert DCP sharded checkpoint (from FSDP training) to a single safetensors file."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file

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

    print(f"Saving consolidated weights to {output_path}...")
    save_file(
        {k: v.contiguous().cpu() for k, v in student.state_dict().items()},
        output_path,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Consolidated: {size_mb:.0f} MB -> {output_path}")
    return student


def load_student(args, dtype):
    """Load student transformer from checkpoint or use teacher weights as baseline.

    Supports:
      - "baseline" — teacher weights (untrained student)
      - path/to/model.safetensors — direct safetensors file
      - checkpoint dir with model.safetensors — consolidated DCP
      - checkpoint dir with .metadata — DCP sharded (auto-consolidates)
      - checkpoint dir with student_transformer/model.safetensors — non-FSDP accelerate save
    """
    if args.checkpoint == "baseline":
        print("Loading baseline (teacher weights as student)...")
        return load_transformer(args.model_dir, dtype)

    checkpoint = args.checkpoint

    if os.path.isdir(checkpoint):
        # Priority 1: consolidated safetensors at top level
        consolidated = os.path.join(checkpoint, "model.safetensors")
        if os.path.exists(consolidated):
            checkpoint = consolidated

        # Priority 2: non-FSDP accelerate save (student_transformer/model.safetensors)
        elif os.path.exists(os.path.join(checkpoint, "student_transformer", "model.safetensors")):
            checkpoint = os.path.join(checkpoint, "student_transformer", "model.safetensors")

        # Priority 3: DCP sharded checkpoint — consolidate on the fly
        elif os.path.exists(os.path.join(checkpoint, ".metadata")):
            print("DCP sharded checkpoint detected, consolidating...")
            return consolidate_dcp_checkpoint(checkpoint, args.model_dir, dtype, consolidated)

        else:
            raise ValueError(
                f"Cannot determine checkpoint type for: {checkpoint}\n"
                f"Expected one of:\n"
                f"  - model.safetensors (consolidated)\n"
                f"  - student_transformer/model.safetensors (accelerate)\n"
                f"  - .metadata (DCP sharded)"
            )

    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file
        print(f"Loading student from {checkpoint}...")
        student = load_transformer(args.model_dir, dtype)
        student.load_state_dict(load_file(checkpoint), strict=False)
        return student

    raise ValueError(f"Unsupported checkpoint format: {checkpoint}")


def load_pipeline_components(args, dtype):
    """Load VAE, text encoder, tokenizer, scheduler."""
    from transformers import AutoModel, AutoTokenizer

    print("Loading VAE...")
    vae = load_vae(args.model_dir)
    vae.to(args.device, dtype=torch.float32)
    vae.eval()

    print("Loading text encoder...")
    text_encoder_dir = os.path.join(args.model_dir, "text_encoder")
    tokenizer_dir = os.path.join(args.model_dir, "tokenizer")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, torch_dtype=dtype, trust_remote_code=True)
    text_encoder.to(args.device)
    text_encoder.eval()

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

    return vae, text_encoder, tokenizer, scheduler


def generate_images(student, vae, text_encoder, tokenizer, scheduler, prompts, args):
    """Generate student images for a list of prompts. Returns list of PIL images."""
    images = []
    total = len(prompts)
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(args.device).manual_seed(args.seed + i)
        t0 = time.time()
        imgs = generate(
            transformer=student,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=prompt,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        )
        elapsed = time.time() - t0
        images.append(imgs[0])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {elapsed:.1f}s — {prompt[:60]}...")
    return images


def compute_kid_from_dirs(student_dir, teacher_dir, num_images):
    """Compute KID between two image directories.

    Filters out corrupted/empty images by copying valid ones to temp dirs.
    """
    import shutil
    import tempfile

    # Filter valid images into temp dirs so torch-fidelity doesn't choke
    tmp_student = tempfile.mkdtemp(prefix="kid_student_")
    tmp_teacher = tempfile.mkdtemp(prefix="kid_teacher_")
    valid_count = 0

    for fname in sorted(os.listdir(student_dir)):
        if not fname.endswith(".png"):
            continue
        s_path = os.path.join(student_dir, fname)
        t_path = os.path.join(teacher_dir, fname)
        if not os.path.exists(t_path) or os.path.getsize(t_path) == 0:
            continue
        if os.path.getsize(s_path) == 0:
            continue
        # Verify both are readable
        try:
            Image.open(s_path).verify()
            Image.open(t_path).verify()
        except Exception:
            continue
        shutil.copy2(s_path, os.path.join(tmp_student, fname))
        shutil.copy2(t_path, os.path.join(tmp_teacher, fname))
        valid_count += 1

    print(f"  KID: {valid_count} valid image pairs (filtered from {num_images})")

    from torch_fidelity import calculate_metrics
    kid_subset_size = min(100, valid_count)
    metrics = calculate_metrics(
        input1=tmp_student,
        input2=tmp_teacher,
        cuda=True,
        kid=True,
        kid_subset_size=kid_subset_size,
        kid_subsets=100,
    )

    shutil.rmtree(tmp_student, ignore_errors=True)
    shutil.rmtree(tmp_teacher, ignore_errors=True)

    return {
        "kid_mean": metrics["kernel_inception_distance_mean"],
        "kid_std": metrics["kernel_inception_distance_std"],
    }


def main():
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Load val prompts
    with open(args.val_metadata) as f:
        val_records = json.load(f)
    all_prompts = [r["text"] for r in val_records]
    print(f"Loaded {len(all_prompts)} validation prompts")

    # Deterministic subset selection
    import random
    rng = random.Random(args.seed)
    indices = list(range(len(all_prompts)))
    rng.shuffle(indices)

    # Select prompts for KID (larger set) and table (smaller set)
    num_kid = min(args.num_kid_images, len(all_prompts))
    num_table = min(args.num_image_pairs, num_kid)
    kid_indices = sorted(indices[:num_kid])
    table_indices = sorted(indices[:num_table])

    kid_prompts = [all_prompts[i] for i in kid_indices]
    table_prompts = [all_prompts[i] for i in table_indices]

    # Map table indices to kid image indices for reuse
    kid_idx_set = {idx: pos for pos, idx in enumerate(kid_indices)}
    table_to_kid = [kid_idx_set[i] for i in table_indices]

    print(f"Will generate {num_kid} images for KID, log {num_table} pairs to wandb")

    # Load models
    student = load_student(args, dtype)
    student.to(args.device)
    student.eval()

    vae, text_encoder, tokenizer, scheduler = load_pipeline_components(args, dtype)

    # Generate student images
    print(f"\nGenerating {num_kid} student images (step={args.val_step})...")
    t_start = time.time()
    student_images = generate_images(student, vae, text_encoder, tokenizer, scheduler, kid_prompts, args)
    gen_time = time.time() - t_start
    mean_latency = gen_time / len(student_images)
    print(f"Generation done: {gen_time:.1f}s total, {mean_latency:.2f}s/image")

    # Free student + text encoder from GPU
    del student, text_encoder
    torch.cuda.empty_cache()

    # Save student images to disk for KID computation
    step_dir = os.path.join(args.output_dir, f"step_{args.val_step:06d}")
    os.makedirs(step_dir, exist_ok=True)
    for i, img in enumerate(student_images):
        img.save(os.path.join(step_dir, f"{kid_indices[i]:05d}.png"))

    # Compute KID
    print(f"\nComputing KID ({num_kid} student vs {len(os.listdir(args.teacher_image_dir))} teacher images)...")
    kid_results = compute_kid_from_dirs(step_dir, args.teacher_image_dir, num_kid)
    print(f"KID = {kid_results['kid_mean']:.6f} +/- {kid_results['kid_std']:.6f}")

    # Save results JSON
    results = {
        "val_step": args.val_step,
        "checkpoint": args.checkpoint,
        "num_kid_images": num_kid,
        "num_table_images": num_table,
        "kid_mean": kid_results["kid_mean"],
        "kid_std": kid_results["kid_std"],
        "mean_latency_s": mean_latency,
        "num_inference_steps": args.num_inference_steps,
        "image_size": args.image_size,
        "seed": args.seed,
    }
    results_path = os.path.join(step_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # --- wandb logging ---
    import wandb

    resume = "must" if args.wandb_run_id else None
    run_id = args.wandb_run_id or None
    run_name = args.wandb_run_name or "ladd-eval-curve"

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=run_id,
        name=run_name,
        resume=resume,
        config={
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "image_size": args.image_size,
            "seed": args.seed,
            "model_dir": args.model_dir,
        } if not args.wandb_run_id else None,
    )

    # Log scalar metrics at this val_step
    wandb.log({
        "kid_mean": kid_results["kid_mean"],
        "kid_std": kid_results["kid_std"],
        "mean_latency_s": mean_latency,
        "num_images": num_kid,
    }, step=args.val_step)

    # Log image-pair table
    columns = ["val_step", "idx", "prompt", "student", "teacher"]
    table = wandb.Table(columns=columns)
    for j, table_kid_idx in enumerate(table_to_kid):
        orig_idx = table_indices[j]
        student_img = student_images[table_kid_idx]
        teacher_path = os.path.join(args.teacher_image_dir, f"{orig_idx:05d}.png")
        if os.path.exists(teacher_path):
            teacher_img = Image.open(teacher_path)
        else:
            teacher_img = Image.new("RGB", (args.image_size, args.image_size), (128, 128, 128))

        table.add_data(
            args.val_step,
            orig_idx,
            table_prompts[j][:200],
            wandb.Image(student_img),
            wandb.Image(teacher_img),
        )

    wandb.log({f"image_pairs/step_{args.val_step}": table}, step=args.val_step)

    # Print run ID for reuse
    print(f"\nwandb run ID: {run.id}")
    print(f"wandb URL: {run.url}")

    # Save run ID to file for automation script
    run_id_file = os.path.join(args.output_dir, "wandb_run_id.txt")
    with open(run_id_file, "w") as f:
        f.write(run.id)

    run.finish()
    print(f"\nDone. val_step={args.val_step}, KID={kid_results['kid_mean']:.6f}")


if __name__ == "__main__":
    main()
