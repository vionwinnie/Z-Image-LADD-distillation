#!/usr/bin/env python3
"""Standalone evaluation script for LADD distillation checkpoints.

Loads a student checkpoint, generates validation images using precomputed
embeddings, computes KID against teacher reference images, and logs
everything (image comparison table + scalar metrics) to a wandb run.

Supports run resumption via --wandb_run_id so that successive checkpoint
evaluations accumulate as a learning curve on the same wandb dashboard.

Usage:
    # Evaluate teacher baseline
    python scripts/eval_checkpoint.py --checkpoint baseline --val_step 0

    # Evaluate a safetensors checkpoint at training step 2000
    python scripts/eval_checkpoint.py \
        --checkpoint output/ladd/checkpoint-2000/model.safetensors \
        --val_step 2000

    # Evaluate a DCP sharded checkpoint
    python scripts/eval_checkpoint.py \
        --checkpoint output/ladd/checkpoint-4000 \
        --val_step 4000
"""

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DCP consolidation (inline to avoid import issues across environments)
# ---------------------------------------------------------------------------

def consolidate_dcp_checkpoint(checkpoint_dir, model_dir, dtype, output_path):
    """Consolidate a DCP sharded checkpoint into a single safetensors file."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")

    from training.ladd_model_utils import load_transformer
    student = load_transformer(model_dir, dtype)
    state_dict = {"model": student.state_dict()}
    dcp.load(state_dict, checkpoint_id=checkpoint_dir)
    student.load_state_dict(state_dict["model"])
    del state_dict

    dist.destroy_process_group()

    save_file(
        {k: v.contiguous().cpu() for k, v in student.state_dict().items()},
        output_path,
    )
    return student


# ---------------------------------------------------------------------------
# Student model loading
# ---------------------------------------------------------------------------

def load_student(checkpoint, model_dir, dtype):
    """Load student transformer from various checkpoint formats.

    Supported formats:
      - "baseline": loads teacher weights directly (no fine-tuning)
      - path to a .safetensors file
      - directory containing model.safetensors
      - directory containing student_transformer/model.safetensors (accelerate)
      - directory with .metadata file (DCP sharded, auto-consolidated)
    """
    from safetensors.torch import load_file
    from training.ladd_model_utils import load_transformer

    if checkpoint == "baseline":
        logger.info("Loading baseline (teacher) weights")
        student = load_transformer(model_dir, dtype)
        return student

    checkpoint_path = Path(checkpoint)

    # Direct safetensors file
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".safetensors":
        logger.info(f"Loading safetensors checkpoint: {checkpoint_path}")
        student = load_transformer(model_dir, dtype)
        sd = load_file(str(checkpoint_path))
        student.load_state_dict(sd, strict=False)
        del sd
        return student

    # Directory-based checkpoints
    if checkpoint_path.is_dir():
        # Check for model.safetensors directly
        direct_st = checkpoint_path / "model.safetensors"
        if direct_st.exists():
            logger.info(f"Loading safetensors from dir: {direct_st}")
            student = load_transformer(model_dir, dtype)
            sd = load_file(str(direct_st))
            student.load_state_dict(sd, strict=False)
            del sd
            return student

        # Check for accelerate save format
        accel_st = checkpoint_path / "student_transformer" / "model.safetensors"
        if accel_st.exists():
            logger.info(f"Loading accelerate checkpoint: {accel_st}")
            student = load_transformer(model_dir, dtype)
            sd = load_file(str(accel_st))
            student.load_state_dict(sd, strict=False)
            del sd
            return student

        # Check for DCP sharded format (.metadata file)
        metadata_file = checkpoint_path / ".metadata"
        if metadata_file.exists():
            logger.info(f"DCP sharded checkpoint detected: {checkpoint_path}")
            consolidated_path = checkpoint_path / "consolidated.safetensors"
            if consolidated_path.exists():
                logger.info(f"Using existing consolidated file: {consolidated_path}")
                student = load_transformer(model_dir, dtype)
                sd = load_file(str(consolidated_path))
                student.load_state_dict(sd, strict=False)
                del sd
                return student
            else:
                logger.info("Consolidating DCP checkpoint...")
                student = consolidate_dcp_checkpoint(
                    str(checkpoint_path), model_dir, dtype, str(consolidated_path),
                )
                return student

    raise ValueError(
        f"Cannot load checkpoint: {checkpoint}. "
        "Expected 'baseline', a .safetensors file, or a directory with "
        "model.safetensors / student_transformer/model.safetensors / .metadata"
    )


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(student, vae, scheduler, embeddings_data, args):
    """Generate validation images from precomputed embeddings."""
    from zimage.pipeline import generate_from_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = student.to(device)
    student.eval()
    vae = vae.to(device, dtype=torch.float32)
    vae.eval()

    emb_list = embeddings_data["embeddings"]
    num_images = min(args.num_kid_images, len(emb_list))

    save_dir = os.path.join(args.output_dir, f"step_{args.val_step:06d}")
    os.makedirs(save_dir, exist_ok=True)

    image_paths = []
    generator = torch.Generator(device).manual_seed(42)

    for i in range(num_images):
        emb = emb_list[i].to(device)
        images = generate_from_embeddings(
            transformer=student,
            vae=vae,
            prompt_embeds_list=[emb],
            scheduler=scheduler,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        path = os.path.join(save_dir, f"{i:05d}.png")
        images[0].save(path)
        image_paths.append(path)

        if (i + 1) % 100 == 0:
            logger.info(f"  Generated {i + 1}/{num_images} images")

    logger.info(f"Generated {len(image_paths)} images in {save_dir}")
    return image_paths, save_dir


# ---------------------------------------------------------------------------
# KID computation with corrupt-image filtering
# ---------------------------------------------------------------------------

def compute_kid_filtered(student_image_dir, teacher_image_dir, num_images):
    """Compute KID, filtering out corrupted (0-byte) teacher images.

    Copies valid student/teacher pairs into temp directories so that
    torch-fidelity sees only matching, valid images.
    """
    from torch_fidelity import calculate_metrics

    tmp_student = tempfile.mkdtemp(prefix="kid_student_")
    tmp_teacher = tempfile.mkdtemp(prefix="kid_teacher_")

    valid_count = 0
    for i in range(num_images):
        student_path = os.path.join(student_image_dir, f"{i:05d}.png")
        teacher_path = os.path.join(teacher_image_dir, f"{i:05d}.png")

        if not os.path.exists(student_path) or not os.path.exists(teacher_path):
            continue
        if os.path.getsize(teacher_path) == 0:
            logger.warning(f"Skipping corrupted teacher image: {teacher_path}")
            continue
        if os.path.getsize(student_path) == 0:
            logger.warning(f"Skipping corrupted student image: {student_path}")
            continue

        shutil.copy2(student_path, os.path.join(tmp_student, f"{valid_count:05d}.png"))
        shutil.copy2(teacher_path, os.path.join(tmp_teacher, f"{valid_count:05d}.png"))
        valid_count += 1

    if valid_count == 0:
        logger.error("No valid image pairs found for KID computation!")
        shutil.rmtree(tmp_student, ignore_errors=True)
        shutil.rmtree(tmp_teacher, ignore_errors=True)
        return {"kid_mean": float("nan"), "kid_std": float("nan"), "valid_count": 0}

    logger.info(f"Computing KID with {valid_count} valid image pairs")

    kid_subset_size = min(100, valid_count)
    metrics = calculate_metrics(
        input1=tmp_student,
        input2=tmp_teacher,
        cuda=torch.cuda.is_available(),
        kid=True,
        kid_subset_size=kid_subset_size,
        kid_subsets=100,
    )

    shutil.rmtree(tmp_student, ignore_errors=True)
    shutil.rmtree(tmp_teacher, ignore_errors=True)

    return {
        "kid_mean": metrics["kernel_inception_distance_mean"],
        "kid_std": metrics["kernel_inception_distance_std"],
        "valid_count": valid_count,
    }


# ---------------------------------------------------------------------------
# Wandb logging
# ---------------------------------------------------------------------------

def log_to_wandb(args, image_paths, prompts, kid_results):
    """Log image comparison table and KID metrics to wandb."""
    import wandb

    # Init or resume wandb run
    init_kwargs = dict(
        project=args.wandb_project,
        config=vars(args),
    )
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "must"
    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name

    run = wandb.init(**init_kwargs)
    logger.info(f"wandb run: {run.id} ({run.url})")

    # Save run ID for resumption
    run_id_path = os.path.join(args.output_dir, "wandb_run_id.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(run_id_path, "w") as f:
        f.write(run.id)

    # Image comparison table
    num_pairs = min(args.num_image_pairs, len(image_paths))
    columns = ["step", "prompt", "student", "teacher"]
    table = wandb.Table(columns=columns)

    for i in range(num_pairs):
        teacher_path = os.path.join(args.teacher_image_dir, f"{i:05d}.png")
        prompt = prompts[i] if i < len(prompts) else ""
        student_img = wandb.Image(image_paths[i])
        teacher_img = wandb.Image(teacher_path) if os.path.exists(teacher_path) else None
        table.add_data(args.val_step, prompt, student_img, teacher_img)

    run.log({"eval/image_pairs": table}, step=args.val_step)

    # KID metrics
    if kid_results:
        run.log(
            {
                "eval/kid_mean": kid_results["kid_mean"],
                "eval/kid_std": kid_results["kid_std"],
            },
            step=args.val_step,
        )

    run.finish()
    return run.id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a LADD student checkpoint against teacher images"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="'baseline' for teacher weights, or path to safetensors/checkpoint dir",
    )
    parser.add_argument(
        "--model_dir", type=str,
        default="/workspace/Z-Image-LADD-distillation/models/Z-Image",
    )
    parser.add_argument("--val_data_meta", type=str, default="data/val/metadata.json")
    parser.add_argument("--val_embeddings_dir", type=str, default="data/val/embeddings")
    parser.add_argument("--teacher_image_dir", type=str, default="data/val/teacher_images")
    parser.add_argument("--num_image_pairs", type=int, default=50,
                        help="Number of image pairs logged to wandb table")
    parser.add_argument("--num_kid_images", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--val_step", type=int, default=0,
                        help="Training step (x-axis for learning curve)")
    parser.add_argument("--output_dir", type=str, default="output/eval")
    parser.add_argument("--wandb_project", type=str, default="ladd-eval")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Resume an existing wandb run for learning curve continuity")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [eval]: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"  val_step={args.val_step}, num_kid_images={args.num_kid_images}")

    # ---- 1. Load student model ----
    dtype = torch.bfloat16
    student = load_student(args.checkpoint, args.model_dir, dtype)
    logger.info("Student model loaded")

    # ---- 2. Load VAE ----
    from training.ladd_model_utils import load_vae
    vae = load_vae(args.model_dir)
    logger.info("VAE loaded")

    # ---- 3. Load scheduler ----
    scheduler_config_path = os.path.join(args.model_dir, "scheduler", "scheduler_config.json")
    if os.path.exists(scheduler_config_path):
        with open(scheduler_config_path) as f:
            sched_cfg = json.load(f)
    else:
        sched_cfg = {}

    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        shift=sched_cfg.get("shift", 3.0),
        use_dynamic_shifting=sched_cfg.get("use_dynamic_shifting", False),
    )
    logger.info("Scheduler loaded")

    # ---- 4. Load val embeddings ----
    embeddings_path = os.path.join(args.val_embeddings_dir, "embeddings.pt")
    embeddings_data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded {len(embeddings_data['embeddings'])} val embeddings")

    # ---- 5. Load prompts ----
    with open(args.val_data_meta, "r") as f:
        val_records = json.load(f)
    prompts = [r["text"] for r in val_records]

    # ---- 6. Generate images ----
    image_paths, student_image_dir = generate_images(
        student, vae, scheduler, embeddings_data, args,
    )

    # Free GPU memory before KID
    del student, vae, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 7. Compute KID ----
    kid_results = None
    if os.path.isdir(args.teacher_image_dir):
        logger.info("Computing KID...")
        kid_results = compute_kid_filtered(
            student_image_dir, args.teacher_image_dir, len(image_paths),
        )
        logger.info(
            f"KID = {kid_results['kid_mean']:.6f} +/- {kid_results['kid_std']:.6f} "
            f"({kid_results['valid_count']} valid pairs)"
        )
    else:
        logger.warning(f"Teacher image dir not found: {args.teacher_image_dir}, skipping KID")

    # ---- 8. Log to wandb ----
    run_id = log_to_wandb(args, image_paths, prompts, kid_results)

    # ---- 9. Save results JSON ----
    results = {
        "checkpoint": args.checkpoint,
        "val_step": args.val_step,
        "num_images": len(image_paths),
        "student_image_dir": student_image_dir,
        "wandb_run_id": run_id,
    }
    if kid_results:
        results.update(kid_results)

    results_path = os.path.join(args.output_dir, f"results_step_{args.val_step:06d}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ---- 10. Summary ----
    print("\n" + "=" * 60)
    print(f"  Evaluation Summary  (step {args.val_step})")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Images     : {len(image_paths)}")
    if kid_results:
        print(f"  KID mean   : {kid_results['kid_mean']:.6f}")
        print(f"  KID std    : {kid_results['kid_std']:.6f}")
        print(f"  Valid pairs: {kid_results['valid_count']}")
    print(f"  wandb run  : {run_id}")
    print(f"  Results    : {results_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
