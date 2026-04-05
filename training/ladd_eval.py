"""Evaluation utilities for LADD training.

Two tiers:
  1. Cheap per-step metrics computed inline from discriminator logits.
  2. Expensive image-quality metrics (KID) run as an async subprocess
     on a dedicated GPU after a checkpoint is saved.

KID (Kernel Inception Distance) is used instead of FID because it is
an unbiased estimator that works well with small sample sizes (≤1000).

Usage as subprocess (launched by train_ladd.py):
    python -m training.ladd_eval \
        --checkpoint output/ladd/checkpoint-1000 \
        --model_dir /path/to/pretrained \
        --val_data_meta data/val/metadata.json \
        --teacher_image_dir data/val/teacher_images \
        --step 1000 \
        --output_dir output/ladd \
        --device cuda:7 \
        --wandb_run_id <run_id> \
        --wandb_project ladd \
        --wandb_entity yeun-yeungs
"""

import argparse
import json
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 1: Cheap per-step metrics (inline, from existing tensors)
# ---------------------------------------------------------------------------

def compute_discriminator_metrics(
    real_result: dict,
    fake_result: dict,
) -> dict:
    """Compute discriminator health metrics from existing logits.

    Args:
        real_result: dict with "logits" (layer_idx -> (B,)) and "total_logit" (B,)
        fake_result: same structure

    Returns:
        dict of metric_name -> float, ready to merge into training logs.
    """
    metrics = {}

    real_total = real_result["total_logit"].detach()
    fake_total = fake_result["total_logit"].detach()

    # Overall accuracy (hinge boundary at 0)
    metrics["disc/accuracy_real"] = (real_total > 0).float().mean().item()
    metrics["disc/accuracy_fake"] = (fake_total < 0).float().mean().item()

    # Logit gap: positive = D can tell them apart, collapsing = trouble
    metrics["disc/logit_gap"] = real_total.mean().item() - fake_total.mean().item()

    # Per-layer logit means
    for layer_idx in real_result["logits"]:
        r = real_result["logits"][layer_idx].detach().mean().item()
        f = fake_result["logits"][layer_idx].detach().mean().item()
        metrics[f"disc/layer_{layer_idx}_real"] = r
        metrics[f"disc/layer_{layer_idx}_fake"] = f

    return metrics


# ---------------------------------------------------------------------------
# Tier 2: Expensive image-quality metrics (FID + CLIP score)
# ---------------------------------------------------------------------------

def generate_eval_images(
    checkpoint_path: str,
    model_dir: str,
    prompts: list[str],
    output_dir: str,
    step: int,
    num_inference_steps: int = 4,
    image_size: int = 512,
    device: str = "cuda",
    seed: int = 42,
) -> list[str]:
    """Generate images from a student checkpoint for evaluation.

    Returns list of saved image paths.
    """
    from zimage.pipeline import generate
    from zimage.scheduler import FlowMatchEulerDiscreteScheduler
    from training.ladd_model_utils import load_transformer, load_vae
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16

    # Load student
    student = load_transformer(model_dir, dtype)
    student_weights = os.path.join(checkpoint_path, "student_transformer", "pytorch_model.bin")
    if os.path.exists(student_weights):
        sd = torch.load(student_weights, map_location="cpu", weights_only=True)
        student.load_state_dict(sd, strict=False)
        del sd
    else:
        # Try safetensors
        from safetensors.torch import load_file
        st_dir = os.path.join(checkpoint_path, "student_transformer")
        st_files = [f for f in os.listdir(st_dir) if f.endswith(".safetensors")]
        if st_files:
            sd = {}
            for sf in st_files:
                sd.update(load_file(os.path.join(st_dir, sf)))
            student.load_state_dict(sd, strict=False)
            del sd
    student.to(device)
    student.eval()

    # Load VAE
    vae = load_vae(model_dir)
    vae.to(device, dtype=torch.float32)
    vae.eval()

    # Load text encoder + tokenizer
    text_encoder_dir = os.path.join(model_dir, "text_encoder")
    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, torch_dtype=dtype, trust_remote_code=True)
    text_encoder.to(device)
    text_encoder.eval()
    if os.path.exists(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True)

    # Load scheduler
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

    # Generate
    save_dir = os.path.join(output_dir, "eval_images", f"step_{step:06d}")
    os.makedirs(save_dir, exist_ok=True)

    image_paths = []
    generator = torch.Generator(device).manual_seed(seed)
    batch_size = 4

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        images = generate(
            transformer=student,
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
        )
        for j, img in enumerate(images):
            path = os.path.join(save_dir, f"{start + j:05d}.png")
            img.save(path)
            image_paths.append(path)

        if (start // batch_size) % 50 == 0:
            logger.info(f"  Generated {start + len(batch_prompts)}/{len(prompts)} images")

    # Cleanup GPU memory
    del student, vae, text_encoder, tokenizer, scheduler
    torch.cuda.empty_cache()

    return image_paths


def compute_kid(
    student_image_dir: str,
    teacher_image_dir: str,
    num_images: int,
) -> dict:
    """Compute KID between student and teacher image directories.

    Uses torch-fidelity with polynomial kernel MMD. Returns dict with
    kid_mean and kid_std.
    """
    from torch_fidelity import calculate_metrics

    kid_subset_size = min(100, num_images)
    metrics = calculate_metrics(
        input1=student_image_dir,
        input2=teacher_image_dir,
        cuda=True,
        kid=True,
        kid_subset_size=kid_subset_size,
        kid_subsets=100,
    )
    return {
        "kid_mean": metrics["kernel_inception_distance_mean"],
        "kid_std": metrics["kernel_inception_distance_std"],
    }


def run_eval(
    checkpoint_path: str,
    model_dir: str,
    val_prompts: list[str],
    teacher_image_dir: str,
    output_dir: str,
    step: int,
    device: str = "cuda",
    num_inference_steps: int = 4,
    image_size: int = 512,
    seed: int = 42,
) -> dict:
    """Full evaluation: generate student images, compute KID against teacher images."""
    import gc

    logger.info(f"[Eval step {step}] Generating {len(val_prompts)} images...")
    image_paths = generate_eval_images(
        checkpoint_path=checkpoint_path,
        model_dir=model_dir,
        prompts=val_prompts,
        output_dir=output_dir,
        step=step,
        num_inference_steps=num_inference_steps,
        image_size=image_size,
        device=device,
        seed=seed,
    )

    # Free GPU before KID computation (Inception model needs memory)
    gc.collect()
    torch.cuda.empty_cache()

    student_image_dir = os.path.dirname(image_paths[0])
    results = {"step": step, "num_images": len(image_paths), "student_image_dir": student_image_dir}

    # KID
    if os.path.isdir(teacher_image_dir):
        logger.info(f"[Eval step {step}] Computing KID...")
        kid_results = compute_kid(student_image_dir, teacher_image_dir, len(image_paths))
        results.update(kid_results)
        logger.info(f"[Eval step {step}] KID = {results['kid_mean']:.6f} ± {results['kid_std']:.6f}")
    else:
        logger.warning(f"Teacher image dir not found: {teacher_image_dir}, skipping KID.")

    # Save results JSON
    results_dir = os.path.join(output_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"step_{step:06d}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[Eval step {step}] Results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [eval]: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="LADD async evaluation subprocess")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--val_data_meta", type=str, required=True)
    parser.add_argument("--teacher_image_dir", type=str, required=True,
                        help="Directory with pre-generated teacher reference images.")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--eval_num_images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # wandb
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="ladd")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    # Load val prompts and sample fixed subset
    with open(args.val_data_meta, "r") as f:
        val_records = json.load(f)
    all_prompts = [r["text"] for r in val_records]

    if args.eval_num_images < len(all_prompts):
        import random
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(all_prompts)), args.eval_num_images)
        val_prompts = [all_prompts[i] for i in sorted(indices)]
    else:
        val_prompts = all_prompts

    logger.info(f"Eval: step={args.step}, images={len(val_prompts)}, device={args.device}")

    results = run_eval(
        checkpoint_path=args.checkpoint,
        model_dir=args.model_dir,
        val_prompts=val_prompts,
        teacher_image_dir=args.teacher_image_dir,
        output_dir=args.output_dir,
        step=args.step,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        image_size=args.image_size,
        seed=args.seed,
    )

    # Log to wandb if configured
    if args.wandb_run_id:
        try:
            import wandb
            from PIL import Image

            wandb.init(
                id=args.wandb_run_id,
                project=args.wandb_project,
                entity=args.wandb_entity,
                resume="allow",
            )
            log_dict = {}
            if "kid_mean" in results:
                log_dict["eval/kid_mean"] = results["kid_mean"]
                log_dict["eval/kid_std"] = results["kid_std"]

            # Log side-by-side student vs teacher images (first N samples)
            student_image_dir = results.get("student_image_dir")
            if student_image_dir and os.path.isdir(args.teacher_image_dir):
                num_log = min(8, results.get("num_images", 0))
                table = wandb.Table(columns=["step", "prompt", "student", "teacher"])
                for i in range(num_log):
                    student_path = os.path.join(student_image_dir, f"{i:05d}.png")
                    teacher_path = os.path.join(args.teacher_image_dir, f"{i:05d}.png")
                    prompt = val_prompts[i][:100] if i < len(val_prompts) else ""
                    if os.path.exists(student_path) and os.path.exists(teacher_path):
                        table.add_data(
                            args.step,
                            prompt,
                            wandb.Image(Image.open(student_path)),
                            wandb.Image(Image.open(teacher_path)),
                        )
                log_dict["eval/samples"] = table

            wandb.log(log_dict, step=args.step)
            wandb.finish()
        except Exception as e:
            logger.error(f"Failed to log to wandb: {e}")

    # Clean up validation checkpoint (lightweight, not needed after eval)
    if os.path.basename(args.checkpoint).startswith("val-checkpoint-"):
        import shutil
        shutil.rmtree(args.checkpoint, ignore_errors=True)
        logger.info(f"Cleaned up {args.checkpoint}")

    logger.info(f"Eval complete for step {args.step}")


if __name__ == "__main__":
    main()
