"""Evaluation utilities for LADD training.

Two tiers:
  1. Cheap per-step metrics computed inline from discriminator logits.
  2. Expensive image-quality metrics (FID, CLIP score) run as an async
     subprocess on a dedicated GPU after a checkpoint is saved.

Usage as subprocess (launched by train_ladd.py):
    python -m training.ladd_eval \
        --checkpoint output/ladd/checkpoint-1000 \
        --model_dir /path/to/pretrained \
        --val_data_meta data/val/metadata.json \
        --fid_reference_stats data/val/fid_reference_stats.npz \
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

import numpy as np
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
    from zimage.autoencoder import AutoencoderKL, load_vae
    from zimage.transformer import ZImageTransformer2DModel, load_transformer
    from zimage.scheduler import FlowMatchEulerDiscreteScheduler
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


def compute_fid(image_paths: list[str], reference_stats_path: str, device: str = "cuda") -> float:
    """Compute FID between generated images and pre-computed reference statistics."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms
    from PIL import Image

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Load pre-computed reference stats (mu, sigma from Inception features)
    ref = np.load(reference_stats_path)
    ref_mu = torch.from_numpy(ref["mu"]).to(device)
    ref_sigma = torch.from_numpy(ref["sigma"]).to(device)

    # Inject reference stats directly into the metric
    # torchmetrics FID stores: real_features_sum, real_features_cov_sum, real_features_num_samples
    # We set them from pre-computed values
    n_ref = int(ref["num_samples"])
    fid.real_features_sum = ref_mu * n_ref
    fid.real_features_cov_sum = ref_sigma * (n_ref - 1) + n_ref * torch.outer(ref_mu, ref_mu)
    fid.real_features_num_samples = torch.tensor(n_ref, device=device)

    # Process generated images
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    batch_size = 32
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths])
        fid.update(imgs.to(device), real=False)

    score = fid.compute().item()
    del fid
    torch.cuda.empty_cache()
    return score


def compute_clip_score(
    image_paths: list[str],
    prompts: list[str],
    device: str = "cuda",
) -> float:
    """Compute mean CLIP score between generated images and their prompts."""
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchvision import transforms
    from PIL import Image

    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    batch_size = 32
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths])
        clip_metric.update(imgs.to(device), batch_prompts)

    score = clip_metric.compute().item()
    del clip_metric
    torch.cuda.empty_cache()
    return score


def run_eval(
    checkpoint_path: str,
    model_dir: str,
    val_prompts: list[str],
    reference_stats_path: str,
    output_dir: str,
    step: int,
    device: str = "cuda",
    num_inference_steps: int = 4,
    image_size: int = 512,
    seed: int = 42,
) -> dict:
    """Full evaluation: generate images, compute FID and CLIP score."""
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

    results = {"step": step}

    # FID
    if reference_stats_path and os.path.exists(reference_stats_path):
        logger.info(f"[Eval step {step}] Computing FID...")
        results["fid"] = compute_fid(image_paths, reference_stats_path, device)
        logger.info(f"[Eval step {step}] FID = {results['fid']:.2f}")
    else:
        logger.warning("FID reference stats not found, skipping FID.")

    # CLIP score
    logger.info(f"[Eval step {step}] Computing CLIP score...")
    results["clip_score"] = compute_clip_score(image_paths, val_prompts, device)
    logger.info(f"[Eval step {step}] CLIP score = {results['clip_score']:.2f}")

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
    parser.add_argument("--fid_reference_stats", type=str, default=None)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--eval_num_images", type=int, default=2048)
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
        reference_stats_path=args.fid_reference_stats,
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
            wandb.init(
                id=args.wandb_run_id,
                project=args.wandb_project,
                entity=args.wandb_entity,
                resume="allow",
            )
            log_dict = {}
            if "fid" in results:
                log_dict["eval/fid"] = results["fid"]
            if "clip_score" in results:
                log_dict["eval/clip_score"] = results["clip_score"]
            wandb.log(log_dict, step=args.step)
            wandb.finish()
        except Exception as e:
            logger.error(f"Failed to log to wandb: {e}")

    logger.info(f"Eval complete for step {args.step}")


if __name__ == "__main__":
    main()
