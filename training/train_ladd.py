"""LADD (Latent Adversarial Diffusion Distillation) training script for Z-Image.

Trains a student transformer to produce high-quality images in fewer denoising
steps by using a frozen teacher transformer + lightweight discriminator heads
in an adversarial distillation setup.

Usage:
    accelerate launch training/train_ladd.py \
        --pretrained_model_name_or_path=models/Z-Image \
        --train_data_meta=data/debug/metadata.json \
        --train_batch_size=1 \
        --max_train_steps=100
"""

import argparse
import contextlib
import gc
import json
import logging
import math
import os
import pickle
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import BatchSampler, RandomSampler
from tqdm.auto import tqdm

# Add src to path for local imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_project_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _project_root)

from config import (
    BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT,
    DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS,
    DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
    DEFAULT_TRANSFORMER_DIM,
    DEFAULT_TRANSFORMER_F_PATCH_SIZE,
    DEFAULT_TRANSFORMER_IN_CHANNELS,
    DEFAULT_TRANSFORMER_N_HEADS,
    DEFAULT_TRANSFORMER_N_KV_HEADS,
    DEFAULT_TRANSFORMER_N_LAYERS,
    DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
    DEFAULT_TRANSFORMER_NORM_EPS,
    DEFAULT_TRANSFORMER_PATCH_SIZE,
    DEFAULT_TRANSFORMER_QK_NORM,
    DEFAULT_TRANSFORMER_T_SCALE,
    MAX_IMAGE_SEQ_LEN,
    MAX_SHIFT,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
)
from zimage.transformer import ZImageTransformer2DModel
from zimage.autoencoder import AutoencoderKL
from zimage.scheduler import FlowMatchEulerDiscreteScheduler

from training.ladd_discriminator import LADDDiscriminator
from training.ladd_utils import (
    DiscreteSampling,
    TextDataset,
    add_noise,
    calculate_shift,
    encode_prompt,
    get_sigmas_from_timesteps,
    logit_normal_sample,
)

try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LADD training for Z-Image distillation.")

    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained Z-Image model directory.")
    parser.add_argument("--student_path", type=str, default=None,
                        help="Optional separate path for student transformer weights.")
    parser.add_argument("--output_dir", type=str, default="output/ladd",
                        help="Directory for checkpoints and logs.")

    # Data
    parser.add_argument("--train_data_meta", type=str, required=True,
                        help="Path to JSON annotation file with text prompts.")
    parser.add_argument("--text_drop_ratio", type=float, default=0.1,
                        help="Probability of dropping text for CFG training.")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Maximum token length for text encoder.")

    # Image / latent
    parser.add_argument("--image_sample_size", type=int, default=512,
                        help="Image resolution (height = width) for latent generation.")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--num_train_epochs", type=int, default=1000)

    # Optimizers
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for student transformer.")
    parser.add_argument("--learning_rate_disc", type=float, default=1e-4,
                        help="Learning rate for discriminator heads.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    # LADD-specific
    parser.add_argument("--gen_update_interval", type=int, default=5,
                        help="Update student every N discriminator steps.")
    parser.add_argument("--disc_layer_indices", nargs="+", type=int,
                        default=[5, 10, 15, 20, 25, 29],
                        help="Teacher layer indices for discriminator heads.")
    parser.add_argument("--disc_hidden_dim", type=int, default=256)
    parser.add_argument("--disc_cond_dim", type=int, default=256)
    parser.add_argument("--student_timesteps", nargs="+", type=float,
                        default=[1.0, 0.75, 0.5, 0.25],
                        help="Discrete timestep set for student denoising.")
    parser.add_argument("--warmup_schedule_steps", type=int, default=500,
                        help="Steps before switching to main timestep schedule.")
    parser.add_argument("--renoise_m", type=float, default=1.0,
                        help="Mean for logit-normal re-noising distribution.")
    parser.add_argument("--renoise_s", type=float, default=1.0,
                        help="Std for logit-normal re-noising distribution.")

    # Infrastructure
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb",
                        choices=["tensorboard", "wandb", "all"],
                        help="Logging backend. Use 'wandb' for Weights & Biases.")
    parser.add_argument("--tracker_project_name", type=str, default="ladd")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name. Auto-generated if not set.")
    parser.add_argument("--wandb_entity", type=str, default="yeun-yeungs",
                        help="W&B entity (team or username).")
    parser.add_argument("--local_rank", type=int, default=-1)

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Validation
    parser.add_argument("--validation_prompts", nargs="+", type=str,
                        default=["A beautiful sunset over the ocean",
                                 "A cat sitting on a windowsill"])
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=4,
                        help="Number of steps for validation sampling.")

    # Noise scheduler
    parser.add_argument("--train_sampling_steps", type=int, default=1000)
    parser.add_argument("--use_dynamic_shifting", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_transformer(model_dir: str, dtype: torch.dtype) -> ZImageTransformer2DModel:
    """Load a ZImageTransformer2DModel from a directory with config.json + safetensors."""
    from utils.loader import load_config, load_sharded_safetensors

    transformer_dir = Path(model_dir) / "transformer"
    config = load_config(str(transformer_dir / "config.json"))

    with torch.device("meta"):
        transformer = ZImageTransformer2DModel(
            all_patch_size=tuple(config.get("all_patch_size", DEFAULT_TRANSFORMER_PATCH_SIZE)),
            all_f_patch_size=tuple(config.get("all_f_patch_size", DEFAULT_TRANSFORMER_F_PATCH_SIZE)),
            in_channels=config.get("in_channels", DEFAULT_TRANSFORMER_IN_CHANNELS),
            dim=config.get("dim", DEFAULT_TRANSFORMER_DIM),
            n_layers=config.get("n_layers", DEFAULT_TRANSFORMER_N_LAYERS),
            n_refiner_layers=config.get("n_refiner_layers", DEFAULT_TRANSFORMER_N_REFINER_LAYERS),
            n_heads=config.get("n_heads", DEFAULT_TRANSFORMER_N_HEADS),
            n_kv_heads=config.get("n_kv_heads", DEFAULT_TRANSFORMER_N_KV_HEADS),
            norm_eps=config.get("norm_eps", DEFAULT_TRANSFORMER_NORM_EPS),
            qk_norm=config.get("qk_norm", DEFAULT_TRANSFORMER_QK_NORM),
            cap_feat_dim=config.get("cap_feat_dim", DEFAULT_TRANSFORMER_CAP_FEAT_DIM),
            rope_theta=config.get("rope_theta", ROPE_THETA),
            t_scale=config.get("t_scale", DEFAULT_TRANSFORMER_T_SCALE),
            axes_dims=config.get("axes_dims", ROPE_AXES_DIMS),
            axes_lens=config.get("axes_lens", ROPE_AXES_LENS),
        ).to(dtype)

    state_dict = load_sharded_safetensors(transformer_dir, device="cpu", dtype=dtype)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    return transformer


def load_vae(model_dir: str) -> AutoencoderKL:
    """Load VAE from model directory."""
    from utils.loader import load_config, load_sharded_safetensors

    vae_dir = Path(model_dir) / "vae"
    vae_config = load_config(str(vae_dir / "config.json"))
    vae = AutoencoderKL(
        in_channels=vae_config.get("in_channels", 3),
        out_channels=vae_config.get("out_channels", 3),
        down_block_types=tuple(vae_config.get("down_block_types", ("DownEncoderBlock2D",))),
        up_block_types=tuple(vae_config.get("up_block_types", ("UpDecoderBlock2D",))),
        block_out_channels=tuple(vae_config.get("block_out_channels", (64,))),
        layers_per_block=vae_config.get("layers_per_block", 1),
        latent_channels=vae_config.get("latent_channels", 4),
        norm_num_groups=vae_config.get("norm_num_groups", 32),
        scaling_factor=vae_config.get("scaling_factor", 0.18215),
        shift_factor=vae_config.get("shift_factor", None),
        use_quant_conv=vae_config.get("use_quant_conv", True),
        use_post_quant_conv=vae_config.get("use_post_quant_conv", True),
        mid_block_add_attention=vae_config.get("mid_block_add_attention", True),
    )
    vae_state_dict = load_sharded_safetensors(vae_dir, device="cpu")
    vae.load_state_dict(vae_state_dict, strict=False)
    del vae_state_dict
    return vae


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def log_validation(
    student, vae, text_encoder, tokenizer, scheduler,
    args, accelerator, weight_dtype, global_step,
):
    """Generate sample images and log to disk + wandb."""
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype):
            logger.info("Running validation...")

            from zimage.pipeline import generate, calculate_shift as pipeline_calc_shift

            unwrapped = accelerator.unwrap_model(student)

            if args.seed is not None:
                generator = torch.Generator(device=accelerator.device).manual_seed(
                    args.seed + accelerator.process_index
                )
            else:
                generator = None

            wandb_images = []
            for i, prompt_text in enumerate(args.validation_prompts):
                images = generate(
                    transformer=unwrapped,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    scheduler=scheduler,
                    prompt=prompt_text,
                    height=args.image_sample_size,
                    width=args.image_sample_size,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=0,
                    generator=generator,
                )
                save_dir = os.path.join(args.output_dir, "samples")
                os.makedirs(save_dir, exist_ok=True)
                for j, img in enumerate(images):
                    save_path = os.path.join(
                        save_dir,
                        f"step{global_step:06d}_rank{accelerator.process_index}_prompt{i}_img{j}.jpg",
                    )
                    img.save(save_path)
                    if wandb is not None:
                        wandb_images.append(
                            wandb.Image(img, caption=f"[step {global_step}] {prompt_text[:80]}")
                        )

            # Log images to wandb
            if wandb_images and accelerator.is_main_process:
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log({"validation/samples": wandb_images}, step=global_step)

            del unwrapped
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Validation error on rank {accelerator.process_index}: {e}")
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Student timestep sampling with schedule
# ---------------------------------------------------------------------------

def sample_student_timestep(
    batch_size: int,
    global_step: int,
    student_timesteps: list,
    warmup_steps: int = 500,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample student denoising timestep from {1, 0.75, 0.5, 0.25} with schedule.

    During warmup (first `warmup_steps` steps): p=[0, 0, 0.5, 0.5]
    After warmup: p=[0.7, 0.1, 0.1, 0.1]

    Returns:
        t: (batch_size,) float tensor of selected timesteps.
    """
    n = len(student_timesteps)
    if global_step < warmup_steps:
        # Warmup: only use lower timesteps
        probs = [0.0] * max(0, n - 2) + [0.5, 0.5]
        # Pad if needed
        while len(probs) < n:
            probs.insert(0, 0.0)
    else:
        # Main schedule: heavily favor t=1 (full denoising)
        probs = [0.7] + [0.3 / max(1, n - 1)] * (n - 1)

    probs = torch.tensor(probs[:n], device=device)
    probs = probs / probs.sum()

    indices = torch.multinomial(probs.expand(batch_size, -1), num_samples=1, generator=generator).squeeze(-1)
    timestep_options = torch.tensor(student_timesteps, device=device)
    return timestep_options[indices]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        torch_rng = None

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    logger.info("Loading models...")

    # Student transformer (trainable)
    student = load_transformer(args.pretrained_model_name_or_path, weight_dtype)
    if args.student_path is not None:
        from safetensors.torch import load_file
        sd = load_file(args.student_path)
        student.load_state_dict(sd, strict=False)
        del sd
        logger.info(f"Loaded student weights from {args.student_path}")

    # Teacher transformer (frozen)
    teacher = load_transformer(args.pretrained_model_name_or_path, weight_dtype)
    teacher.requires_grad_(False)
    teacher.eval()

    # Discriminator heads (trainable, randomly initialized)
    discriminator = LADDDiscriminator(
        feature_dim=DEFAULT_TRANSFORMER_DIM,
        hidden_dim=args.disc_hidden_dim,
        cond_dim=args.disc_cond_dim,
        layer_indices=tuple(args.disc_layer_indices),
    )

    # VAE (frozen, for validation only)
    vae = load_vae(args.pretrained_model_name_or_path)
    vae.requires_grad_(False)
    vae.eval()

    # Text encoder (frozen)
    from transformers import AutoModel, AutoTokenizer as HFAutoTokenizer
    text_encoder_dir = os.path.join(args.pretrained_model_name_or_path, "text_encoder")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, dtype=weight_dtype, trust_remote_code=True)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_dir = os.path.join(args.pretrained_model_name_or_path, "tokenizer")
    if os.path.exists(tokenizer_dir):
        tokenizer = HFAutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = HFAutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True)

    # Noise scheduler
    scheduler_dir = os.path.join(args.pretrained_model_name_or_path, "scheduler")
    scheduler_config_path = os.path.join(scheduler_dir, "scheduler_config.json")
    if os.path.exists(scheduler_config_path):
        with open(scheduler_config_path) as f:
            sched_cfg = json.load(f)
    else:
        sched_cfg = {}
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS),
        shift=sched_cfg.get("shift", 3.0),
        use_dynamic_shifting=sched_cfg.get("use_dynamic_shifting", args.use_dynamic_shifting),
    )

    # -----------------------------------------------------------------------
    # Configure training
    # -----------------------------------------------------------------------
    student.train()
    student.requires_grad_(True)
    discriminator.train()

    if args.gradient_checkpointing:
        # Enable gradient checkpointing on student if the method exists
        if hasattr(student, "enable_gradient_checkpointing"):
            student.enable_gradient_checkpointing()
        else:
            logger.info("Student model does not support enable_gradient_checkpointing, using torch.utils.checkpoint manually.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizers
    student_optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.learning_rate_disc,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and dataloader
    train_dataset = TextDataset(args.train_data_meta, text_drop_ratio=args.text_drop_ratio)

    def collate_fn(examples):
        return {"text": [ex["text"] for ex in examples]}

    batch_sampler_gen = torch.Generator().manual_seed(args.seed if args.seed else 0)
    batch_sampler = BatchSampler(
        RandomSampler(train_dataset, generator=batch_sampler_gen),
        batch_size=args.train_batch_size,
        drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    # LR schedulers
    from diffusers.optimization import get_scheduler
    student_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=student_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    disc_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=disc_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare with accelerator
    student, student_optimizer, train_dataloader, student_lr_scheduler = accelerator.prepare(
        student, student_optimizer, train_dataloader, student_lr_scheduler,
    )
    discriminator, disc_optimizer, disc_lr_scheduler = accelerator.prepare(
        discriminator, disc_optimizer, disc_lr_scheduler,
    )

    # Move frozen models to device
    teacher.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device, dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = {k: v for k, v in vars(args).items() if not isinstance(v, list)}
        init_kwargs = {}
        if args.report_to in ("wandb", "all") and wandb is not None:
            init_kwargs["wandb"] = {
                "name": args.wandb_run_name,
                "entity": args.wandb_entity,
            }
        accelerator.init_trackers(args.tracker_project_name, tracker_config, init_kwargs=init_kwargs)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running LADD training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Max training steps = {args.max_train_steps}")
    logger.info(f"  Student timesteps = {args.student_timesteps}")
    logger.info(f"  Discriminator layers = {args.disc_layer_indices}")
    logger.info(f"  Gen update interval = {args.gen_update_interval}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        else:
            path = os.path.basename(args.resume_from_checkpoint)

        if path is not None:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print("No checkpoint found, starting fresh.")

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc="LADD Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Precompute latent shape info
    vae_scale_factor = 8  # standard for most VAEs
    if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_scale = vae_scale_factor * 2  # patch_size=2 in Z-Image
    height_latent = 2 * (args.image_sample_size // vae_scale)
    width_latent = 2 * (args.image_sample_size // vae_scale)
    in_channels = DEFAULT_TRANSFORMER_IN_CHANNELS

    # Scheduler setup for noise levels
    image_seq_len = (height_latent // 2) * (width_latent // 2)
    H_tokens = height_latent // 2
    W_tokens = width_latent // 2
    mu = calculate_shift(
        image_seq_len,
        noise_scheduler.config.get("base_image_seq_len", BASE_IMAGE_SEQ_LEN),
        noise_scheduler.config.get("max_image_seq_len", MAX_IMAGE_SEQ_LEN),
        noise_scheduler.config.get("base_shift", BASE_SHIFT),
        noise_scheduler.config.get("max_shift", MAX_SHIFT),
    )
    noise_scheduler.sigma_min = 0.0
    noise_scheduler.set_timesteps(args.train_sampling_steps, device=accelerator.device, mu=mu)

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=True)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        batch_sampler.sampler.generator = torch.Generator().manual_seed(
            (args.seed if args.seed else 0) + epoch
        )

        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            with torch.cuda.amp.autocast(dtype=weight_dtype):
                bsz = len(batch["text"])

                # 1. Encode prompts
                with torch.no_grad():
                    prompt_embeds = encode_prompt(
                        batch["text"],
                        device=accelerator.device,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        max_sequence_length=args.max_sequence_length,
                    )

                # 2. Sample student timestep
                student_t = sample_student_timestep(
                    bsz,
                    global_step,
                    args.student_timesteps,
                    warmup_steps=args.warmup_schedule_steps,
                    device=accelerator.device,
                    generator=torch_rng,
                )  # (B,) values in {1.0, 0.75, 0.5, 0.25}

                # 3. Generate noise and compute noisy input
                latent_shape = (bsz, in_channels, height_latent, width_latent)
                noise = torch.randn(
                    latent_shape, device=accelerator.device,
                    generator=torch_rng, dtype=weight_dtype,
                )
                # For flow matching: x_t at student_t
                # When student_t=1.0, x_t = noise (pure noise)
                # x0 is unknown (we only have text), so student starts from noise
                # The student predicts x0 from noisy input

                # Create the noisy input: x_t = (1 - sigma) * x0 + sigma * noise
                # Since we don't have x0, we start from pure noise and let student denoise
                # student_t serves as the timestep for the transformer
                # The student input IS the noise (the student generates from scratch)
                student_input_list = list(noise.unsqueeze(2).unbind(dim=0))  # list of (C, 1, H, W)

                # 4. Student forward -> denoised prediction
                is_gen_step = (global_step % args.gen_update_interval == 0)

                if is_gen_step:
                    # Student needs gradients
                    student_out, _ = accelerator.unwrap_model(student)(
                        student_input_list,
                        student_t,
                        prompt_embeds,
                        return_hidden_states=False,
                    )
                    # student_out is list of (C, 1, H, W) tensors
                    student_pred = torch.stack(student_out, dim=0).squeeze(2)  # (B, C, H, W)
                else:
                    with torch.no_grad():
                        student_out, _ = accelerator.unwrap_model(student)(
                            student_input_list,
                            student_t,
                            prompt_embeds,
                            return_hidden_states=False,
                        )
                        student_pred = torch.stack(student_out, dim=0).squeeze(2).detach()

                # 5. Re-noise student output for teacher discrimination
                t_hat = logit_normal_sample(
                    bsz, m=args.renoise_m, s=args.renoise_s,
                    device=accelerator.device, generator=torch_rng,
                )  # (B,) in [0.001, 0.999]

                renoise = torch.randn_like(student_pred)
                student_renoised = add_noise(student_pred.detach().float(), renoise.float(), t_hat.float())
                student_renoised = student_renoised.to(weight_dtype)

                # Also create "real" noisy input at the same t_hat from noise (teacher sees noise as "real data")
                # In LADD, "real" = teacher sees fresh noise at t_hat (consistent distribution)
                real_noise = torch.randn_like(student_pred)
                real_noise_2 = torch.randn_like(student_pred)
                real_noisy = add_noise(real_noise.float(), real_noise_2.float(), t_hat.float())
                real_noisy = real_noisy.to(weight_dtype)

                # 6. Teacher forward on student-renoised (fake) and real noisy input
                with torch.no_grad():
                    # Fake path: teacher sees re-noised student output
                    fake_input_list = list(student_renoised.unsqueeze(2).unbind(dim=0))
                    _, fake_extras = teacher(
                        fake_input_list,
                        t_hat,
                        prompt_embeds,
                        return_hidden_states=True,
                    )
                    fake_hidden_states = fake_extras["hidden_states"]
                    fake_x_seqlens = fake_extras["x_item_seqlens"]
                    fake_cap_seqlens = fake_extras["cap_item_seqlens"]

                    # Real path: teacher sees real noisy input
                    real_input_list = list(real_noisy.unsqueeze(2).unbind(dim=0))
                    _, real_extras = teacher(
                        real_input_list,
                        t_hat,
                        prompt_embeds,
                        return_hidden_states=True,
                    )
                    real_hidden_states = real_extras["hidden_states"]
                    real_x_seqlens = real_extras["x_item_seqlens"]
                    real_cap_seqlens = real_extras["cap_item_seqlens"]

                # 7. Discriminator: compute logits
                spatial_sizes = [(H_tokens, W_tokens)] * bsz

                fake_result = discriminator(
                    fake_hidden_states, fake_x_seqlens, fake_cap_seqlens,
                    spatial_sizes, t_hat,
                )
                real_result = discriminator(
                    real_hidden_states, real_x_seqlens, real_cap_seqlens,
                    spatial_sizes, t_hat,
                )

                # 8. Compute hinge losses
                d_loss, g_loss = LADDDiscriminator.compute_loss(
                    real_result["total_logit"],
                    fake_result["total_logit"],
                )

                # ---- Discriminator update (every step) ----
                accelerator.backward(d_loss, retain_graph=is_gen_step)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                disc_optimizer.step()
                disc_lr_scheduler.step()
                disc_optimizer.zero_grad()

                # ---- Student (generator) update (every gen_update_interval steps) ----
                if is_gen_step:
                    # Recompute fake logits with student grad path.
                    # The teacher forward runs WITHOUT torch.no_grad() so that
                    # gradients flow through the teacher's operations back to
                    # the student (the teacher's weights are frozen via
                    # requires_grad_(False), but the computation graph is kept).
                    student_renoised_grad = add_noise(
                        student_pred.float(), renoise.float(), t_hat.float()
                    ).to(weight_dtype)
                    fake_input_grad = list(student_renoised_grad.unsqueeze(2).unbind(dim=0))

                    # Teacher forward WITH gradient graph (frozen weights, live graph)
                    _, fake_extras_grad = teacher(
                        fake_input_grad,
                        t_hat,
                        prompt_embeds,
                        return_hidden_states=True,
                    )

                    fake_result_grad = discriminator(
                        fake_extras_grad["hidden_states"],
                        fake_extras_grad["x_item_seqlens"],
                        fake_extras_grad["cap_item_seqlens"],
                        spatial_sizes,
                        t_hat,
                    )
                    g_loss_update = -torch.mean(fake_result_grad["total_logit"])

                    accelerator.backward(g_loss_update)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    student_optimizer.step()
                    student_lr_scheduler.step()
                    student_optimizer.zero_grad()

            # Logging
            if accelerator.is_main_process:
                logs = {
                    "d_loss": d_loss.detach().item(),
                    "g_loss": g_loss.detach().item(),
                    "student_t_mean": student_t.mean().item(),
                    "t_hat_mean": t_hat.mean().item(),
                    "lr_student": student_lr_scheduler.get_last_lr()[0],
                    "lr_disc": disc_lr_scheduler.get_last_lr()[0],
                }
                if is_gen_step:
                    logs["g_loss_update"] = g_loss_update.detach().item()

                # Gradient norms
                disc_grad_norm = torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), float("inf")
                ).item() if any(p.grad is not None for p in discriminator.parameters()) else 0.0
                logs["grad_norm/discriminator"] = disc_grad_norm

                if is_gen_step:
                    student_grad_norm = torch.nn.utils.clip_grad_norm_(
                        student.parameters(), float("inf")
                    ).item() if any(p.grad is not None for p in accelerator.unwrap_model(student).parameters()) else 0.0
                    logs["grad_norm/student"] = student_grad_norm

                # GPU memory
                if torch.cuda.is_available():
                    logs["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                    logs["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(
                    d_loss=f"{logs['d_loss']:.4f}",
                    g_loss=f"{logs['g_loss']:.4f}",
                    lr_s=f"{logs['lr_student']:.2e}",
                )

            progress_bar.update(1)
            global_step += 1

            # Checkpointing
            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

                    # Also save student weights separately for easy loading
                    student_save_dir = os.path.join(save_path, "student_transformer")
                    os.makedirs(student_save_dir, exist_ok=True)
                    unwrapped_student = accelerator.unwrap_model(student)
                    torch.save(unwrapped_student.state_dict(), os.path.join(student_save_dir, "pytorch_model.bin"))

                    # Limit total checkpoints
                    if args.checkpoints_total_limit is not None:
                        dirs = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        while len(dirs) > args.checkpoints_total_limit:
                            shutil.rmtree(os.path.join(args.output_dir, dirs.pop(0)))

                    logger.info(f"Saved checkpoint at step {global_step}")

            # Validation
            if global_step % args.validation_steps == 0 and args.validation_prompts:
                log_validation(
                    student, vae, text_encoder, tokenizer, noise_scheduler,
                    args, accelerator, weight_dtype, global_step,
                )

        if global_step >= args.max_train_steps:
            break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        unwrapped_student = accelerator.unwrap_model(student)
        torch.save(
            unwrapped_student.state_dict(),
            os.path.join(save_path, "student_transformer", "pytorch_model.bin"),
        )
        logger.info(f"Training complete. Final checkpoint at step {global_step}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
