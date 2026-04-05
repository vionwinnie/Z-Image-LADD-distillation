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
from training.ladd_eval import compute_discriminator_metrics
from training.ladd_model_utils import load_transformer, load_vae
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
    parser.add_argument("--embeddings_dir", type=str, default=None,
                        help="Path to precomputed embeddings (from precompute_embeddings.py). "
                             "Skips text encoder loading to save ~3GB GPU memory.")
    parser.add_argument("--teacher_latents_dir", type=str, default=None,
                        help="Path to precomputed teacher latents (.pt files). "
                             "Generated offline with CFG=5 via data/precompute_teacher_latents.py. "
                             "Required for correct LADD training.")
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
    parser.add_argument("--cpu_offload_optimizer", action="store_true",
                        help="Keep student optimizer states on CPU to reduce GPU memory. "
                             "Slower but fits on a single GPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb",
                        choices=["tensorboard", "wandb", "all", "none"],
                        help="Logging backend. Use 'none' to disable.")
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
    parser.add_argument("--skip_save", action="store_true",
                        help="Skip checkpoint saving (for quick experiment runs).")

    # Validation (async subprocess: generates images, computes KID, logs to wandb)
    parser.add_argument("--validation_steps", type=int, default=1000,
                        help="Run validation every N steps. Set very high to disable.")
    parser.add_argument("--num_inference_steps", type=int, default=4,
                        help="Number of denoising steps for validation image generation.")
    parser.add_argument("--eval_num_images", type=int, default=1000,
                        help="Number of images to generate for KID eval.")
    parser.add_argument("--val_data_meta", type=str, default="data/val/metadata.json",
                        help="Path to validation set JSON.")
    parser.add_argument("--teacher_image_dir", type=str, default="data/val/teacher_images",
                        help="Directory with pre-generated teacher reference images for KID.")
    parser.add_argument("--eval_device", type=str, default=None,
                        help="GPU for async eval subprocess (e.g. 'cuda:7'). Auto-detected if None.")

    # Early stopping (for research loop — abort bad configs fast)
    parser.add_argument("--early_stop", action="store_true",
                        help="Enable early stopping on discriminator health check failure.")
    parser.add_argument("--early_stop_check_interval", type=int, default=50,
                        help="Check disc health every N steps.")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Abort after N consecutive failed health checks.")

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
# Validation (async subprocess: image logging + KID)
# ---------------------------------------------------------------------------

_val_process = None  # Track running validation subprocess


def _launch_validation(student_model, teacher_model, discriminator_model, vae_model,
                       args, global_step, accelerator, precomputed_state_dict=None):
    """Save lightweight student weights and spawn validation subprocess.

    The subprocess generates student + teacher images on val prompts,
    logs side-by-side comparisons to wandb, and computes KID.

    Multi-GPU: runs on a separate GPU (non-blocking).
    Single-GPU: offloads training models to CPU, runs eval as a blocking
    subprocess, then reloads models back to GPU.

    Args:
        precomputed_state_dict: If provided, reuse this state dict instead of
            calling get_state_dict again (saves ~10min FSDP gather).
    """
    global _val_process
    import subprocess
    from safetensors.torch import save_file

    n_gpus = torch.cuda.device_count()
    single_gpu = (n_gpus <= 1)

    # Reuse precomputed state dict if available, otherwise gather
    if precomputed_state_dict is not None:
        state_dict = precomputed_state_dict
    else:
        # get_state_dict is a collective op under FSDP — ALL ranks must call it.
        state_dict = accelerator.get_state_dict(student_model)

    # Only rank 0 does file I/O and subprocess launch
    if not accelerator.is_main_process:
        return

    # Skip if previous validation is still running (multi-GPU only)
    if not single_gpu and _val_process is not None and _val_process.poll() is None:
        logger.info(f"Skipping validation at step {global_step}: previous still running (pid={_val_process.pid})")
        del state_dict
        return

    # Save lightweight student weights for the subprocess
    val_ckpt_dir = os.path.join(args.output_dir, f"val-checkpoint-{global_step}")
    student_dir = os.path.join(val_ckpt_dir, "student_transformer")
    os.makedirs(student_dir, exist_ok=True)
    if state_dict is not None:
        save_file(
            {k: v.contiguous().cpu() for k, v in state_dict.items()},
            os.path.join(student_dir, "model.safetensors"),
        )
    del state_dict

    # Determine eval device
    eval_device = args.eval_device
    if eval_device is None:
        eval_device = f"cuda:{n_gpus - 1}" if n_gpus > 1 else "cuda:0"

    # Get wandb run ID if available
    wandb_run_id = None
    wandb_entity = None
    wandb_project = None
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_run_id = tracker.run.id
            wandb_entity = args.wandb_entity
            wandb_project = args.tracker_project_name
            break

    cmd = [
        sys.executable, "-m", "training.ladd_eval",
        "--checkpoint", val_ckpt_dir,
        "--model_dir", args.pretrained_model_name_or_path,
        "--val_data_meta", args.val_data_meta,
        "--teacher_image_dir", args.teacher_image_dir,
        "--step", str(global_step),
        "--output_dir", args.output_dir,
        "--device", eval_device,
        "--num_inference_steps", str(args.num_inference_steps),
        "--image_size", str(args.image_sample_size),
        "--eval_num_images", str(args.eval_num_images),
    ]
    if wandb_run_id:
        cmd.extend(["--wandb_run_id", wandb_run_id])
    if wandb_project:
        cmd.extend(["--wandb_project", wandb_project])
    if wandb_entity:
        cmd.extend(["--wandb_entity", wandb_entity])

    # Ensure subprocess can find zimage and training modules
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    src_dir = os.path.join(project_root, "src")
    env["PYTHONPATH"] = f"{src_dir}:{project_root}:{env.get('PYTHONPATH', '')}"

    if single_gpu:
        # Single GPU: offload ALL training models to CPU, run eval blocking, reload
        logger.info(f"Single-GPU validation at step {global_step}: offloading training models to CPU...")
        gpu_models = [
            ("student", student_model),
            ("teacher", teacher_model),
            ("discriminator", discriminator_model),
            ("vae", vae_model),
        ]
        for name, model in gpu_models:
            if model is not None:
                model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Launching blocking validation subprocess on {eval_device}")
        proc = subprocess.run(cmd, text=True, env=env)
        if proc.returncode != 0:
            logger.error(f"Validation subprocess failed (exit code {proc.returncode})")
        else:
            logger.info(f"Validation complete for step {global_step}")

        # Log eval results + images to wandb from the parent process
        # (subprocess can't resume the parent's wandb run while it's open)
        eval_json = os.path.join(args.output_dir, "eval_results", f"step_{global_step:06d}.json")
        if os.path.exists(eval_json):
            import json as _json
            with open(eval_json) as _f:
                eval_data = _json.load(_f)
            log_dict = {}
            if "kid_mean" in eval_data:
                log_dict["eval/kid_mean"] = eval_data["kid_mean"]
                log_dict["eval/kid_std"] = eval_data["kid_std"]
            # Log side-by-side student vs teacher images
            student_dir = eval_data.get("student_image_dir", "")
            teacher_dir = args.teacher_image_dir or ""
            if student_dir and os.path.isdir(student_dir) and os.path.isdir(teacher_dir):
                try:
                    import wandb
                    from PIL import Image
                    # Load val prompts for captions
                    val_prompts = []
                    if args.val_data_meta and os.path.exists(args.val_data_meta):
                        with open(args.val_data_meta) as _vf:
                            val_prompts = [r["text"] for r in _json.load(_vf)]
                    num_log = min(50, eval_data.get("num_images", 0))
                    table = wandb.Table(columns=["step", "prompt", "student", "teacher"])
                    for i in range(num_log):
                        s_path = os.path.join(student_dir, f"{i:05d}.png")
                        t_path = os.path.join(teacher_dir, f"{i:05d}.png")
                        prompt = val_prompts[i][:100] if i < len(val_prompts) else ""
                        if os.path.exists(s_path) and os.path.exists(t_path):
                            table.add_data(
                                global_step, prompt,
                                wandb.Image(Image.open(s_path)),
                                wandb.Image(Image.open(t_path)),
                            )
                    log_dict["eval/samples"] = table
                except Exception as e:
                    logger.warning(f"Failed to build wandb image table: {e}")
            accelerator.log(log_dict, step=global_step)

        # Reload training models to GPU
        device = accelerator.device
        for name, model in gpu_models:
            if model is not None:
                model.to(device)
        logger.info(f"Reloaded training models to {device}")
    else:
        # Multi-GPU: non-blocking subprocess on separate GPU
        logger.info(f"Launching validation subprocess for step {global_step} on {eval_device}")
        _val_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        logger.info(f"Validation subprocess started (pid={_val_process.pid})")


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

    # When --cpu_offload_optimizer is set, use two DeepSpeed plugins:
    # - student: ZeRO-2 with CPU optimizer offload (saves ~24GB for 6B params)
    # - disc: ZeRO-2 on GPU (small model, no offload needed)
    # Per Accelerate docs, disjoint multi-model training needs two Accelerator instances.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to != "none" else None,
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

    # Text encoder (frozen) — skip if using precomputed embeddings
    text_encoder = None
    tokenizer = None
    if not args.embeddings_dir:
        from transformers import AutoModel, AutoTokenizer as HFAutoTokenizer
        text_encoder_dir = os.path.join(args.pretrained_model_name_or_path, "text_encoder")
        text_encoder = AutoModel.from_pretrained(text_encoder_dir, dtype=weight_dtype, trust_remote_code=True)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer_dir = os.path.join(args.pretrained_model_name_or_path, "tokenizer")
        if os.path.exists(tokenizer_dir):
            tokenizer = HFAutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        else:
            tokenizer = HFAutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True)
    else:
        logger.info(f"Using precomputed embeddings from {args.embeddings_dir} — skipping text encoder")

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

    # Optimizers and LR schedulers
    # When using DeepSpeed (cpu_offload_optimizer), DS manages the optimizer/scheduler
    # internally.  We pass DummyOptim/DummyScheduler so accelerator.prepare() lets
    # DeepSpeed create the real ones from the JSON config.
    if args.cpu_offload_optimizer:
        # 8-bit Adam: uses ~2 bytes/param for optimizer states instead of 8,
        # saving ~18GB for 6B param models. Drop-in replacement for AdamW.
        # Compatible with FSDP/DDP for multi-GPU scaling.
        import bitsandbytes as bnb
        student_optimizer = bnb.optim.AdamW8bit(
            student.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        logger.info("Using 8-bit Adam for student optimizer")
    else:
        student_optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    # Discriminator is small (14M params) — always use plain PyTorch optimizer
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.learning_rate_disc,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and dataloader
    train_dataset = TextDataset(args.train_data_meta, text_drop_ratio=args.text_drop_ratio,
                                embeddings_dir=args.embeddings_dir,
                                teacher_latents_dir=args.teacher_latents_dir)

    def collate_fn(examples):
        batch = {"text": [ex["text"] for ex in examples]}
        if "embedding" in examples[0]:
            batch["embeddings"] = [ex["embedding"] for ex in examples]
        if "teacher_latent" in examples[0]:
            batch["teacher_latents"] = torch.stack([ex["teacher_latent"] for ex in examples])
        return batch

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

    # Prepare with accelerator (plain Accelerate — no DeepSpeed)
    student, student_optimizer, train_dataloader, student_lr_scheduler = accelerator.prepare(
        student, student_optimizer, train_dataloader, student_lr_scheduler,
    )
    discriminator, disc_optimizer, disc_lr_scheduler = accelerator.prepare(
        discriminator, disc_optimizer, disc_lr_scheduler,
    )

    # Move frozen models to device
    teacher.to(accelerator.device, dtype=weight_dtype)
    if text_encoder is not None:
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
    early_stop_failures = 0  # consecutive health check failures
    early_stopped = False
    last_disc_metrics = {}  # track most recent disc metrics for summary

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

                # 1. Encode prompts (or use precomputed embeddings)
                if "embeddings" in batch:
                    prompt_embeds = [e.to(accelerator.device, dtype=weight_dtype)
                                    for e in batch["embeddings"]]
                else:
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

                # 3. Create student input from teacher latent + noise
                latent_shape = (bsz, in_channels, height_latent, width_latent)
                noise = torch.randn(
                    latent_shape, device=accelerator.device,
                    generator=torch_rng, dtype=weight_dtype,
                )

                # Load precomputed teacher latent (clean x_0, generated with CFG)
                if "teacher_latents" in batch:
                    teacher_x0 = batch["teacher_latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    teacher_x0 = torch.randn_like(noise)

                # Student input at correct noise level:
                # x_t = (1 - t) * teacher_x0 + t * noise
                # At t=1.0: pure noise. At t=0.25: mostly teacher_x0.
                t_bc = student_t.view(-1, 1, 1, 1)
                x_t = ((1.0 - t_bc) * teacher_x0 + t_bc * noise).to(weight_dtype)
                student_input_list = list(x_t.unsqueeze(2).unbind(dim=0))

                # 4. Student forward -> velocity -> denoised latent
                is_gen_step = (global_step % args.gen_update_interval == 0)

                if is_gen_step:
                    # Student needs gradients
                    # Forward through wrapped module (required for FSDP all-gather)
                    student_out, _ = student(
                        student_input_list,
                        student_t,
                        prompt_embeds,
                        return_hidden_states=False,
                    )
                    student_velocity = torch.stack(student_out, dim=0).squeeze(2)
                else:
                    with torch.no_grad():
                        student_out, _ = student(
                            student_input_list,
                            student_t,
                            prompt_embeds,
                            return_hidden_states=False,
                        )
                        student_velocity = torch.stack(student_out, dim=0).squeeze(2).detach()

                # Convert velocity to denoised latent: x̂_0 = x_t - t * v
                student_pred = x_t - t_bc * student_velocity

                # 5. Re-noise both sides for teacher discrimination
                t_hat = logit_normal_sample(
                    bsz, m=args.renoise_m, s=args.renoise_s,
                    device=accelerator.device, generator=torch_rng,
                )  # (B,) in [0.001, 0.999]

                # Fake path: re-noise student's denoised prediction
                renoise = torch.randn_like(student_pred)
                student_renoised = add_noise(student_pred.detach().float(), renoise.float(), t_hat.float())
                student_renoised = student_renoised.to(weight_dtype)

                # Real path: re-noise teacher's clean latent
                real_noise = torch.randn_like(student_pred)
                real_noisy = add_noise(teacher_x0.float(), real_noise.float(), t_hat.float())
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
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                # Capture disc grad norm before zero_grad
                _disc_grad_norm = sum(
                    p.grad.norm().item() for p in discriminator.parameters()
                    if p.grad is not None
                )
                disc_optimizer.step()
                disc_lr_scheduler.step()
                disc_optimizer.zero_grad()

                # ---- Student (generator) update (every gen_update_interval steps) ----
                _student_grad_norm = 0.0
                if is_gen_step:
                    # Recompute x̂_0 with gradient path (student_velocity has grad).
                    # The teacher forward runs WITHOUT torch.no_grad() so that
                    # gradients flow through the teacher's operations back to
                    # the student (the teacher's weights are frozen via
                    # requires_grad_(False), but the computation graph is kept).
                    student_pred_grad = x_t - t_bc * student_velocity  # x̂_0 with grad
                    student_renoised_grad = add_noise(
                        student_pred_grad.float(), renoise.float(), t_hat.float()
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

                    # Use the DeepSpeed engine's backward so ZeRO-2 properly
                    # manages gradient reduction and CPU offloading.
                    # student IS the DeepSpeed engine when cpu_offload_optimizer is set.
                    accelerator.backward(g_loss_update)
                    # Capture student grad norm before clipping/zero_grad
                    _student_grad_norm = sum(
                        p.grad.float().norm().item()
                        for p in student.parameters()
                        if p.grad is not None
                    )
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    student_optimizer.step()
                    student_lr_scheduler.step()
                    student_optimizer.zero_grad()
                    # Zero disc grads that leaked through the gen step
                    disc_optimizer.zero_grad()

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

                # Discriminator health metrics
                logs.update(compute_discriminator_metrics(real_result, fake_result))

                # Gradient norms (captured before zero_grad above)
                logs["grad_norm/discriminator"] = _disc_grad_norm
                if is_gen_step:
                    logs["grad_norm/student"] = _student_grad_norm

                # Student weight change tracking — log delta from initial weights
                # to confirm weights are actually changing during training.
                # Disabled under FSDP: unwrap_model returns sharded params.
                if str(accelerator.distributed_type) != "DistributedType.FSDP":
                    student_unwrapped = accelerator.unwrap_model(student)
                    if not hasattr(main, '_initial_weights'):
                        main._initial_weights = {}
                    for _track_name, _track_param in student_unwrapped.named_parameters():
                        if any(k in _track_name for k in ["layers.0.attention.to_q.weight",
                                                           "layers.15.attention.to_q.weight",
                                                           "layers.29.attention.to_q.weight"]):
                            _cur = _track_param.data.float()
                            if _track_name not in main._initial_weights:
                                main._initial_weights[_track_name] = _cur.clone()
                            _delta = (_cur - main._initial_weights[_track_name]).norm().item()
                            logs[f"weight_delta/{_track_name.replace('.', '_')}"] = _delta

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

                # Track disc metrics for summary and early stopping
                last_disc_metrics = {
                    k: v for k, v in logs.items()
                    if k.startswith("disc/") or k in ("d_loss", "g_loss")
                }

                # Early stopping: check disc health every N steps
                if (args.early_stop
                        and global_step > args.lr_warmup_steps
                        and global_step % args.early_stop_check_interval == 0):
                    acc_real = logs.get("disc/accuracy_real", 0.5)
                    acc_fake = logs.get("disc/accuracy_fake", 0.5)
                    logit_gap = logs.get("disc/logit_gap", 1.0)

                    failed = False
                    if acc_real > 0.95 and acc_fake > 0.95:
                        logger.warning(f"Step {global_step}: disc too strong (acc_real={acc_real:.2f}, acc_fake={acc_fake:.2f})")
                        failed = True
                    elif acc_real < 0.55 and acc_fake < 0.55:
                        logger.warning(f"Step {global_step}: disc collapsed (acc_real={acc_real:.2f}, acc_fake={acc_fake:.2f})")
                        failed = True
                    elif logit_gap > 15:
                        logger.warning(f"Step {global_step}: disc diverging (logit_gap={logit_gap:.2f})")
                        failed = True
                    elif logit_gap < 0.1:
                        logger.warning(f"Step {global_step}: disc useless (logit_gap={logit_gap:.4f})")
                        failed = True

                    if failed:
                        early_stop_failures += 1
                        if early_stop_failures >= args.early_stop_patience:
                            logger.error(f"Early stopping at step {global_step}: {early_stop_failures} consecutive health check failures")
                            early_stopped = True
                    else:
                        early_stop_failures = 0

            progress_bar.update(1)
            global_step += 1

            if early_stopped:
                break

            # Checkpointing + Validation
            # Both need get_state_dict (expensive FSDP gather), so combine into one call.
            need_checkpoint = not args.skip_save and global_step % args.checkpointing_steps == 0
            need_validation = global_step % args.validation_steps == 0
            is_fsdp = str(accelerator.distributed_type) == "DistributedType.FSDP"

            if need_checkpoint or need_validation:
                # Single get_state_dict call (collective op, all ranks participate)
                state_dict = accelerator.get_state_dict(student)

                if need_checkpoint:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # accelerator.save_state() is incompatible with 8-bit Adam + FSDP
                    if not is_fsdp:
                        accelerator.save_state(save_path)
                    if accelerator.is_main_process and state_dict is not None:
                        from safetensors.torch import save_file as _save_file
                        student_save_dir = os.path.join(save_path, "student_transformer")
                        os.makedirs(student_save_dir, exist_ok=True)
                        _save_file(
                            {k: v.contiguous().cpu() for k, v in state_dict.items()},
                            os.path.join(student_save_dir, "model.safetensors"),
                        )
                        if args.checkpoints_total_limit is not None:
                            dirs = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            while len(dirs) > args.checkpoints_total_limit:
                                shutil.rmtree(os.path.join(args.output_dir, dirs.pop(0)))
                        logger.info(f"Saved checkpoint at step {global_step}")

                if need_validation:
                    _launch_validation(
                        student_model=student,
                        teacher_model=teacher,
                        discriminator_model=discriminator,
                        vae_model=vae,
                        args=args,
                        global_step=global_step,
                        accelerator=accelerator,
                        precomputed_state_dict=state_dict,
                    )

                del state_dict

        if global_step >= args.max_train_steps or early_stopped:
            break

    # Print machine-readable training summary for research agent
    if accelerator.is_main_process:
        print("\n---")
        print(f"training_steps:     {global_step}")
        print(f"early_stopped:      {early_stopped}")
        for k, v in sorted(last_disc_metrics.items()):
            if isinstance(v, float):
                print(f"{k}:  {v:.6f}")
        if torch.cuda.is_available():
            print(f"peak_vram_mb:       {torch.cuda.max_memory_allocated() / 1e6:.1f}")
        print("---\n")

    # Final save
    accelerator.wait_for_everyone()
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    is_fsdp = str(accelerator.distributed_type) == "DistributedType.FSDP"
    if args.skip_save and is_fsdp:
        # Under FSDP with --skip_save, skip final checkpoint entirely
        # (get_state_dict is expensive and mid-training checkpoints suffice)
        if accelerator.is_main_process:
            logger.info(f"Skipping final save (--skip_save + FSDP). Last checkpoint has the weights.")
    elif not args.skip_save:
        # Full save: accelerator state + student weights
        # Skip accelerator.save_state under FSDP (8-bit Adam incompatible)
        if not is_fsdp:
            accelerator.save_state(save_path)
        state_dict = accelerator.get_state_dict(student)
        if accelerator.is_main_process:
            from safetensors.torch import save_file as _save_file_final
            student_dir = os.path.join(save_path, "student_transformer")
            os.makedirs(student_dir, exist_ok=True)
            _save_file_final(
                {k: v.contiguous().cpu() for k, v in state_dict.items()},
                os.path.join(student_dir, "model.safetensors"),
            )
        del state_dict
    else:
        # Lightweight save: student weights only (safetensors, ~12GB)
        state_dict = accelerator.get_state_dict(student)
        if accelerator.is_main_process:
            from safetensors.torch import save_file
            student_dir = os.path.join(save_path, "student_transformer")
            os.makedirs(student_dir, exist_ok=True)
            save_file(
                {k: v.contiguous().cpu() for k, v in state_dict.items()},
                os.path.join(student_dir, "model.safetensors"),
            )
        del state_dict
    if accelerator.is_main_process:
        logger.info(f"Training complete. Checkpoint at {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
