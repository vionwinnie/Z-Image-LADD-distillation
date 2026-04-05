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
import gc
import json
import logging
import math
import os
import shutil
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

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
    DEFAULT_TRANSFORMER_DIM,
    DEFAULT_TRANSFORMER_IN_CHANNELS,
    MAX_IMAGE_SEQ_LEN,
    MAX_SHIFT,
)
from zimage.transformer import ZImageTransformer2DModel
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
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--student_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/ladd")

    # Data
    parser.add_argument("--train_data_meta", type=str, required=True)
    parser.add_argument("--text_drop_ratio", type=float, default=0.1)
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--teacher_latents_dir", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    # Image / latent
    parser.add_argument("--image_sample_size", type=int, default=512)

    # Training
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--num_train_epochs", type=int, default=1000)

    # Optimizers
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--learning_rate_disc", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    # LADD-specific
    parser.add_argument("--gen_update_interval", type=int, default=5)
    parser.add_argument("--disc_layer_indices", nargs="+", type=int, default=[5, 10, 15, 20, 25, 29])
    parser.add_argument("--disc_hidden_dim", type=int, default=256)
    parser.add_argument("--disc_cond_dim", type=int, default=256)
    parser.add_argument("--student_timesteps", nargs="+", type=float, default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--warmup_schedule_steps", type=int, default=500)
    parser.add_argument("--renoise_m", type=float, default=1.0)
    parser.add_argument("--renoise_s", type=float, default=1.0)

    # Infrastructure
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--cpu_offload_optimizer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["tensorboard", "wandb", "all", "none"])
    parser.add_argument("--tracker_project_name", type=str, default="ladd")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default="yeun-yeungs")
    parser.add_argument("--local_rank", type=int, default=-1)

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--skip_save", action="store_true")

    # Validation
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--validation_prompts", nargs="*", type=str, default=[
        "A beautiful sunset over the ocean with golden clouds",
        "A cat sitting on a windowsill looking outside",
        "A futuristic city skyline at night with neon lights",
        "A watercolor painting of a mountain landscape",
    ])
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--eval_num_images", type=int, default=1000)
    parser.add_argument("--val_data_meta", type=str, default="data/val/metadata.json")
    parser.add_argument("--teacher_image_dir", type=str, default="data/val/teacher_images")
    parser.add_argument("--eval_device", type=str, default=None)

    # Early stopping
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_check_interval", type=int, default=50)
    parser.add_argument("--early_stop_patience", type=int, default=3)

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
# Helpers
# ---------------------------------------------------------------------------

def sample_student_timestep(
    batch_size: int,
    global_step: int,
    student_timesteps: list,
    warmup_steps: int = 500,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample student denoising timestep with warmup schedule."""
    n = len(student_timesteps)
    if global_step < warmup_steps:
        probs = [0.0] * max(0, n - 2) + [0.5, 0.5]
        while len(probs) < n:
            probs.insert(0, 0.0)
    else:
        probs = [0.7] + [0.3 / max(1, n - 1)] * (n - 1)

    probs = torch.tensor(probs[:n], device=device)
    probs = probs / probs.sum()
    indices = torch.multinomial(probs.expand(batch_size, -1), num_samples=1, generator=generator).squeeze(-1)
    timestep_options = torch.tensor(student_timesteps, device=device)
    return timestep_options[indices]


def _is_fsdp(accelerator):
    return str(accelerator.distributed_type) == "DistributedType.FSDP"


def _save_checkpoint(student, accelerator, save_path, args):
    """Save student checkpoint. Uses DCP sharded save under FSDP, safetensors otherwise."""
    if _is_fsdp(accelerator):
        import torch.distributed.checkpoint as dcp
        dcp.save({"model": student.state_dict()}, checkpoint_id=save_path)
    else:
        accelerator.save_state(save_path)
        state_dict = accelerator.get_state_dict(student)
        if accelerator.is_main_process and state_dict is not None:
            from safetensors.torch import save_file
            student_dir = os.path.join(save_path, "student_transformer")
            os.makedirs(student_dir, exist_ok=True)
            save_file(
                {k: v.contiguous().cpu() for k, v in state_dict.items()},
                os.path.join(student_dir, "model.safetensors"),
            )
        del state_dict

    if accelerator.is_main_process:
        if args.checkpoints_total_limit is not None:
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]),
            )
            while len(dirs) > args.checkpoints_total_limit:
                shutil.rmtree(os.path.join(args.output_dir, dirs.pop(0)))
        logger.info(f"Saved checkpoint at step {os.path.basename(save_path)}")


def _run_validation(student, vae, text_encoder, tokenizer, noise_scheduler,
                    val_embeddings, val_prompts_text,
                    accelerator, args, global_step, weight_dtype):
    """Generate sample images using the FSDP-wrapped student directly (no state dict gather).

    All ranks must call this (FSDP forward is collective). Only rank 0 logs to wandb.
    Uses text_encoder if available, otherwise falls back to precomputed val_embeddings.
    """
    if accelerator.is_main_process:
        logger.info(f"Validation at step {global_step}...")

    student.eval()
    images = []
    prompts_for_log = []

    with torch.no_grad():
        if text_encoder is not None and tokenizer is not None:
            from zimage.pipeline import generate as pipeline_generate
            for prompt in (args.validation_prompts or []):
                img_list = pipeline_generate(
                    transformer=student, vae=vae,
                    text_encoder=text_encoder, tokenizer=tokenizer,
                    scheduler=noise_scheduler, prompt=prompt,
                    height=args.image_sample_size, width=args.image_sample_size,
                    num_inference_steps=args.num_inference_steps, guidance_scale=0.0,
                )
                images.extend(img_list)
                prompts_for_log.append(prompt)
        elif val_embeddings is not None and len(val_embeddings) > 0:
            from zimage.pipeline import generate_from_embeddings
            n = min(args.eval_num_images, len(val_embeddings))
            for i in range(n):
                emb = val_embeddings[i].to(accelerator.device, dtype=weight_dtype)
                img_list = generate_from_embeddings(
                    transformer=student, vae=vae,
                    prompt_embeds_list=[emb], scheduler=noise_scheduler,
                    height=args.image_sample_size, width=args.image_sample_size,
                    num_inference_steps=args.num_inference_steps,
                )
                images.extend(img_list)
                prompts_for_log.append(val_prompts_text[i][:100] if i < len(val_prompts_text) else "")
        else:
            if accelerator.is_main_process:
                logger.info("Skipping validation: no text encoder or precomputed embeddings")
            student.train()
            return

    student.train()

    if accelerator.is_main_process:
        logger.info(f"Generated {len(images)} validation images")

        # Log to wandb
        if wandb is not None:
            try:
                table = wandb.Table(columns=["step", "prompt", "image"])
                for i, img in enumerate(images):
                    caption = prompts_for_log[i] if i < len(prompts_for_log) else ""
                    table.add_data(global_step, caption, wandb.Image(img))
                accelerator.log({"eval/samples": table}, step=global_step)
                logger.info(f"Logged sample table to wandb")
            except Exception as e:
                logger.warning(f"Failed to log wandb image table: {e}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to != "none" else None,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    logger.info("Loading models...")

    student = load_transformer(args.pretrained_model_name_or_path, weight_dtype)
    if args.student_path is not None:
        from safetensors.torch import load_file
        sd = load_file(args.student_path)
        student.load_state_dict(sd, strict=False)
        del sd
        logger.info(f"Loaded student weights from {args.student_path}")

    teacher = load_transformer(args.pretrained_model_name_or_path, weight_dtype)
    teacher.requires_grad_(False)
    teacher.eval()

    discriminator = LADDDiscriminator(
        feature_dim=DEFAULT_TRANSFORMER_DIM,
        hidden_dim=args.disc_hidden_dim,
        cond_dim=args.disc_cond_dim,
        layer_indices=tuple(args.disc_layer_indices),
    )

    vae = load_vae(args.pretrained_model_name_or_path)
    vae.requires_grad_(False)
    vae.eval()

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
    scheduler_config_path = os.path.join(args.pretrained_model_name_or_path, "scheduler", "scheduler_config.json")
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
        if hasattr(student, "enable_gradient_checkpointing"):
            student.enable_gradient_checkpointing()
        else:
            logger.info("Student model does not support enable_gradient_checkpointing, using torch.utils.checkpoint manually.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizers
    if args.cpu_offload_optimizer:
        import bitsandbytes as bnb
        student_optimizer = bnb.optim.AdamW8bit(
            student.parameters(), lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
        )
        logger.info("Using 8-bit Adam for student optimizer")
    else:
        student_optimizer = torch.optim.AdamW(
            student.parameters(), lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
        )
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=args.learning_rate_disc,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
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
        batch_size=args.train_batch_size, drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    # LR schedulers
    from diffusers.optimization import get_scheduler
    student_lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=student_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    disc_lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=disc_optimizer,
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
    if text_encoder is not None:
        text_encoder.to(accelerator.device)
    vae.to(accelerator.device, dtype=torch.float32)

    # Load validation embeddings (precomputed, for validation without text encoder)
    val_embeddings = None
    val_prompts_text = []
    val_emb_path = os.path.join(os.path.dirname(args.val_data_meta), "embeddings", "embeddings.pt")
    if os.path.exists(val_emb_path):
        val_embeddings = torch.load(val_emb_path, map_location="cpu", weights_only=True)
        logger.info(f"Loaded {len(val_embeddings)} validation embeddings from {val_emb_path}")
    if os.path.exists(args.val_data_meta):
        try:
            with open(args.val_data_meta) as f:
                val_prompts_text = [r["text"] for r in json.load(f)]
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = {k: v for k, v in vars(args).items() if not isinstance(v, list)}
        init_kwargs = {}
        if args.report_to in ("wandb", "all") and wandb is not None:
            init_kwargs["wandb"] = {"name": args.wandb_run_name, "entity": args.wandb_entity}
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
    early_stop_failures = 0
    early_stopped = False
    last_disc_metrics = {}

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
        range(args.max_train_steps), initial=global_step,
        desc="LADD Steps", disable=not accelerator.is_local_main_process,
    )

    # Precompute latent shape info
    vae_scale_factor = 8
    if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_scale = vae_scale_factor * 2
    height_latent = 2 * (args.image_sample_size // vae_scale)
    width_latent = 2 * (args.image_sample_size // vae_scale)
    in_channels = DEFAULT_TRANSFORMER_IN_CHANNELS
    H_tokens = height_latent // 2
    W_tokens = width_latent // 2
    image_seq_len = H_tokens * W_tokens

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

            with torch.amp.autocast("cuda", dtype=weight_dtype):
                bsz = len(batch["text"])

                # 1. Encode prompts
                if "embeddings" in batch:
                    prompt_embeds = [e.to(accelerator.device, dtype=weight_dtype) for e in batch["embeddings"]]
                else:
                    with torch.no_grad():
                        prompt_embeds = encode_prompt(
                            batch["text"], device=accelerator.device,
                            text_encoder=text_encoder, tokenizer=tokenizer,
                            max_sequence_length=args.max_sequence_length,
                        )

                # 2. Sample student timestep
                student_t = sample_student_timestep(
                    bsz, global_step, args.student_timesteps,
                    warmup_steps=args.warmup_schedule_steps,
                    device=accelerator.device, generator=torch_rng,
                )

                # 3. Create student input from teacher latent + noise
                noise = torch.randn(
                    (bsz, in_channels, height_latent, width_latent),
                    device=accelerator.device, generator=torch_rng, dtype=weight_dtype,
                )
                if "teacher_latents" in batch:
                    teacher_x0 = batch["teacher_latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    teacher_x0 = torch.randn_like(noise)

                t_bc = student_t.view(-1, 1, 1, 1)
                x_t = ((1.0 - t_bc) * teacher_x0 + t_bc * noise).to(weight_dtype)
                student_input_list = list(x_t.unsqueeze(2).unbind(dim=0))

                # 4. Student forward
                is_gen_step = (global_step % args.gen_update_interval == 0)
                if is_gen_step:
                    student_out, _ = student(student_input_list, student_t, prompt_embeds, return_hidden_states=False)
                    student_velocity = torch.stack(student_out, dim=0).squeeze(2)
                else:
                    with torch.no_grad():
                        student_out, _ = student(student_input_list, student_t, prompt_embeds, return_hidden_states=False)
                        student_velocity = torch.stack(student_out, dim=0).squeeze(2).detach()

                student_pred = x_t - t_bc * student_velocity

                # 5. Re-noise both sides for teacher discrimination
                t_hat = logit_normal_sample(bsz, m=args.renoise_m, s=args.renoise_s,
                                            device=accelerator.device, generator=torch_rng)
                renoise = torch.randn_like(student_pred)
                student_renoised = add_noise(student_pred.detach().float(), renoise.float(), t_hat.float()).to(weight_dtype)
                real_noise = torch.randn_like(student_pred)
                real_noisy = add_noise(teacher_x0.float(), real_noise.float(), t_hat.float()).to(weight_dtype)

                # 6. Teacher forward
                with torch.no_grad():
                    fake_input_list = list(student_renoised.unsqueeze(2).unbind(dim=0))
                    _, fake_extras = teacher(fake_input_list, t_hat, prompt_embeds, return_hidden_states=True)

                    real_input_list = list(real_noisy.unsqueeze(2).unbind(dim=0))
                    _, real_extras = teacher(real_input_list, t_hat, prompt_embeds, return_hidden_states=True)

                # 7. Discriminator
                spatial_sizes = [(H_tokens, W_tokens)] * bsz
                fake_result = discriminator(fake_extras["hidden_states"], fake_extras["x_item_seqlens"],
                                            fake_extras["cap_item_seqlens"], spatial_sizes, t_hat)
                real_result = discriminator(real_extras["hidden_states"], real_extras["x_item_seqlens"],
                                            real_extras["cap_item_seqlens"], spatial_sizes, t_hat)

                # 8. Losses
                d_loss, g_loss = LADDDiscriminator.compute_loss(real_result["total_logit"], fake_result["total_logit"])

                # ---- Discriminator update ----
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                disc_optimizer.step()
                disc_lr_scheduler.step()
                disc_optimizer.zero_grad()

                # ---- Student (generator) update ----
                if is_gen_step:
                    student_pred_grad = x_t - t_bc * student_velocity
                    student_renoised_grad = add_noise(
                        student_pred_grad.float(), renoise.float(), t_hat.float()
                    ).to(weight_dtype)
                    fake_input_grad = list(student_renoised_grad.unsqueeze(2).unbind(dim=0))

                    _, fake_extras_grad = teacher(fake_input_grad, t_hat, prompt_embeds, return_hidden_states=True)
                    fake_result_grad = discriminator(
                        fake_extras_grad["hidden_states"], fake_extras_grad["x_item_seqlens"],
                        fake_extras_grad["cap_item_seqlens"], spatial_sizes, t_hat,
                    )
                    g_loss_update = -torch.mean(fake_result_grad["total_logit"])

                    accelerator.backward(g_loss_update)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    student_optimizer.step()
                    student_lr_scheduler.step()
                    student_optimizer.zero_grad()
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
                logs.update(compute_discriminator_metrics(real_result, fake_result))

                if torch.cuda.is_available():
                    logs["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                    logs["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(
                    d_loss=f"{logs['d_loss']:.4f}",
                    g_loss=f"{logs['g_loss']:.4f}",
                    lr_s=f"{logs['lr_student']:.2e}",
                )

                last_disc_metrics = {k: v for k, v in logs.items()
                                     if k.startswith("disc/") or k in ("d_loss", "g_loss")}

                # Early stopping
                if (args.early_stop
                        and global_step > args.lr_warmup_steps
                        and global_step % args.early_stop_check_interval == 0):
                    acc_real = logs.get("disc/accuracy_real", 0.5)
                    acc_fake = logs.get("disc/accuracy_fake", 0.5)
                    logit_gap = logs.get("disc/logit_gap", 1.0)

                    failed = (
                        (acc_real > 0.95 and acc_fake > 0.95)
                        or (acc_real < 0.55 and acc_fake < 0.55)
                        or logit_gap > 15
                        or logit_gap < 0.1
                    )
                    if failed:
                        early_stop_failures += 1
                        if early_stop_failures >= args.early_stop_patience:
                            logger.error(f"Early stopping at step {global_step}")
                            early_stopped = True
                    else:
                        early_stop_failures = 0

            progress_bar.update(1)
            global_step += 1

            if early_stopped:
                break

            # Checkpointing
            if not args.skip_save and global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                _save_checkpoint(student, accelerator, save_path, args)

            # Validation
            if global_step % args.validation_steps == 0:
                _run_validation(student, vae, text_encoder, tokenizer,
                                noise_scheduler, val_embeddings, val_prompts_text,
                                accelerator, args, global_step, weight_dtype)

        if global_step >= args.max_train_steps or early_stopped:
            break

    # Summary
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
    if not args.skip_save:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        _save_checkpoint(student, accelerator, save_path, args)

    if accelerator.is_main_process:
        logger.info(f"Training complete.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
