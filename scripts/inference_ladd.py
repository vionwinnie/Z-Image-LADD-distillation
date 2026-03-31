"""Inference script for LADD-distilled Z-Image student model.

Generates images using the distilled student (4 steps) and optionally compares
with the teacher model (50 steps) side-by-side.

Usage:
    python scripts/inference_ladd.py \
        --student_checkpoint=output_ladd/checkpoint-20000 \
        --teacher_model=models/Z-Image \
        --output_dir=inference_results

    # Student-only (faster, no teacher comparison):
    python scripts/inference_ladd.py \
        --student_checkpoint=output_ladd/checkpoint-20000 \
        --teacher_model=models/Z-Image \
        --output_dir=inference_results \
        --skip_teacher
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_project_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _project_root)

from zimage.pipeline import generate, calculate_shift
from training.train_ladd import load_transformer, load_vae


DEFAULT_PROMPTS = [
    "A serene mountain landscape at golden hour with dramatic clouds",
    "A fluffy orange cat sleeping on a windowsill in warm sunlight",
    "A cyberpunk cityscape at night with neon reflections on wet streets",
    "A photorealistic portrait of an elderly man with deep wrinkles and kind eyes",
    "A plate of sushi arranged artistically on a wooden board, top-down view",
    "An ancient Chinese pagoda in a misty bamboo forest, ink wash painting style",
    "A macro photograph of a dewdrop on a red rose petal",
    "A futuristic spacecraft orbiting a ringed planet, concept art style",
    "A cozy bookstore interior with warm lighting and tall wooden shelves",
    "Chinese calligraphy of the character 龙 (dragon) in bold brush strokes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="LADD student inference + teacher comparison.")
    parser.add_argument("--student_checkpoint", type=str, required=True,
                        help="Path to student checkpoint directory (e.g. output_ladd/checkpoint-20000).")
    parser.add_argument("--teacher_model", type=str, required=True,
                        help="Path to pretrained Z-Image model directory.")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory to save generated images.")
    parser.add_argument("--prompts", nargs="+", type=str, default=None,
                        help="Custom prompts. Uses defaults if not specified.")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="JSON file with prompts (list of strings or list of dicts with 'text' key).")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--student_steps", type=int, default=4,
                        help="Number of denoising steps for student.")
    parser.add_argument("--teacher_steps", type=int, default=50,
                        help="Number of denoising steps for teacher.")
    parser.add_argument("--guidance_scale", type=float, default=0.0,
                        help="CFG scale (0 = no CFG, as typical for distilled models).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_teacher", action="store_true",
                        help="Skip teacher generation (student-only mode).")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_prompts(args):
    if args.prompts:
        return args.prompts
    if args.prompts_file:
        import json
        with open(args.prompts_file) as f:
            data = json.load(f)
        if isinstance(data[0], dict):
            return [item.get("text", item.get("prompt", "")) for item in data]
        return data
    return DEFAULT_PROMPTS


def make_comparison_grid(student_img, teacher_img, prompt_text, index):
    """Create a side-by-side comparison image with labels."""
    margin = 10
    label_height = 40
    total_width = student_img.width * 2 + margin * 3
    total_height = student_img.height + margin * 2 + label_height

    grid = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    grid.paste(student_img, (margin, margin + label_height))
    grid.paste(teacher_img, (student_img.width + margin * 2, margin + label_height))

    # Try to add text labels
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw.text((margin, margin // 2), f"Student ({args.student_steps} steps)", fill=(0, 0, 0), font=font)
        draw.text((student_img.width + margin * 2, margin // 2),
                   f"Teacher ({args.teacher_steps} steps)", fill=(0, 0, 0), font=font)

        # Truncate prompt for display
        short_prompt = prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
        draw.text((margin, total_height - margin - 5), short_prompt, fill=(100, 100, 100), font=font)
    except ImportError:
        pass  # PIL draw not available

    return grid


def main():
    global args
    args = parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    prompts = load_prompts(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "student"), exist_ok=True)
    if not args.skip_teacher:
        os.makedirs(os.path.join(args.output_dir, "teacher"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "comparison"), exist_ok=True)

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Prompts: {len(prompts)}")
    print(f"Student steps: {args.student_steps}, Teacher steps: {args.teacher_steps}")
    print()

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print("Loading student model...")
    student = load_transformer(args.teacher_model, dtype)

    # Load student checkpoint weights
    student_weights_path = os.path.join(args.student_checkpoint, "student_transformer", "pytorch_model.bin")
    if os.path.exists(student_weights_path):
        student_sd = torch.load(student_weights_path, map_location="cpu", weights_only=True)
        student.load_state_dict(student_sd, strict=False)
        del student_sd
        print(f"  Loaded student weights from {student_weights_path}")
    else:
        # Try safetensors
        from safetensors.torch import load_file
        safetensor_path = os.path.join(args.student_checkpoint, "student_transformer")
        safetensor_files = [f for f in os.listdir(safetensor_path) if f.endswith(".safetensors")]
        if safetensor_files:
            student_sd = {}
            for sf in safetensor_files:
                student_sd.update(load_file(os.path.join(safetensor_path, sf)))
            student.load_state_dict(student_sd, strict=False)
            del student_sd
            print(f"  Loaded student weights from safetensors in {safetensor_path}")
        else:
            print(f"  WARNING: No student weights found at {student_weights_path}")
            print(f"  Using base model weights (this is expected for testing)")

    student.to(device)
    student.eval()

    # Load shared components
    print("Loading VAE...")
    vae = load_vae(args.teacher_model)
    vae.to(device, dtype=torch.float32)
    vae.eval()

    print("Loading text encoder...")
    from transformers import AutoModel, AutoTokenizer
    text_encoder_dir = os.path.join(args.teacher_model, "text_encoder")
    tokenizer_dir = os.path.join(args.teacher_model, "tokenizer")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, torch_dtype=dtype, trust_remote_code=True)
    text_encoder.to(device)
    text_encoder.eval()
    if os.path.exists(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True)

    print("Loading scheduler...")
    import json
    scheduler_config_path = os.path.join(args.teacher_model, "scheduler", "scheduler_config.json")
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

    # Load teacher if needed
    teacher = None
    if not args.skip_teacher:
        print("Loading teacher model...")
        teacher = load_transformer(args.teacher_model, dtype)
        teacher.to(device)
        teacher.eval()

    print("\nAll models loaded. Starting generation...\n")

    # -----------------------------------------------------------------------
    # Generate images
    # -----------------------------------------------------------------------
    generator = torch.Generator(device).manual_seed(args.seed)
    student_times = []
    teacher_times = []

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

        # Student generation
        gen_student = torch.Generator(device).manual_seed(args.seed + i)
        t0 = time.time()
        student_images = generate(
            transformer=student,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.student_steps,
            guidance_scale=args.guidance_scale,
            generator=gen_student,
        )
        student_time = time.time() - t0
        student_times.append(student_time)

        student_img = student_images[0]
        student_path = os.path.join(args.output_dir, "student", f"{i:03d}.png")
        student_img.save(student_path)
        print(f"  Student: {student_time:.2f}s -> {student_path}")

        # Teacher generation (optional)
        if teacher is not None:
            gen_teacher = torch.Generator(device).manual_seed(args.seed + i)
            t0 = time.time()
            teacher_images = generate(
                transformer=teacher,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.teacher_steps,
                guidance_scale=args.guidance_scale,
                generator=gen_teacher,
            )
            teacher_time = time.time() - t0
            teacher_times.append(teacher_time)

            teacher_img = teacher_images[0]
            teacher_path = os.path.join(args.output_dir, "teacher", f"{i:03d}.png")
            teacher_img.save(teacher_path)
            print(f"  Teacher: {teacher_time:.2f}s -> {teacher_path}")

            # Comparison grid
            grid = make_comparison_grid(student_img, teacher_img, prompt, i)
            grid_path = os.path.join(args.output_dir, "comparison", f"{i:03d}.png")
            grid.save(grid_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Prompts generated: {len(prompts)}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Student ({args.student_steps} steps):")
    print(f"    Mean latency: {sum(student_times)/len(student_times):.2f}s")
    print(f"    Total time:   {sum(student_times):.2f}s")
    if teacher_times:
        print(f"  Teacher ({args.teacher_steps} steps):")
        print(f"    Mean latency: {sum(teacher_times)/len(teacher_times):.2f}s")
        print(f"    Total time:   {sum(teacher_times):.2f}s")
        speedup = (sum(teacher_times) / len(teacher_times)) / (sum(student_times) / len(student_times))
        print(f"  Speedup: {speedup:.1f}x")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
