"""Batch prompt inference for Z-Image."""

import os
import time
from pathlib import Path

import torch

from inference import ensure_weights
from utils import load_from_local_dir, set_attention_backend
from zimage import generate

PROMPTS = [
    "A single ripe strawberry on a white ceramic plate, high-resolution studio photograph, shallow depth of field.",
    "A sleepy orange cat curled up on a stack of physics textbooks, digital painting, soft brush strokes.",
    "An abandoned lighthouse on a rocky coast during a storm, oil painting on canvas, visible brush texture.",
    "A small campsite in a dense pine forest at night, only lit by a campfire, cinematic wide shot.",
    "A futuristic electric motorcycle parked in a neon-lit alley, 3D render, glossy materials, reflective puddles.",
    "A tiny houseplant growing out of a computer keyboard, minimalist flat illustration, pastel color palette.",
    "A mountain lake at sunrise, thick fog over the water, soft golden hour lighting, wide landscape panorama.", 
    "A lone tree in a vast desert, harsh midday sun, strong shadows, centered composition.", 
    "A close-up of a honeybee collecting pollen from a vibrant sunflower, macro photography, detailed textures.",
    "A whimsical treehouse village built among giant mushrooms, fantasy art, colorful and imaginative style.",
    "A serene beach at sunset with gentle waves and a pastel sky, impressionist painting style.",
    "A bustling city street during a rainstorm, reflections on wet pavement, cinematic noir style.",
    "A rainy city street at night viewed from above, long‑exposure look with light trails from cars."
]


def slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem-safe slug from the prompt."""

    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:max_len].rstrip("-") or "prompt"


def select_device() -> str:
    """Choose the best available device without repeating detection logic."""

    if torch.cuda.is_available():
        print("Chosen device: cuda")
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        print("Chosen device: tpu")
        return device
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            print("Chosen device: mps")
            return "mps"
        print("Chosen device: cpu")
        return "cpu"


def main():
    model_path = ensure_weights("ckpts/Z-Image-Turbo")
    dtype = torch.bfloat16
    compile = False
    height = 1024
    width = 1024
    num_inference_steps = 8
    guidance_scale = 0.0
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "native")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    device = select_device()

    components = load_from_local_dir(
        model_path, device=device, dtype=dtype, compile=compile
    )
    set_attention_backend(attn_backend)
    print(
        f"Attention backend: {attn_backend} (set ZIMAGE_ATTENTION to override, e.g., '_flash_3', 'flash', '_native_flash', 'native')"
    )

    for idx, prompt in enumerate(PROMPTS, start=1):
        output_path = output_dir / f"prompt-{idx:02d}-{slugify(prompt)}.png"
        seed = 42 + idx - 1
        generator = torch.Generator(device).manual_seed(seed)

        start_time = time.time()
        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        elapsed = time.time() - start_time
        images[0].save(output_path)
        print(f"[{idx}/{len(PROMPTS)}] Saved {output_path} in {elapsed:.2f} seconds")

    print("Done.")


if __name__ == "__main__":
    main()
