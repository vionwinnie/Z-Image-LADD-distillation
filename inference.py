"""Z-Image PyTorch Native Inference."""

import os
import time
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")
from utils import load_from_local_dir, set_attention_backend
from zimage import generate


# Before starting, weights will be auto-downloaded to `ckpts/Z-Image-Turbo` if missing.
def ensure_weights(model_path: str, repo_id: str = "Tongyi-MAI/Z-Image-Turbo") -> Path:
    """Download model weights if they are not already present locally."""

    target_dir = Path(model_path)
    config_path = target_dir / "transformer" / "config.json"

    if config_path.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {target_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Expected config not found at {config_path} after download")

    print("Download complete.")
    return target_dir


def main():
    model_path = ensure_weights("ckpts/Z-Image-Turbo")
    dtype = torch.bfloat16
    compile = False  # default False for compatibility
    output_path = "example.png"
    height = 1024
    width = 1024
    num_inference_steps = 8
    guidance_scale = 0.0
    seed = 42
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "native")
    prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    )

    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        print("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            print("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                print("Chosen device: mps")
            else:
                device = "cpu"
                print("Chosen device: cpu")
    # Load models
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    set_attention_backend(attn_backend)
    print(f"Attention backend: {attn_backend} (set ZIMAGE_ATTENTION to override, e.g., '_flash_3', 'flash', '_native_flash', 'native')")

    # Gen an image
    start_time = time.time()
    images = generate(
        prompt=prompt,
        **components,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device).manual_seed(seed),
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    images[0].save(output_path)

    ### !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`
    ### This would give you sub-second generation speed on Hopper GPU (H100/H200/H800) after warm-up


if __name__ == "__main__":
    main()
