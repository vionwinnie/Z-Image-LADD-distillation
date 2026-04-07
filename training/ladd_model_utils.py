"""Shared model loading helpers for LADD training and evaluation."""

import os
import sys
from pathlib import Path

import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_project_root, "src")
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import (
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
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
)
from zimage.transformer import ZImageTransformer2DModel
from zimage.autoencoder import AutoencoderKL


def _build_transformer_from_config(model_dir: str, dtype: torch.dtype, device: str = "meta"):
    """Create a ZImageTransformer2DModel shell on the given device (no weights loaded)."""
    from utils.loader import load_config

    transformer_dir = Path(model_dir) / "transformer"
    config = load_config(str(transformer_dir / "config.json"))

    with torch.device(device):
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
    return transformer


def load_transformer(model_dir: str, dtype: torch.dtype) -> ZImageTransformer2DModel:
    """Load a ZImageTransformer2DModel from a directory with config.json + safetensors."""
    from utils.loader import load_sharded_safetensors

    transformer = _build_transformer_from_config(model_dir, dtype, device="meta")
    transformer_dir = Path(model_dir) / "transformer"
    state_dict = load_sharded_safetensors(transformer_dir, device="cpu", dtype=dtype)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    return transformer


def load_transformer_meta(model_dir: str, dtype: torch.dtype) -> ZImageTransformer2DModel:
    """Create a ZImageTransformer2DModel on meta device (no weights). For FSDP sync_module_states."""
    return _build_transformer_from_config(model_dir, dtype, device="meta")


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
