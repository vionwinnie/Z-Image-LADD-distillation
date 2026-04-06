"""LADD Discriminator heads for adversarial distillation.

Each head is a lightweight 2D conv network applied to teacher hidden states
at a selected layer index. FiLM conditioning from timestep + text is applied
via scale/shift modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zimage.transformer import TimestepEmbedder


class LADDDiscriminatorHead(nn.Module):
    """Lightweight 2D conv head applied to one teacher attention block's features."""

    def __init__(self, feature_dim: int = 3840, hidden_dim: int = 256, cond_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # FiLM conditioning: timestep_embed + text_embed -> MLP -> (scale, shift)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # scale + shift
        )

        # Project from token dim to manageable size
        self.proj = nn.Linear(feature_dim, hidden_dim)

        # 2D conv layers (applied after reshaping tokens to spatial layout)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.gn2 = nn.GroupNorm(16, hidden_dim // 2)
        self.conv_out = nn.Conv2d(hidden_dim // 2, 1, 1)

    def forward(
        self,
        features: torch.Tensor,
        spatial_size: tuple,
        t_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features:    (B, num_image_tokens, feature_dim) -- image tokens only
            spatial_size: (H_tokens, W_tokens) for reshaping
            t_embed:     (B, cond_dim) -- timestep embedding
            text_embed:  (B, cond_dim) -- pooled text embedding (mean of text tokens)

        Returns:
            logits: (B,) -- mean logit per sample
        """
        B = features.shape[0]
        H_t, W_t = spatial_size

        # Project to hidden dim
        h = self.proj(features)  # (B, N, hidden_dim)

        # FiLM conditioning
        cond = torch.cat([t_embed, text_embed], dim=-1)  # (B, cond_dim * 2)
        film_params = self.cond_mlp(cond)  # (B, hidden_dim * 2)
        scale, shift = film_params.chunk(2, dim=-1)  # each (B, hidden_dim)
        scale = 1.0 + scale.unsqueeze(1)  # (B, 1, hidden_dim)
        shift = shift.unsqueeze(1)        # (B, 1, hidden_dim)
        h = h * scale + shift

        # Reshape to spatial: take only the first H_t * W_t tokens (ignore padding)
        h = h[:, :H_t * W_t, :]  # (B, H_t * W_t, hidden_dim)
        h = h.permute(0, 2, 1).reshape(B, self.hidden_dim, H_t, W_t)

        # 2D conv blocks
        h = F.silu(self.gn1(self.conv1(h)))
        h = F.silu(self.gn2(self.conv2(h)))
        h = self.conv_out(h)  # (B, 1, H_t, W_t)

        # Mean logit per sample
        logits = h.mean(dim=[1, 2, 3])  # (B,)
        return logits


class LADDDiscriminator(nn.Module):
    """Collection of discriminator heads, one per selected teacher layer."""

    def __init__(
        self,
        feature_dim: int = 3840,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        layer_indices: tuple = (5, 10, 15, 20, 25, 29),
        clip_dim: int = 0,
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.use_clip = clip_dim > 0
        self.heads = nn.ModuleDict(
            {str(i): LADDDiscriminatorHead(feature_dim, hidden_dim, cond_dim) for i in layer_indices}
        )
        # Shared timestep embedder
        self.t_embedder = TimestepEmbedder(cond_dim, mid_size=1024)
        # Project text embed -> cond_dim
        text_input_dim = clip_dim if self.use_clip else feature_dim
        self.text_proj = nn.Linear(text_input_dim, cond_dim)

    def _extract_image_features(self, hidden_states, x_item_seqlens):
        """Extract image-only tokens from unified hidden states.

        Args:
            hidden_states: (B, max_seq, dim) -- unified (image + text) padded tensor
            x_item_seqlens: list of int -- number of image tokens per sample

        Returns:
            image_features: (B, max_img_tokens, dim) -- zero-padded image features
        """
        B = hidden_states.shape[0]
        max_img_len = max(x_item_seqlens)
        dim = hidden_states.shape[-1]

        image_features = hidden_states.new_zeros(B, max_img_len, dim)
        for i in range(B):
            img_len = x_item_seqlens[i]
            # Image tokens come first in the unified sequence (see transformer.py line 552)
            image_features[i, :img_len] = hidden_states[i, :img_len]
        return image_features

    def _extract_text_pooled(self, hidden_states, x_item_seqlens, cap_item_seqlens):
        """Extract mean-pooled text embedding from unified hidden states.

        Args:
            hidden_states: (B, max_seq, dim)
            x_item_seqlens: list of int -- image token counts
            cap_item_seqlens: list of int -- text token counts

        Returns:
            text_pooled: (B, dim)
        """
        B = hidden_states.shape[0]
        dim = hidden_states.shape[-1]
        text_pooled = hidden_states.new_zeros(B, dim)
        for i in range(B):
            img_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            text_tokens = hidden_states[i, img_len:img_len + cap_len]
            text_pooled[i] = text_tokens.mean(dim=0)
        return text_pooled

    def forward(
        self,
        hidden_states_list: list,
        x_item_seqlens: list,
        cap_item_seqlens: list,
        spatial_sizes: list,
        timesteps: torch.Tensor,
        clip_text_embeds: torch.Tensor = None,
    ) -> dict:
        """Compute per-head logits.

        Args:
            hidden_states_list: list of 30 tensors (B, max_seq, dim) from teacher layers
            x_item_seqlens: list of int -- image token counts per sample
            cap_item_seqlens: list of int -- text token counts per sample
            spatial_sizes: list of (H_tokens, W_tokens) per sample (use first for batch)
            timesteps: (B,) -- float timesteps in [0, 1]
            clip_text_embeds: (B, clip_dim) -- precomputed CLIP text embeddings (optional)

        Returns:
            dict with keys: 'logits' (dict of layer_idx -> (B,)), 'total_logit' (B,)
        """
        # Cast inputs to match parameter dtype (needed when DeepSpeed wraps this module)
        param_dtype = next(self.parameters()).dtype
        timesteps = timesteps.to(param_dtype)
        hidden_states_list = [hs.to(param_dtype) for hs in hidden_states_list]

        # Timestep embedding -- t_embedder expects raw timesteps, scale by 1000 as in transformer
        t_embed = self.t_embedder(timesteps * 1000.0)  # (B, cond_dim)

        # Text embedding: use CLIP if provided, otherwise extract from teacher hidden states
        if self.use_clip and clip_text_embeds is not None:
            text_embed = self.text_proj(clip_text_embeds.to(param_dtype))  # (B, cond_dim)
        else:
            text_embed = None  # computed per-layer below

        # Use the first spatial size (assumes uniform batch for simplicity)
        spatial_size = spatial_sizes[0] if isinstance(spatial_sizes[0], tuple) else spatial_sizes

        logits_dict = {}
        total_logit = None

        for layer_idx in self.layer_indices:
            hs = hidden_states_list[layer_idx]  # (B, max_seq, dim)

            # Extract image features
            img_feats = self._extract_image_features(hs, x_item_seqlens)

            # Per-layer text pooling fallback (original behavior)
            if text_embed is None:
                text_pooled = self._extract_text_pooled(hs, x_item_seqlens, cap_item_seqlens)
                layer_text_embed = self.text_proj(text_pooled)
            else:
                layer_text_embed = text_embed

            head_logits = self.heads[str(layer_idx)](img_feats, spatial_size, t_embed, layer_text_embed)
            logits_dict[layer_idx] = head_logits

            if total_logit is None:
                total_logit = head_logits
            else:
                total_logit = total_logit + head_logits

        return {"logits": logits_dict, "total_logit": total_logit}

    @staticmethod
    def compute_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor):
        """Hinge loss for GAN training.

        Args:
            real_logits: (B,) from teacher features
            fake_logits: (B,) from student features

        Returns:
            d_loss: discriminator loss (scalar)
            g_loss: generator loss (scalar)
        """
        d_loss = torch.mean(F.relu(1.0 - real_logits)) + torch.mean(F.relu(1.0 + fake_logits))
        g_loss = -torch.mean(fake_logits)
        return d_loss, g_loss
