"""Utilities for LADD training.

Includes timestep sampling, text dataset, prompt encoding, and noise utilities.
"""

import json
import math
import random
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Discrete timestep sampling (adapted from VideoX-Fun)
# ---------------------------------------------------------------------------

class DiscreteSampling:
    """Sample discrete timestep indices, optionally distributed across ranks.

    Args:
        num_timesteps: Total number of training timesteps (e.g. 1000).
        uniform_sampling: If True, sample uniformly. Otherwise use logit-normal.
    """

    def __init__(self, num_timesteps: int = 1000, uniform_sampling: bool = True):
        self.num_timesteps = num_timesteps
        self.uniform_sampling = uniform_sampling

    def __call__(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        if self.uniform_sampling:
            indices = torch.randint(
                0, self.num_timesteps, (batch_size,),
                generator=generator, device=device,
            )
        else:
            # Logit-normal centered distribution
            u = torch.normal(
                mean=0.0, std=1.0, size=(batch_size,),
                generator=generator, device=device,
            )
            t = torch.sigmoid(u)
            indices = (t * self.num_timesteps).clamp(0, self.num_timesteps - 1).long()
        return indices


# ---------------------------------------------------------------------------
# Logit-normal sampling for re-noising timesteps
# ---------------------------------------------------------------------------

def logit_normal_sample(
    batch_size: int,
    m: float = 1.0,
    s: float = 1.0,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample from logit-normal distribution for re-noising timesteps.

    u ~ Normal(m, s^2), t = sigmoid(u), clamped to [0.001, 0.999].

    Returns:
        t: (batch_size,) float tensor in [0.001, 0.999]
    """
    u = torch.normal(mean=m, std=s, size=(batch_size,), generator=generator, device=device)
    t = torch.sigmoid(u)
    t = t.clamp(0.001, 0.999)
    return t


# ---------------------------------------------------------------------------
# Text dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Simple JSON prompt dataset.

    Expects a JSON file that is a list of dicts, each with a "text" key.

    Args:
        ann_path: Path to the JSON annotation file.
        text_drop_ratio: Probability of dropping text (classifier-free guidance).
    """

    def __init__(self, ann_path: str, text_drop_ratio: float = 0.0):
        with open(ann_path, "r") as f:
            self.dataset = json.load(f)
        self.text_drop_ratio = text_drop_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item.get("text", item.get("prompt", ""))
        if random.random() < self.text_drop_ratio:
            text = ""
        return {"text": text, "idx": idx}


# ---------------------------------------------------------------------------
# Prompt encoding (adapted from pipeline.py / VideoX-Fun train_distill.py)
# ---------------------------------------------------------------------------

def encode_prompt(
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    text_encoder=None,
    tokenizer=None,
    max_sequence_length: int = 512,
) -> List[torch.FloatTensor]:
    """Encode text prompts through the text encoder, returning variable-length embeddings.

    Uses tokenizer.apply_chat_template with enable_thinking=True (Qwen3 style).

    Returns:
        List of tensors, one per prompt, each of shape (seq_len, embed_dim).
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    # Format prompts through chat template
    for i, prompt_item in enumerate(prompt):
        messages = [{"role": "user", "content": prompt_item}]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []
    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list


# ---------------------------------------------------------------------------
# Scheduler shift calculation
# ---------------------------------------------------------------------------

def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent shift for flow matching scheduler."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# ---------------------------------------------------------------------------
# Noise addition (flow matching interpolation)
# ---------------------------------------------------------------------------

def add_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Add noise to clean samples using flow matching interpolation.

    x_t = (1 - sigma) * x0 + sigma * noise

    Args:
        x0: Clean samples, any shape with batch dim first.
        noise: Gaussian noise, same shape as x0.
        sigma: (B,) or broadcastable -- noise level in [0, 1].

    Returns:
        x_t: Noised samples, same shape as x0.
    """
    # Expand sigma to match x0 dimensions
    while sigma.ndim < x0.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * x0 + sigma * noise


def get_sigmas_from_timesteps(
    timesteps: torch.Tensor,
    noise_scheduler,
) -> torch.Tensor:
    """Convert scheduler timesteps to sigma values.

    Args:
        timesteps: (B,) -- timesteps from the noise scheduler.
        noise_scheduler: Scheduler with .timesteps and .sigmas attributes.

    Returns:
        sigma: (B,) float tensor.
    """
    device = timesteps.device
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    sigmas = noise_scheduler.sigmas.to(device)

    step_indices = []
    for t in timesteps:
        idx = torch.argmin(torch.abs(schedule_timesteps - t)).item()
        step_indices.append(idx)
    step_indices = torch.tensor(step_indices, device=device)
    return sigmas[step_indices].float()
