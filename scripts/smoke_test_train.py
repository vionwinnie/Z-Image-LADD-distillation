"""Smoke test for training/train.py — both baseline MSE and --use_ladd paths.

Validates that the components used by train.py (as opposed to train_ladd.py)
work correctly: model creation, forward pass, loss, gradient flow, optimizer
step, and checkpoint round-trip.  Uses tiny dummy models so no real weights
are needed.

Usage:
    python scripts/smoke_test_train.py --dummy
"""

import argparse
import os
import shutil
import sys
import tempfile
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — mirrors what train.py does
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_project_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _project_root)

from config import (
    DEFAULT_TRANSFORMER_DIM,
    DEFAULT_TRANSFORMER_IN_CHANNELS,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
)
from zimage.transformer import ZImageTransformer2DModel
from training.ladd_discriminator import LADDDiscriminator
from training.ladd_utils import add_noise, logit_normal_sample

# Re-use helpers from the existing smoke test
from scripts.smoke_test import (
    TINY_CAP_FEAT_DIM,
    TINY_DIM,
    TINY_N_HEADS,
    TINY_N_KV_HEADS,
    TINY_N_LAYERS,
    TINY_N_REFINER_LAYERS,
    DummyTextEncoder,
    DummyTokenizer,
    create_tiny_transformer,
    encode_prompt_dummy,
    print_header,
    print_pass,
    print_fail,
)


# ---------------------------------------------------------------------------
# Functions copied from training/train.py to avoid importing the full module
# (which pulls in torchvision, diffusers, datasets, etc.)
# ---------------------------------------------------------------------------

def encode_prompt(
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    text_encoder=None,
    tokenizer=None,
    max_sequence_length: int = 512,
) -> List[torch.FloatTensor]:
    """Matches training/train.py encode_prompt exactly."""
    if isinstance(prompt, str):
        prompt = [prompt]

    for i, prompt_item in enumerate(prompt):
        messages = [{"role": "user", "content": prompt_item}]
        prompt_item = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids, attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []
    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Matches training/train.py calculate_shift exactly."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def sample_student_timestep(batch_size, global_step, student_timesteps, warmup_steps=500,
                            device="cpu", generator=None):
    """Matches training/train.py sample_student_timestep exactly."""
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


def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    """Matches training/train.py custom_mse_loss exactly."""
    noise_pred = noise_pred.float()
    target = target.float()
    diff = noise_pred - target
    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    return final_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for train.py")
    parser.add_argument("--dummy", action="store_true",
                        help="Use tiny randomly-initialized models (no weights needed).")
    parser.add_argument("--image_sample_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["fp32", "bf16", "fp16"])
    return parser.parse_args()


def select_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = select_device(args.device)
    if args.dtype:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if not args.dummy:
        print("ERROR: Only --dummy mode is supported for this smoke test.")
        sys.exit(1)

    print(f"Device: {device}, dtype: {dtype}")
    results = []

    # -------------------------------------------------------------------
    # 1. Create models (same tiny architecture as smoke_test.py)
    # -------------------------------------------------------------------
    print_header("Step 1: Create models")

    feature_dim = TINY_DIM
    n_layers = TINY_N_LAYERS
    disc_layer_indices = tuple(range(n_layers))

    student = create_tiny_transformer(dtype).to(device)
    student.train()
    student.requires_grad_(True)
    print_pass(f"Student: {sum(p.numel() for p in student.parameters())/1e6:.2f}M params")

    teacher = create_tiny_transformer(dtype).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    print_pass(f"Teacher (frozen): {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M params")

    discriminator = LADDDiscriminator(
        feature_dim=feature_dim,
        hidden_dim=256,
        cond_dim=256,
        layer_indices=disc_layer_indices,
    ).to(device).to(dtype)
    discriminator.train()
    print_pass(f"Discriminator: {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M params")

    text_encoder = DummyTextEncoder().to(device).to(dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    tokenizer = DummyTokenizer()
    print_pass("Dummy text encoder + tokenizer created")

    results.append(("Model creation", True))

    # -------------------------------------------------------------------
    # 2. train.py encode_prompt
    # -------------------------------------------------------------------
    print_header("Step 2: train.py encode_prompt")

    test_prompts = ["A beautiful sunset over the ocean"]
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            test_prompts, device=device,
            text_encoder=text_encoder, tokenizer=tokenizer,
            max_sequence_length=512,
        )

    assert isinstance(prompt_embeds, list), "encode_prompt should return a list"
    assert len(prompt_embeds) == 1, f"Expected 1 embedding, got {len(prompt_embeds)}"
    print_pass(f"encode_prompt output: {len(prompt_embeds)} embeddings, shape {prompt_embeds[0].shape}")

    results.append(("train.py encode_prompt", True))

    # -------------------------------------------------------------------
    # 3. train.py calculate_shift
    # -------------------------------------------------------------------
    print_header("Step 3: train.py calculate_shift")

    vae_scale = 16
    height_latent = 2 * (args.image_sample_size // vae_scale)
    width_latent = 2 * (args.image_sample_size // vae_scale)
    in_channels = DEFAULT_TRANSFORMER_IN_CHANNELS
    H_tokens = height_latent // 2
    W_tokens = width_latent // 2
    image_seq_len = H_tokens * W_tokens

    mu = calculate_shift(image_seq_len)
    assert isinstance(mu, float), f"calculate_shift should return float, got {type(mu)}"
    print_pass(f"calculate_shift(image_seq_len={image_seq_len}) = {mu:.4f}")

    results.append(("calculate_shift", True))

    # -------------------------------------------------------------------
    # 4. train.py sample_student_timestep
    # -------------------------------------------------------------------
    print_header("Step 4: sample_student_timestep")

    student_timesteps = [1.0, 0.75, 0.5, 0.25]
    bsz = 2

    # Warmup regime (global_step < warmup_steps)
    t_warmup = sample_student_timestep(
        bsz, global_step=0, student_timesteps=student_timesteps,
        warmup_steps=500, device=device,
    )
    assert t_warmup.shape == (bsz,), f"Expected shape ({bsz},), got {t_warmup.shape}"
    for t_val in t_warmup:
        assert t_val.item() in student_timesteps, f"Sampled {t_val.item()} not in {student_timesteps}"
    print_pass(f"Warmup timesteps: {t_warmup.tolist()}")

    # Post-warmup regime
    t_post = sample_student_timestep(
        bsz, global_step=1000, student_timesteps=student_timesteps,
        warmup_steps=500, device=device,
    )
    assert t_post.shape == (bsz,)
    for t_val in t_post:
        assert t_val.item() in student_timesteps
    print_pass(f"Post-warmup timesteps: {t_post.tolist()}")

    results.append(("sample_student_timestep", True))

    # -------------------------------------------------------------------
    # 5. Baseline MSE training path (non-LADD)
    # -------------------------------------------------------------------
    print_header("Step 5: Baseline MSE training (train.py main path)")

    bsz = 1
    noise = torch.randn(bsz, in_channels, height_latent, width_latent, device=device, dtype=dtype)
    latent = torch.randn_like(noise)
    sigma = torch.tensor([0.5], device=device, dtype=dtype)
    noisy_latent = (1.0 - sigma.view(-1, 1, 1, 1)) * latent + sigma.view(-1, 1, 1, 1) * noise
    target = noise - latent  # velocity target

    timesteps = torch.tensor([0.5], device=device, dtype=dtype)

    # train.py passes inputs through the transformer the same way
    student.zero_grad()
    noisy_input = noisy_latent.unsqueeze(2)  # add frame dim
    input_list = list(noisy_input.unbind(dim=0))
    student_out, _ = student(input_list, timesteps, prompt_embeds, return_hidden_states=False)
    noise_pred = torch.stack(student_out, dim=0).squeeze(2)

    # train.py negates noise_pred: loss = custom_mse_loss(-noise_pred.float(), target.float(), ...)
    loss = custom_mse_loss(-noise_pred.float(), target.float())
    assert torch.isfinite(loss), f"Baseline MSE loss not finite: {loss}"
    print_pass(f"Baseline MSE loss = {loss.item():.4f}")

    # Create optimizer before backward so it tracks the params
    baseline_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    baseline_optimizer.zero_grad()

    loss.backward()
    student_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in student.parameters() if p.requires_grad
    )
    assert student_has_grad, "Student has no gradients in baseline mode"
    print_pass("Student has gradients after baseline backward")

    # Optimizer step — pick a param that has a non-zero gradient
    param_with_grad = None
    for p in student.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            param_with_grad = p
            break
    assert param_with_grad is not None, "No param with grad found"
    param_before = param_with_grad.data.clone()
    baseline_optimizer.step()
    param_after = param_with_grad.data
    assert not torch.equal(param_before, param_after), "Student weights didn't change"
    print_pass("Student weights updated after optimizer step")
    baseline_optimizer.zero_grad()

    results.append(("Baseline MSE training", True))

    # -------------------------------------------------------------------
    # 6. LADD path (train.py --use_ladd / main_ladd)
    # -------------------------------------------------------------------
    print_header("Step 6: LADD training path (train.py --use_ladd)")

    bsz = 1
    student_t = sample_student_timestep(
        bsz, global_step=1000, student_timesteps=student_timesteps,
        warmup_steps=500, device=device,
    )

    noise = torch.randn(bsz, in_channels, height_latent, width_latent, device=device, dtype=dtype)
    student_input_list = list(noise.unsqueeze(2).unbind(dim=0))

    # Student forward
    student_out, _ = student(student_input_list, student_t, prompt_embeds, return_hidden_states=False)
    student_pred = torch.stack(student_out, dim=0).squeeze(2)
    print_pass(f"Student output shape: {student_pred.shape}")

    # Re-noise for discrimination (same as train.py main_ladd)
    t_hat = logit_normal_sample(bsz, m=1.0, s=1.0, device=device)
    renoise = torch.randn_like(student_pred)
    student_renoised = add_noise(student_pred.detach().float(), renoise.float(), t_hat.float()).to(dtype)

    real_noise = torch.randn_like(student_pred)
    real_noise_2 = torch.randn_like(student_pred)
    real_noisy = add_noise(real_noise.float(), real_noise_2.float(), t_hat.float()).to(dtype)

    # Teacher forward (no_grad, same as train.py)
    with torch.no_grad():
        fake_input_list = list(student_renoised.unsqueeze(2).unbind(dim=0))
        _, fake_extras = teacher(fake_input_list, t_hat, prompt_embeds, return_hidden_states=True)

        real_input_list = list(real_noisy.unsqueeze(2).unbind(dim=0))
        _, real_extras = teacher(real_input_list, t_hat, prompt_embeds, return_hidden_states=True)

    print_pass(f"Teacher hidden states: {len(fake_extras['hidden_states'])} layers")

    # Discriminator
    spatial_sizes = [(H_tokens, W_tokens)] * bsz
    fake_result = discriminator(
        fake_extras["hidden_states"], fake_extras["x_item_seqlens"],
        fake_extras["cap_item_seqlens"], spatial_sizes, t_hat,
    )
    real_result = discriminator(
        real_extras["hidden_states"], real_extras["x_item_seqlens"],
        real_extras["cap_item_seqlens"], spatial_sizes, t_hat,
    )

    d_loss, g_loss = LADDDiscriminator.compute_loss(
        real_result["total_logit"], fake_result["total_logit"],
    )
    assert torch.isfinite(d_loss), f"d_loss not finite: {d_loss}"
    assert torch.isfinite(g_loss), f"g_loss not finite: {g_loss}"
    print_pass(f"d_loss = {d_loss.item():.4f}, g_loss = {g_loss.item():.4f}")

    results.append(("LADD forward + loss", True))

    # -------------------------------------------------------------------
    # 7. LADD gradient flow — disc update
    # -------------------------------------------------------------------
    print_header("Step 7: LADD discriminator gradient flow")

    discriminator.zero_grad()
    d_loss.backward(retain_graph=True)

    disc_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in discriminator.parameters() if p.requires_grad
    )
    assert disc_has_grad, "Discriminator has no gradients after d_loss backward"
    print_pass("Discriminator has non-zero gradients")

    teacher_has_grad = any(p.grad is not None for p in teacher.parameters())
    assert not teacher_has_grad, "Teacher should not have gradients"
    print_pass("Teacher frozen — no gradients")

    # Disc optimizer step
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
    disc_param_before = next(discriminator.parameters()).data.clone()
    disc_optimizer.step()
    disc_param_after = next(discriminator.parameters()).data
    assert not torch.equal(disc_param_before, disc_param_after), "Disc weights didn't change"
    print_pass("Discriminator weights updated")
    disc_optimizer.zero_grad()
    discriminator.zero_grad()

    results.append(("LADD disc gradient flow", True))

    # -------------------------------------------------------------------
    # 8. LADD gradient flow — student update through teacher
    # -------------------------------------------------------------------
    print_header("Step 8: LADD student gradient flow through teacher")

    student.zero_grad()

    # Fresh forward: student -> add_noise -> teacher (no torch.no_grad!) -> disc
    # This mirrors train.py main_ladd gen-step path
    student_out2, _ = student(student_input_list, student_t, prompt_embeds, return_hidden_states=False)
    student_pred2 = torch.stack(student_out2, dim=0).squeeze(2)
    student_renoised_grad = add_noise(student_pred2.float(), renoise.float(), t_hat.float()).to(dtype)
    fake_input_grad = list(student_renoised_grad.unsqueeze(2).unbind(dim=0))

    # No torch.no_grad — graph flows through teacher to student
    _, fake_extras_grad = teacher(fake_input_grad, t_hat, prompt_embeds, return_hidden_states=True)
    fake_result_grad = discriminator(
        fake_extras_grad["hidden_states"], fake_extras_grad["x_item_seqlens"],
        fake_extras_grad["cap_item_seqlens"], spatial_sizes, t_hat,
    )
    g_loss_update = -torch.mean(fake_result_grad["total_logit"])
    g_loss_update.backward()

    student_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in student.parameters() if p.requires_grad
    )
    assert student_has_grad, "Student has no gradients — gradient flow through teacher is broken"
    print_pass("Student receives gradients through teacher")

    teacher_has_param_grad = any(p.grad is not None for p in teacher.parameters())
    assert not teacher_has_param_grad, "Teacher params should not accumulate .grad"
    print_pass("Teacher parameters remain gradient-free")

    # Student optimizer step — pick a param with non-zero grad
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    s_param_with_grad = None
    for p in student.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            s_param_with_grad = p
            break
    assert s_param_with_grad is not None, "No student param with grad"
    student_param_before = s_param_with_grad.data.clone()
    student_optimizer.step()
    student_param_after = s_param_with_grad.data
    assert not torch.equal(student_param_before, student_param_after), "Student weights didn't change"
    print_pass("Student weights updated after gen step")

    results.append(("LADD student gradient flow", True))

    # -------------------------------------------------------------------
    # 9. Checkpoint save/load round-trip
    # -------------------------------------------------------------------
    print_header("Step 9: Checkpoint save/load round-trip")

    tmpdir = tempfile.mkdtemp(prefix="train_smoke_", dir=_project_root)
    try:
        # Save student
        student_path = os.path.join(tmpdir, "student.pt")
        torch.save(student.state_dict(), student_path)
        print_pass(f"Student saved ({os.path.getsize(student_path)/1e6:.1f} MB)")

        # Save discriminator
        disc_path = os.path.join(tmpdir, "discriminator.pt")
        torch.save(discriminator.state_dict(), disc_path)
        print_pass(f"Discriminator saved ({os.path.getsize(disc_path)/1e6:.1f} MB)")

        # Reload and verify student
        student_loaded = create_tiny_transformer(dtype).to(device)
        student_loaded.load_state_dict(torch.load(student_path, map_location=device, weights_only=True))
        for (n1, p1), (n2, p2) in zip(student.named_parameters(), student_loaded.named_parameters()):
            assert torch.equal(p1.data, p2.data), f"Student mismatch in {n1}"
        print_pass("Student checkpoint round-trip: parameters match")

        # Reload and verify discriminator
        disc_loaded = LADDDiscriminator(
            feature_dim=feature_dim, hidden_dim=256, cond_dim=256,
            layer_indices=disc_layer_indices,
        ).to(device).to(dtype)
        disc_loaded.load_state_dict(torch.load(disc_path, map_location=device, weights_only=True))
        for (n1, p1), (n2, p2) in zip(discriminator.named_parameters(), disc_loaded.named_parameters()):
            assert torch.equal(p1.data, p2.data), f"Disc mismatch in {n1}"
        print_pass("Discriminator checkpoint round-trip: parameters match")

        results.append(("Checkpoint round-trip", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -------------------------------------------------------------------
    # 10. Alternating D/G update schedule
    # -------------------------------------------------------------------
    print_header("Step 10: Alternating D/G update schedule")

    gen_update_interval = 5
    d_steps = 0
    g_steps = 0
    for step in range(20):
        is_gen_step = (step % gen_update_interval == 0)
        d_steps += 1
        if is_gen_step:
            g_steps += 1

    assert d_steps == 20, f"Expected 20 disc steps, got {d_steps}"
    assert g_steps == 4, f"Expected 4 gen steps (every 5th), got {g_steps}"
    print_pass(f"20 steps -> {d_steps} disc updates, {g_steps} gen updates (interval={gen_update_interval})")

    results.append(("D/G update schedule", True))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print_header("SMOKE TEST SUMMARY (train.py)")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  All {len(results)} checks passed! train.py pipeline is ready.")
    else:
        print(f"\n  Some checks failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
