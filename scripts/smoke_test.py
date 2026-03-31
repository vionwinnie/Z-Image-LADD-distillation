"""End-to-end smoke test for LADD distillation pipeline.

Validates that all components (student, teacher, discriminator, losses,
gradients, checkpointing) work correctly before launching full training.

Usage:
    # With real weights:
    python scripts/smoke_test.py \
        --pretrained_model_name_or_path=models/Z-Image \
        --train_data_meta=data/debug/metadata.json

    # With dummy weights (no model download needed):
    python scripts/smoke_test.py --dummy
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
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
from zimage.autoencoder import AutoencoderKL
from zimage.scheduler import FlowMatchEulerDiscreteScheduler

from training.ladd_discriminator import LADDDiscriminator
from training.ladd_utils import (
    TextDataset,
    add_noise,
    logit_normal_sample,
)


# ---------------------------------------------------------------------------
# Dummy model helpers for --dummy mode
# ---------------------------------------------------------------------------

# Tiny transformer config: dim must satisfy dim // n_heads == sum(axes_dims) == 128
TINY_DIM = 128
TINY_N_HEADS = 1
TINY_N_KV_HEADS = 1
TINY_N_LAYERS = 4          # 4 layers instead of 30
TINY_N_REFINER_LAYERS = 1  # 1 instead of 2
TINY_CAP_FEAT_DIM = 64     # small text embedding dim


class DummyTextEncoder(nn.Module):
    """Minimal text encoder that returns embeddings of the right shape."""

    def __init__(self, vocab_size=1000, embed_dim=TINY_CAP_FEAT_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        embeds = self.embed(input_ids)

        class Output:
            pass

        out = Output()
        # hidden_states[-2] is what encode_prompt uses
        out.hidden_states = [embeds, embeds, embeds]
        return out


class DummyTokenizer:
    """Minimal tokenizer that produces token IDs and attention masks."""

    def __init__(self, vocab_size=1000, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        # Just return the raw text content
        return messages[0]["content"]

    def __call__(self, texts, padding="max_length", max_length=512, truncation=True, return_tensors="pt"):
        seq_len = 32  # SEQ_MULTI_OF — real token count
        batch_ids = []
        batch_masks = []
        for text in texts:
            # Simple hash-based tokenization: produce seq_len real tokens
            tokens = [(hash(text + str(i)) % (self.vocab_size - 1)) + 1 for i in range(seq_len)]
            mask = [1] * seq_len
            # Pad to max_length
            tokens += [self.pad_token_id] * (max_length - seq_len)
            mask += [0] * (max_length - seq_len)
            batch_ids.append(tokens[:max_length])
            batch_masks.append(mask[:max_length])

        class Output:
            pass

        out = Output()
        out.input_ids = torch.tensor(batch_ids)
        out.attention_mask = torch.tensor(batch_masks)
        return out


def create_tiny_transformer(dtype):
    """Create a small randomly-initialized transformer for testing."""
    model = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=DEFAULT_TRANSFORMER_IN_CHANNELS,
        dim=TINY_DIM,
        n_layers=TINY_N_LAYERS,
        n_refiner_layers=TINY_N_REFINER_LAYERS,
        n_heads=TINY_N_HEADS,
        n_kv_heads=TINY_N_KV_HEADS,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=TINY_CAP_FEAT_DIM,
        axes_dims=ROPE_AXES_DIMS,
        axes_lens=ROPE_AXES_LENS,
    ).to(dtype)
    return model


def encode_prompt_dummy(prompts, device, text_encoder, tokenizer, max_sequence_length=512):
    """Encode prompts through the (possibly dummy) text encoder."""
    if isinstance(prompts, str):
        prompts = [prompts]

    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        ))

    text_inputs = tokenizer(
        formatted, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=input_ids, attention_mask=masks, output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []
    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][masks[i]])

    return embeddings_list


# ---------------------------------------------------------------------------
# CLI and display
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for LADD pipeline.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                        help="Path to pretrained Z-Image model directory.")
    parser.add_argument("--train_data_meta", type=str, default="data/debug/metadata.json",
                        help="Path to JSON annotation file.")
    parser.add_argument("--image_sample_size", type=int, default=512,
                        help="Image resolution for latent generation.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (auto-detected if not set).")
    parser.add_argument("--dummy", action="store_true",
                        help="Use tiny randomly-initialized models (no weights needed).")
    args = parser.parse_args()
    if not args.dummy and args.pretrained_model_name_or_path is None:
        parser.error("--pretrained_model_name_or_path is required unless --dummy is set")
    return args


def select_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def print_pass(msg):
    print(f"  [PASS] {msg}")


def print_fail(msg):
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = select_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dummy = args.dummy

    print(f"Device: {device}, dtype: {dtype}, dummy: {dummy}")
    results = []

    # -----------------------------------------------------------------------
    # 1. Load / create models
    # -----------------------------------------------------------------------
    print_header("Step 1: Loading models")

    if dummy:
        feature_dim = TINY_DIM
        n_layers = TINY_N_LAYERS
        disc_layer_indices = tuple(range(n_layers))  # all 4 layers

        student = create_tiny_transformer(dtype).to(device)
        student.train()
        student.requires_grad_(True)
        print_pass(f"Student created (tiny): {sum(p.numel() for p in student.parameters())/1e6:.2f}M params")

        teacher = create_tiny_transformer(dtype).to(device)
        teacher.requires_grad_(False)
        teacher.eval()
        print_pass(f"Teacher created (tiny, frozen): {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M params")

        text_encoder = DummyTextEncoder().to(device).to(dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        tokenizer = DummyTokenizer()
        print_pass("Dummy text encoder + tokenizer created")
    else:
        from training.train_ladd import load_transformer
        from transformers import AutoModel, AutoTokenizer

        feature_dim = DEFAULT_TRANSFORMER_DIM
        n_layers = 30
        disc_layer_indices = (5, 10, 15, 20, 25, 29)

        student = load_transformer(args.pretrained_model_name_or_path, dtype).to(device)
        student.train()
        student.requires_grad_(True)
        print_pass(f"Student loaded: {sum(p.numel() for p in student.parameters())/1e9:.2f}B params")

        teacher = load_transformer(args.pretrained_model_name_or_path, dtype).to(device)
        teacher.requires_grad_(False)
        teacher.eval()
        print_pass(f"Teacher loaded (frozen): {sum(p.numel() for p in teacher.parameters())/1e9:.2f}B params")

        text_encoder_dir = os.path.join(args.pretrained_model_name_or_path, "text_encoder")
        tokenizer_dir = os.path.join(args.pretrained_model_name_or_path, "tokenizer")
        text_encoder = AutoModel.from_pretrained(text_encoder_dir, torch_dtype=dtype, trust_remote_code=True)
        text_encoder.to(device)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        if os.path.exists(tokenizer_dir):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True)
        print_pass("Text encoder + tokenizer loaded")

    discriminator = LADDDiscriminator(
        feature_dim=feature_dim,
        hidden_dim=256,
        cond_dim=256,
        layer_indices=disc_layer_indices,
    ).to(device).to(dtype)
    discriminator.train()
    print_pass(f"Discriminator: {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M params, "
               f"heads at layers {list(disc_layer_indices)}")

    results.append(("Model loading", True))

    # -----------------------------------------------------------------------
    # 2. Forward pass shapes
    # -----------------------------------------------------------------------
    print_header("Step 2: Forward pass shape validation")

    vae_scale = 16  # vae_scale_factor(8) * patch(2)
    height_latent = 2 * (args.image_sample_size // vae_scale)
    width_latent = 2 * (args.image_sample_size // vae_scale)
    in_channels = DEFAULT_TRANSFORMER_IN_CHANNELS
    bsz = 1

    # Encode a test prompt
    test_prompts = ["A beautiful sunset over the ocean"]
    with torch.no_grad():
        if dummy:
            prompt_embeds = encode_prompt_dummy(
                test_prompts, device=device, text_encoder=text_encoder,
                tokenizer=tokenizer, max_sequence_length=512,
            )
        else:
            from training.ladd_utils import encode_prompt
            prompt_embeds = encode_prompt(
                test_prompts, device=device, text_encoder=text_encoder,
                tokenizer=tokenizer, max_sequence_length=512,
            )

    print_pass(f"Prompt encoded: {len(prompt_embeds)} embeddings, shape {prompt_embeds[0].shape}")

    # Create noise input
    noise = torch.randn(bsz, in_channels, height_latent, width_latent, device=device, dtype=dtype)
    student_t = torch.tensor([1.0], device=device, dtype=dtype)
    input_list = list(noise.unsqueeze(2).unbind(dim=0))

    # Student forward (no hidden states)
    student_out, _ = student(input_list, student_t, prompt_embeds, return_hidden_states=False)
    student_pred = torch.stack(student_out, dim=0).squeeze(2)
    assert student_pred.shape == (bsz, in_channels, height_latent, width_latent), \
        f"Student output shape mismatch: {student_pred.shape}"
    print_pass(f"Student output shape: {student_pred.shape}")

    results.append(("Forward pass shapes", True))

    # -----------------------------------------------------------------------
    # 3. Teacher feature extraction
    # -----------------------------------------------------------------------
    print_header("Step 3: Teacher feature extraction (hidden states)")

    t_hat = torch.tensor([0.5], device=device, dtype=dtype)
    renoise = torch.randn_like(student_pred)
    student_renoised = add_noise(student_pred.detach().float(), renoise.float(), t_hat.float()).to(dtype)

    with torch.no_grad():
        fake_input = list(student_renoised.unsqueeze(2).unbind(dim=0))
        _, extras = teacher(fake_input, t_hat, prompt_embeds, return_hidden_states=True)

    hidden_states = extras["hidden_states"]
    x_seqlens = extras["x_item_seqlens"]
    cap_seqlens = extras["cap_item_seqlens"]

    assert len(hidden_states) == n_layers, f"Expected {n_layers} hidden states, got {len(hidden_states)}"
    print_pass(f"Got {len(hidden_states)} hidden states")
    print_pass(f"Hidden state shape: {hidden_states[0].shape}")
    print_pass(f"Image token counts: {x_seqlens}, text token counts: {cap_seqlens}")

    results.append(("Teacher feature extraction", True))

    # -----------------------------------------------------------------------
    # 4. Discriminator head outputs
    # -----------------------------------------------------------------------
    print_header("Step 4: Discriminator head outputs")

    H_tokens = height_latent // 2
    W_tokens = width_latent // 2
    spatial_sizes = [(H_tokens, W_tokens)] * bsz

    fake_result = discriminator(hidden_states, x_seqlens, cap_seqlens, spatial_sizes, t_hat)

    assert "logits" in fake_result, "Missing 'logits' key"
    assert "total_logit" in fake_result, "Missing 'total_logit' key"
    assert fake_result["total_logit"].shape == (bsz,), \
        f"Total logit shape mismatch: {fake_result['total_logit'].shape}"

    for layer_idx, logits in fake_result["logits"].items():
        assert logits.shape == (bsz,), f"Layer {layer_idx} logit shape: {logits.shape}"
        print_pass(f"Layer {layer_idx} logit: {logits.item():.4f}")

    print_pass(f"Total logit: {fake_result['total_logit'].item():.4f}")
    results.append(("Discriminator outputs", True))

    # -----------------------------------------------------------------------
    # 5. Loss computation
    # -----------------------------------------------------------------------
    print_header("Step 5: Loss computation (hinge loss)")

    # Create "real" path through teacher
    with torch.no_grad():
        real_noise = torch.randn_like(student_pred)
        real_noise2 = torch.randn_like(student_pred)
        real_noisy = add_noise(real_noise.float(), real_noise2.float(), t_hat.float()).to(dtype)
        real_input = list(real_noisy.unsqueeze(2).unbind(dim=0))
        _, real_extras = teacher(real_input, t_hat, prompt_embeds, return_hidden_states=True)

    real_result = discriminator(
        real_extras["hidden_states"], real_extras["x_item_seqlens"],
        real_extras["cap_item_seqlens"], spatial_sizes, t_hat,
    )

    d_loss, g_loss = LADDDiscriminator.compute_loss(
        real_result["total_logit"], fake_result["total_logit"],
    )

    assert torch.isfinite(d_loss), f"d_loss is not finite: {d_loss}"
    assert torch.isfinite(g_loss), f"g_loss is not finite: {g_loss}"
    assert d_loss.ndim == 0, f"d_loss should be scalar, got shape {d_loss.shape}"
    print_pass(f"d_loss = {d_loss.item():.4f} (finite, scalar)")
    print_pass(f"g_loss = {g_loss.item():.4f} (finite, scalar)")

    results.append(("Loss computation", True))

    # -----------------------------------------------------------------------
    # 6. Gradient flow check
    # -----------------------------------------------------------------------
    print_header("Step 6: Gradient flow validation")

    # Backward on d_loss
    d_loss.backward(retain_graph=True)

    # Discriminator should have gradients
    disc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in discriminator.parameters() if p.requires_grad)
    assert disc_has_grad, "Discriminator has no gradients after d_loss backward"
    print_pass("Discriminator has non-zero gradients")

    # Teacher should NOT have gradients
    teacher_has_grad = any(p.grad is not None for p in teacher.parameters())
    assert not teacher_has_grad, "Teacher should not have gradients"
    print_pass("Teacher has no gradients (frozen)")

    # Zero grads for clean g_loss test
    discriminator.zero_grad()
    student.zero_grad()

    # For generator gradient check: fresh forward with student grad path
    # Teacher runs WITHOUT torch.no_grad() so gradients flow through
    # the teacher's operations back to the student (per LADD paper).
    student_out2, _ = student(input_list, student_t, prompt_embeds, return_hidden_states=False)
    student_pred2 = torch.stack(student_out2, dim=0).squeeze(2)
    student_renoised2 = add_noise(student_pred2.float(), renoise.float(), t_hat.float()).to(dtype)
    fake_input2 = list(student_renoised2.unsqueeze(2).unbind(dim=0))
    # No torch.no_grad() here — gradients must flow through teacher to student
    _, fake_extras2 = teacher(fake_input2, t_hat, prompt_embeds, return_hidden_states=True)
    fake_result2 = discriminator(
        fake_extras2["hidden_states"], fake_extras2["x_item_seqlens"],
        fake_extras2["cap_item_seqlens"], spatial_sizes, t_hat,
    )
    g_loss2 = -torch.mean(fake_result2["total_logit"])
    g_loss2.backward()

    # Verify student receives gradients through teacher
    student_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in student.parameters() if p.requires_grad)
    assert student_has_grad, "Student has no gradients after g_loss backward — gradient flow through teacher is broken"
    print_pass("Student receives gradients through teacher (graph flows through frozen teacher)")

    # Teacher weights should still have no .grad (requires_grad=False)
    teacher_has_param_grad = any(p.grad is not None for p in teacher.parameters())
    assert not teacher_has_param_grad, "Teacher params should not accumulate .grad"
    print_pass("Teacher parameters have no .grad (frozen, but graph passes through)")

    disc_grad_norms = []
    for name, p in discriminator.named_parameters():
        if p.grad is not None:
            disc_grad_norms.append(p.grad.norm().item())
    if disc_grad_norms:
        print_pass(f"Discriminator grad norm (mean): {sum(disc_grad_norms)/len(disc_grad_norms):.6f}")

    student_grad_norms = []
    for name, p in student.named_parameters():
        if p.grad is not None:
            student_grad_norms.append(p.grad.norm().item())
    if student_grad_norms:
        print_pass(f"Student grad norm (mean): {sum(student_grad_norms)/len(student_grad_norms):.6f}")
    else:
        print_fail("Student has no gradient norms to report")

    results.append(("Gradient flow", True))

    # -----------------------------------------------------------------------
    # 7. Optimizer step
    # -----------------------------------------------------------------------
    print_header("Step 7: Optimizer step (weights change)")

    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)

    # Snapshot a param before step
    sample_param = next(discriminator.parameters())
    param_before = sample_param.data.clone()

    # Do a fresh forward/backward for disc
    discriminator.zero_grad()
    fake_result3 = discriminator(hidden_states, x_seqlens, cap_seqlens, spatial_sizes, t_hat)
    real_result3 = discriminator(
        real_extras["hidden_states"], real_extras["x_item_seqlens"],
        real_extras["cap_item_seqlens"], spatial_sizes, t_hat,
    )
    d_loss3, _ = LADDDiscriminator.compute_loss(real_result3["total_logit"], fake_result3["total_logit"])
    d_loss3.backward()
    disc_optimizer.step()

    param_after = sample_param.data
    weight_changed = not torch.equal(param_before, param_after)
    assert weight_changed, "Discriminator weights did not change after optimizer step"
    print_pass("Discriminator weights updated after optimizer step")

    results.append(("Optimizer step", True))

    # -----------------------------------------------------------------------
    # 8. Checkpoint save/load round-trip
    # -----------------------------------------------------------------------
    print_header("Step 8: Checkpoint save/load round-trip")

    tmpdir = tempfile.mkdtemp(prefix="ladd_smoke_", dir=_project_root)
    try:
        # Save discriminator
        disc_save_path = os.path.join(tmpdir, "discriminator.pt")
        torch.save(discriminator.state_dict(), disc_save_path)
        print_pass(f"Discriminator saved ({os.path.getsize(disc_save_path)/1e6:.1f} MB)")

        # Save student
        student_save_path = os.path.join(tmpdir, "student.pt")
        torch.save(student.state_dict(), student_save_path)
        print_pass(f"Student saved ({os.path.getsize(student_save_path)/1e6:.1f} MB)")

        # Reload discriminator
        disc_loaded = LADDDiscriminator(
            feature_dim=feature_dim,
            hidden_dim=256,
            cond_dim=256,
            layer_indices=disc_layer_indices,
        ).to(device).to(dtype)
        disc_loaded.load_state_dict(torch.load(disc_save_path, map_location=device, weights_only=True))

        # Verify parameter match
        for (n1, p1), (n2, p2) in zip(discriminator.named_parameters(), disc_loaded.named_parameters()):
            assert torch.equal(p1.data, p2.data), f"Mismatch in {n1}"
        print_pass("Discriminator checkpoint round-trip: parameters match")

        results.append(("Checkpoint round-trip", True))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 9. Baseline (non-LADD) forward/backward check
    # -----------------------------------------------------------------------
    print_header("Step 9: Baseline MSE training mode check")

    try:
        student.zero_grad()
        # Simulate baseline training: noise prediction with MSE loss
        baseline_noise = torch.randn(bsz, in_channels, height_latent, width_latent, device=device, dtype=dtype)
        baseline_latent = torch.randn_like(baseline_noise)
        sigma = torch.tensor([0.5], device=device, dtype=dtype)
        # Flow matching noisy input: x_t = (1-sigma)*x0 + sigma*noise
        noisy_latent = (1.0 - sigma.view(-1, 1, 1, 1)) * baseline_latent + sigma.view(-1, 1, 1, 1) * baseline_noise
        target = baseline_noise - baseline_latent  # velocity target

        t_baseline = torch.tensor([0.5], device=device, dtype=dtype)
        baseline_input = list(noisy_latent.unsqueeze(2).unbind(dim=0))
        baseline_out, _ = student(baseline_input, t_baseline, prompt_embeds, return_hidden_states=False)
        baseline_pred = torch.stack(baseline_out, dim=0).squeeze(2)

        # MSE loss (same as original train.py)
        mse_loss = F.mse_loss(baseline_pred.float(), target.float())
        assert torch.isfinite(mse_loss), f"MSE loss is not finite: {mse_loss}"
        mse_loss.backward()

        student_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in student.parameters() if p.requires_grad)
        assert student_has_grad, "Student has no gradients in baseline mode"
        print_pass(f"Baseline MSE loss = {mse_loss.item():.4f} (finite)")
        print_pass("Student has gradients in baseline mode")
        print_pass("Baseline training mode is functional")
        results.append(("Baseline MSE training", True))
    except Exception as e:
        print_fail(f"Baseline training check failed: {e}")
        results.append(("Baseline MSE training", False))
    finally:
        student.zero_grad()

    # -----------------------------------------------------------------------
    # 10. Memory usage report
    # -----------------------------------------------------------------------
    print_header("Step 10: Memory and summary")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print_pass(f"GPU memory allocated: {allocated:.2f} GB")
        print_pass(f"GPU memory reserved:  {reserved:.2f} GB")

    # Dataset check
    if os.path.exists(args.train_data_meta):
        dataset = TextDataset(args.train_data_meta)
        print_pass(f"Dataset loaded: {len(dataset)} prompts from {args.train_data_meta}")
    else:
        print(f"  [INFO] Dataset not found at {args.train_data_meta} (expected in non-dummy mode)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_header("SMOKE TEST SUMMARY")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  All {len(results)} checks passed! Pipeline is ready for training.")
    else:
        print(f"\n  Some checks failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
