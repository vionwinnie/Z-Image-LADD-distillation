# Debug Checklist

## Smoke Test Results

### Verified ✓
- [ ] Model loading: student, teacher, VAE, text encoder load without errors
- [ ] Forward pass: student produces correct output shape (B, C, H, W)
- [ ] Teacher feature extraction: return_hidden_states=True returns 30 hidden state tensors
- [ ] Discriminator heads: produce scalar logits per sample, conditioned on t_hat and text embeddings
- [ ] Hinge loss computation: d_loss and g_loss are finite, non-NaN
- [ ] Gradient flow: discriminator params have gradients; teacher params have no gradients (frozen)
- [ ] Optimizer step: student weights change after backward + step
- [ ] Checkpoint save/load: full Accelerate state + student weights round-trip
- [ ] Memory report: peak VRAM usage within A100 80GB budget

### Known Issues & Resolutions
1. **Checkpoint save fails on low-disk systems**: student model is ~12GB; /tmp may not have enough space. Fix: use --output_dir on a volume with sufficient space (e.g., /workspace on RunPod).
2. **Token ordering**: Z-Image concatenates image tokens first, text tokens second in the unified sequence. The discriminator's _extract_image_features() relies on this ordering via x_item_seqlens.
3. **Variable-length sequences**: Z-Image processes inputs as lists of variable-length tensors, not batched tensors. student_input_list is constructed via noise.unsqueeze(2).unbind(dim=0).

### Pre-Training Sanity Checks
- [ ] Verify d_loss and g_loss both start near ~1.0 (hinge loss equilibrium)
- [ ] Verify gradient norms are finite and not spiking
- [ ] Verify validation images are generated at validation_steps intervals
- [ ] Run 100 steps on debug split before launching full training
