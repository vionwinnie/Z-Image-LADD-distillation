"""Precompute CLIP text embeddings for discriminator conditioning.

Usage:
    python scripts/precompute_clip_embeddings.py \
        --metadata data/train/metadata_latent_subset.json \
        --output_dir data/train/clip_embeddings

    # Multiple splits:
    python scripts/precompute_clip_embeddings.py \
        --metadata data/train/metadata_latent_subset.json data/val/metadata.json \
        --output_dir data/train/clip_embeddings data/val/clip_embeddings
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CLIP text embeddings.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="HuggingFace CLIP model name")
    parser.add_argument("--metadata", nargs="+", type=str, required=True)
    parser.add_argument("--output_dir", nargs="+", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.metadata) != len(args.output_dir):
        print(f"ERROR: {len(args.metadata)} metadata files but {len(args.output_dir)} output dirs")
        sys.exit(1)

    # Load CLIP
    print(f"Loading CLIP model: {args.clip_model}")
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(args.clip_model)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    model.to(args.device).eval()
    embed_dim = model.config.projection_dim
    print(f"CLIP embed dim: {embed_dim}")

    for meta_path, out_dir in zip(args.metadata, args.output_dir):
        print(f"\nProcessing {meta_path} -> {out_dir}")
        with open(meta_path) as f:
            data = json.load(f)
        texts = [item.get("text", item.get("prompt", "")) for item in data]
        print(f"  {len(texts)} prompts")

        os.makedirs(out_dir, exist_ok=True)

        all_embeddings = []
        t0 = time.time()

        for i in tqdm(range(0, len(texts), args.batch_size), desc="  Encoding"):
            batch_texts = texts[i:i + args.batch_size]
            # Truncate long texts for CLIP (77 token limit)
            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                              max_length=77, return_tensors="pt").to(args.device)
            with torch.no_grad():
                text_out = model.text_model(**inputs)
                pooled = text_out.pooler_output  # (B, hidden_size)
                text_features = model.text_projection(pooled)  # (B, embed_dim)
                # L2 normalize (standard CLIP practice)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)  # (N, embed_dim)
        elapsed = time.time() - t0
        print(f"  Encoded {len(embeddings)} prompts in {elapsed:.1f}s")
        print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        # Save
        out_path = os.path.join(out_dir, "clip_embeddings.pt")
        torch.save({
            "embeddings": embeddings,
            "embed_dim": embed_dim,
            "num_prompts": len(embeddings),
            "clip_model": args.clip_model,
        }, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
