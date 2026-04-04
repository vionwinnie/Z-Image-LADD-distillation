"""Precompute text encoder embeddings and save as Arrow dataset.

Encodes all prompts through the text encoder once, saving bf16 embeddings
for fast loading during training (avoids keeping the ~3GB text encoder on GPU).

Usage:
    python scripts/precompute_embeddings.py \
        --model_path models/Z-Image \
        --metadata data/debug/metadata.json \
        --output_dir data/debug/embeddings

    # Multiple splits:
    python scripts/precompute_embeddings.py \
        --model_path models/Z-Image \
        --metadata data/debug/metadata.json data/val/metadata.json \
        --output_dir data/debug/embeddings data/val/embeddings
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))
sys.path.insert(0, _project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute text encoder embeddings.")
    parser.add_argument("--model_path", type=str, default="models/Z-Image")
    parser.add_argument("--metadata", nargs="+", type=str, required=True,
                        help="Path(s) to metadata JSON files.")
    parser.add_argument("--output_dir", nargs="+", type=str, required=True,
                        help="Output dir(s) for embeddings (one per metadata file).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def encode_batch(texts, tokenizer, text_encoder, device, max_seq_len):
    """Encode a batch of texts, return list of variable-length bf16 tensors."""
    formatted = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        formatted.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        ))

    inputs = tokenizer(
        formatted, padding="max_length", max_length=max_seq_len,
        truncation=True, return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    masks = inputs.attention_mask.to(device).bool()

    with torch.no_grad():
        hidden = text_encoder(
            input_ids=input_ids, attention_mask=masks, output_hidden_states=True,
        ).hidden_states[-2]

    # Extract variable-length embeddings (only non-padded tokens)
    embeddings = []
    for i in range(len(texts)):
        embeddings.append(hidden[i][masks[i]].cpu())

    return embeddings


def main():
    args = parse_args()

    if len(args.metadata) != len(args.output_dir):
        print(f"ERROR: {len(args.metadata)} metadata files but {len(args.output_dir)} output dirs")
        sys.exit(1)

    # Load text encoder + tokenizer
    print("Loading text encoder...")
    from transformers import AutoModel, AutoTokenizer
    text_encoder_dir = os.path.join(args.model_path, "text_encoder")
    tokenizer_dir = os.path.join(args.model_path, "tokenizer")
    text_encoder = AutoModel.from_pretrained(
        text_encoder_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    text_encoder.to(args.device).eval()
    tokenizer = (AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
                 if os.path.exists(tokenizer_dir) else
                 AutoTokenizer.from_pretrained(text_encoder_dir, trust_remote_code=True))
    print(f"Text encoder loaded on {args.device}")

    for meta_path, out_dir in zip(args.metadata, args.output_dir):
        print(f"\nProcessing {meta_path} -> {out_dir}")
        with open(meta_path) as f:
            data = json.load(f)
        texts = [item.get("text", item.get("prompt", "")) for item in data]
        print(f"  {len(texts)} prompts")

        os.makedirs(out_dir, exist_ok=True)

        # Encode in batches
        all_embeddings = []
        all_seq_lens = []
        t0 = time.time()

        for start in tqdm(range(0, len(texts), args.batch_size), desc="  Encoding"):
            batch_texts = texts[start:start + args.batch_size]
            embeddings = encode_batch(
                batch_texts, tokenizer, text_encoder,
                args.device, args.max_sequence_length,
            )
            for emb in embeddings:
                all_seq_lens.append(emb.shape[0])
                all_embeddings.append(emb)

        elapsed = time.time() - t0
        print(f"  Encoded {len(all_embeddings)} prompts in {elapsed:.1f}s "
              f"({elapsed / len(all_embeddings) * 1000:.1f} ms/prompt)")
        print(f"  Seq lens: min={min(all_seq_lens)}, max={max(all_seq_lens)}, "
              f"mean={sum(all_seq_lens)/len(all_seq_lens):.0f}")

        # Save as a single .pt file with list of tensors + metadata
        # Format: {"embeddings": [tensor, ...], "seq_lens": [int, ...], "dtype": "bfloat16"}
        save_path = os.path.join(out_dir, "embeddings.pt")
        torch.save({
            "embeddings": all_embeddings,
            "seq_lens": all_seq_lens,
            "embed_dim": all_embeddings[0].shape[-1],
            "dtype": str(all_embeddings[0].dtype),
            "num_prompts": len(all_embeddings),
            "source": meta_path,
        }, save_path)

        file_size = os.path.getsize(save_path) / 1e6
        print(f"  Saved to {save_path} ({file_size:.1f} MB)")

        # Also save an empty-prompt embedding for CFG (text_drop_ratio)
        empty_embeddings = encode_batch(
            [""], tokenizer, text_encoder,
            args.device, args.max_sequence_length,
        )
        torch.save(empty_embeddings[0], os.path.join(out_dir, "empty_embedding.pt"))
        print(f"  Saved empty embedding ({empty_embeddings[0].shape})")


if __name__ == "__main__":
    main()
