#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "mlx_sam3"))

from sam3 import build_sam3_image_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLX SAM3 text-encoder reference tensors.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "golden" / "text" / "object")
    parser.add_argument("--dump-blocks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        checkpoint_path=str(args.weights),
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )

    tokenizer = model.backbone.language_backbone.tokenizer
    text_encoder = model.backbone.language_backbone
    tokenized = tokenizer([args.prompt], context_length=text_encoder.context_length)
    _, text_memory_resized, _ = text_encoder([args.prompt])

    np.savetxt(args.output_prefix.with_suffix(".tokens.txt"), np.array(tokenized[0]), fmt="%d")
    np.save(args.output_prefix.with_suffix(".memory.npy"), np.array(text_memory_resized, dtype=np.float32))

    if args.dump_blocks:
        x = text_encoder.encoder.token_embedding(tokenized)
        x = x + text_encoder.encoder.positional_embedding[: tokenized.shape[1]]
        np.save(args.output_prefix.with_name(args.output_prefix.name + ".input.npy"), np.array(x.transpose(1, 0, 2), dtype=np.float32))
        for layer_idx, block in enumerate(text_encoder.encoder.transformer.resblocks):
            x = block(x, attn_mask=text_encoder.encoder.attn_mask[: tokenized.shape[1], : tokenized.shape[1]])
            np.save(
                args.output_prefix.with_name(f"{args.output_prefix.name}.block_{layer_idx:02d}.npy"),
                np.array(x.transpose(1, 0, 2), dtype=np.float32),
            )
        x = text_encoder.encoder.ln_final(x)
        np.save(args.output_prefix.with_name(args.output_prefix.name + ".ln_final.npy"), np.array(x.transpose(1, 0, 2), dtype=np.float32))

    print(f"prompt={args.prompt!r}")
    print(f"tokens={args.output_prefix.with_suffix('.tokens.txt')}")
    print(f"memory={args.output_prefix.with_suffix('.memory.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
