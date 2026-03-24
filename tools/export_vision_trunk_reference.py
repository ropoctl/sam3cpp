#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "mlx_sam3"))

from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model.vitdet import get_abs_pos  # noqa: E402


def transform(image: Image.Image, resolution: int) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    image_arr = (image_arr - 0.5) / 0.5
    return image_arr.transpose(2, 0, 1)[None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLX SAM3 vision-trunk reference tensors.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "golden" / "vision_trunk" / "sample")
    parser.add_argument("--dump-layers", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        checkpoint_path=str(args.weights),
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    vt = model.backbone.vision_backbone.trunk

    image = Image.open(args.image).convert("RGB")
    image_arr = transform(image, resolution=1008)
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".image.npy"), image_arr.astype(np.float32))

    x = vt.patch_embed(mx.array(image_arr))
    h, w = x.shape[1], x.shape[2]
    x = x + get_abs_pos(
        vt.pos_embed,
        vt.pretrain_use_cls_token,
        (h, w),
        vt.retain_cls_token,
        tiling=vt.tile_abs_pos,
    )
    if args.dump_layers:
        np.save(
            args.output_prefix.with_name(args.output_prefix.name + ".stem.npy"),
            np.array(x.transpose(0, 3, 1, 2), dtype=np.float32),
        )

    x = vt.ln_pre(x)
    if args.dump_layers:
        np.save(
            args.output_prefix.with_name(args.output_prefix.name + ".ln_pre.npy"),
            np.array(x.transpose(0, 3, 1, 2), dtype=np.float32),
        )

    for i, block in enumerate(vt.blocks):
        x = block(x)
        if args.dump_layers:
            np.save(
                args.output_prefix.with_name(f"{args.output_prefix.name}.layer_{i:02d}.npy"),
                np.array(x.transpose(0, 3, 1, 2), dtype=np.float32),
            )

    trunk = np.array(x.transpose(0, 3, 1, 2), dtype=np.float32)
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".trunk.npy"), trunk)

    print(f"image={args.image}")
    print(f"image_tensor={args.output_prefix.with_name(args.output_prefix.name + '.image.npy')}")
    print(f"trunk={args.output_prefix.with_name(args.output_prefix.name + '.trunk.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
