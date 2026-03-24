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


def transform(image: Image.Image, resolution: int) -> mx.array:
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    image_arr = (image_arr - 0.5) / 0.5
    return mx.array(image_arr.transpose(2, 0, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLX SAM3 vision neck reference tensors.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "golden" / "neck" / "sample")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        checkpoint_path=str(args.weights),
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )

    image = Image.open(args.image).convert("RGB")
    image_arr = transform(image, resolution=1008)[None]
    trunk_out = model.backbone.vision_backbone.trunk(image_arr)[-1]
    x = trunk_out.transpose(0, 2, 3, 1)

    np.save(args.output_prefix.with_name(args.output_prefix.name + ".trunk.npy"), np.array(trunk_out, dtype=np.float32))

    sam3_out = []
    for i, conv in enumerate(model.backbone.vision_backbone.convs):
        y = conv(x)
        sam3_out.append(y.transpose(0, 3, 1, 2))
        np.save(
            args.output_prefix.with_name(f"{args.output_prefix.name}.level_{i}.npy"),
            np.array(sam3_out[-1], dtype=np.float32),
        )
        pos = model.backbone.vision_backbone.position_encoding(sam3_out[-1].shape)
        np.save(
            args.output_prefix.with_name(f"{args.output_prefix.name}.pos_{i}.npy"),
            np.array(pos, dtype=np.float32),
        )

    print(f"image={args.image}")
    print(f"trunk={args.output_prefix.with_name(args.output_prefix.name + '.trunk.npy')}")
    for i in range(len(sam3_out)):
        print(f"level_{i}={args.output_prefix.with_name(f'{args.output_prefix.name}.level_{i}.npy')}")
        print(f"pos_{i}={args.output_prefix.with_name(f'{args.output_prefix.name}.pos_{i}.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
