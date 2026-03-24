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
    parser = argparse.ArgumentParser(description="Export MLX SAM3 backbone.call_image() reference tensors.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "golden" / "call_image" / "sample")
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
    backbone_out = model.backbone.call_image(image_arr)

    np.save(
        args.output_prefix.with_name(args.output_prefix.name + ".vision_features.npy"),
        np.array(backbone_out["vision_features"], dtype=np.float32),
    )

    backbone_fpn = backbone_out["backbone_fpn"]
    vision_pos_enc = backbone_out["vision_pos_enc"]
    for i, feat in enumerate(backbone_fpn):
        np.save(
            args.output_prefix.with_name(f"{args.output_prefix.name}.fpn_{i}.npy"),
            np.array(feat, dtype=np.float32),
        )
        np.save(
            args.output_prefix.with_name(f"{args.output_prefix.name}.pos_{i}.npy"),
            np.array(vision_pos_enc[i], dtype=np.float32),
        )

    print(f"image={args.image}")
    print(f"vision_features={args.output_prefix.with_name(args.output_prefix.name + '.vision_features.npy')}")
    for i in range(len(backbone_fpn)):
        print(f"fpn_{i}={args.output_prefix.with_name(f'{args.output_prefix.name}.fpn_{i}.npy')}")
        print(f"pos_{i}={args.output_prefix.with_name(f'{args.output_prefix.name}.pos_{i}.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
