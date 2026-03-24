#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "mlx_sam3"))

from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model.data_misc import FindStage  # noqa: E402


def transform(image: Image.Image, resolution: int) -> mx.array:
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    image_arr = (image_arr - 0.5) / 0.5
    return mx.array(image_arr.transpose(2, 0, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLX SAM3 dummy-prompt tensors.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "golden" / "dummy_prompt" / "sample")
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
    backbone_out.update(model.backbone.call_text([args.prompt]))

    find_stage = FindStage(
        img_ids=mx.array([0], dtype=mx.int64),
        text_ids=mx.array([0], dtype=mx.int64),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )

    feat_tuple = model._get_img_feats(backbone_out, find_stage.img_ids)
    _, img_feats, img_pos_embeds, _ = feat_tuple
    geo_feats, geo_masks = model.geometry_encoder(
        model._get_dummy_prompt(),
        img_feats,
        [x.shape[-2:] for x in backbone_out["vision_pos_enc"][-1:]],
        img_pos_embeds,
    )

    text_feats = backbone_out["language_features"][:, find_stage.text_ids]
    text_masks = backbone_out["language_mask"][find_stage.text_ids]
    prompt = mx.concat([text_feats, geo_feats], axis=0)
    prompt_mask = mx.concat([text_masks, geo_masks], axis=1)

    np.save(args.output_prefix.with_name(args.output_prefix.name + ".image.npy"), np.ascontiguousarray(np.array(backbone_out["vision_features"], dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".pos.npy"), np.ascontiguousarray(np.array(backbone_out["vision_pos_enc"][-1], dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".text.npy"), np.ascontiguousarray(np.array(text_feats, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".text_mask.npy"), np.ascontiguousarray(np.array(text_masks, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".geo_token.npy"), np.ascontiguousarray(np.array(geo_feats, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".prompt.npy"), np.ascontiguousarray(np.array(prompt, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".prompt_mask.npy"), np.ascontiguousarray(np.array(prompt_mask, dtype=np.float32)))

    print(f"image={args.image}")
    print(f"prompt={args.prompt!r}")
    print(f"output_prefix={args.output_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
