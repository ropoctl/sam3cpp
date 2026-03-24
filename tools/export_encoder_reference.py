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
    parser = argparse.ArgumentParser(description="Export MLX SAM3 encoder-fusion reference tensors.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "golden" / "encoder" / "sample")
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

    prompt, prompt_mask, backbone_out = model._encode_prompt(
        backbone_out,
        find_stage,
        model._get_dummy_prompt(),
    )

    np.save(
        args.output_prefix.with_name(args.output_prefix.name + ".image.npy"),
        np.array(backbone_out["vision_features"], dtype=np.float32),
    )
    np.save(
        args.output_prefix.with_name(args.output_prefix.name + ".pos.npy"),
        np.array(backbone_out["vision_pos_enc"][-1], dtype=np.float32),
    )
    np.save(
        args.output_prefix.with_name(args.output_prefix.name + ".prompt.npy"),
        np.array(prompt, dtype=np.float32),
    )
    np.save(
        args.output_prefix.with_name(args.output_prefix.name + ".prompt_mask.npy"),
        np.array(prompt_mask, dtype=np.float32),
    )

    feat_tuple = model._get_img_feats(backbone_out, find_stage.img_ids)
    _, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
    img_feats_for_layers = [mx.array(np.array(x, dtype=np.float32)) for x in img_feats]
    img_pos_for_layers = [mx.array(np.array(x, dtype=np.float32)) for x in img_pos_embeds]
    memory = model.transformer.encoder(
        src=img_feats,
        src_key_padding_mask=None,
        src_pos=img_pos_embeds,
        prompt=prompt,
        prompt_pos=mx.zeros_like(prompt),
        prompt_key_padding_mask=prompt_mask,
        feat_sizes=vis_feat_sizes,
        encoder_extra_kwargs=None,
    )

    np.save(
        args.output_prefix.with_name(args.output_prefix.name + ".memory.npy"),
        np.array(memory["memory"], dtype=np.float32),
    )
    np.save(
        args.output_prefix.with_name(args.output_prefix.name + ".pos_embed.npy"),
        np.array(memory["pos_embed"], dtype=np.float32),
    )

    if args.dump_layers:
        output = mx.array(np.array(img_feats_for_layers[0], dtype=np.float32).transpose(1, 0, 2))
        query_pos = mx.array(np.array(img_pos_for_layers[0], dtype=np.float32).transpose(1, 0, 2))
        prompt_bt = mx.array(np.array(prompt, dtype=np.float32).transpose(1, 0, 2))
        for layer_idx, layer in enumerate(model.transformer.encoder.layers):
            output = layer(
                tgt=output,
                memory=prompt_bt,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=prompt_mask,
                pos=None,
                query_pos=query_pos,
            )
            np.save(
                args.output_prefix.with_name(f"{args.output_prefix.name}.layer_{layer_idx:02d}.npy"),
                np.array(output, dtype=np.float32).transpose(1, 0, 2),
            )

    print(f"image={args.image}")
    print(f"prompt={args.prompt!r}")
    print(f"image_features={args.output_prefix.with_name(args.output_prefix.name + '.image.npy')}")
    print(f"pos={args.output_prefix.with_name(args.output_prefix.name + '.pos.npy')}")
    print(f"prompt_tensor={args.output_prefix.with_name(args.output_prefix.name + '.prompt.npy')}")
    print(f"prompt_mask={args.output_prefix.with_name(args.output_prefix.name + '.prompt_mask.npy')}")
    print(f"memory={args.output_prefix.with_name(args.output_prefix.name + '.memory.npy')}")
    print(f"pos_embed={args.output_prefix.with_name(args.output_prefix.name + '.pos_embed.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
