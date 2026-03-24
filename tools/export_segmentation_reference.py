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
    parser = argparse.ArgumentParser(description="Export MLX SAM3 segmentation head reference tensors.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "golden" / "segmentation" / "sample")
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
    backbone_out, encoder_out, _ = model._run_encoder(
        backbone_out,
        find_stage,
        prompt,
        prompt_mask,
    )

    bs = encoder_out["encoder_hidden_states"].shape[1]
    query_embed = model.transformer.decoder.query_embed.weight
    tgt = mx.tile(query_embed[:, None], (1, bs, 1))
    hs, _, _, _ = model.transformer.decoder(
        tgt=tgt,
        memory=encoder_out["encoder_hidden_states"],
        memory_key_padding_mask=encoder_out["padding_mask"],
        pos=encoder_out["pos_embed"],
        reference_boxes=None,
        level_start_index=encoder_out["level_start_index"],
        spatial_shapes=encoder_out["spatial_shapes"],
        valid_ratios=encoder_out["valid_ratios"],
        tgt_mask=None,
        memory_text=prompt,
        text_attention_mask=prompt_mask,
        apply_dac=False,
    )

    seg_head = model.segmentation_head
    enc_attended = encoder_out["encoder_hidden_states"]
    if seg_head.cross_attend_prompt is not None:
        t_encoder_hidden_states = enc_attended.transpose(1, 0, 2)
        t_prompt = prompt.transpose(1, 0, 2)
        tgt2 = seg_head.cross_attn_norm(t_encoder_hidden_states)
        tgt2 = seg_head.cross_attend_prompt(
            queries=tgt2,
            keys=t_prompt,
            values=t_prompt,
            key_padding_mask=prompt_mask,
        ).transpose(1, 0, 2)
        enc_attended = tgt2 + enc_attended

    encoder_feature_map = enc_attended.transpose(1, 2, 0)[..., : backbone_out["backbone_fpn"][-1].shape[-2] * backbone_out["backbone_fpn"][-1].shape[-1]].reshape(
        -1, *backbone_out["backbone_fpn"][-1].shape[1:]
    )
    backbone_feats = [x.transpose(0, 2, 3, 1) for x in backbone_out["backbone_fpn"]]
    prev_fpn = encoder_feature_map.transpose(0, 2, 3, 1)
    curr_fpn = backbone_feats[1]
    upsample0 = mx.repeat(mx.repeat(prev_fpn, 2, axis=1), 2, axis=2)
    stage0_input = curr_fpn + upsample0
    stage0_conv = seg_head.pixel_decoder.conv_layers[0](stage0_input)
    stage0_output = mx.maximum(seg_head.pixel_decoder.norms[0](stage0_conv), 0)
    curr_fpn = backbone_feats[0]
    upsample1 = mx.repeat(mx.repeat(stage0_output, 2, axis=1), 2, axis=2)
    stage1_input = curr_fpn + upsample1
    stage1_conv = seg_head.pixel_decoder.conv_layers[1](stage1_input)
    pixel_embed = mx.maximum(seg_head.pixel_decoder.norms[1](stage1_conv), 0).transpose(0, 3, 1, 2)
    instance_embed = seg_head.instance_seg_head(pixel_embed.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    semantic_seg = seg_head.semantic_seg_head(pixel_embed.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    hs_batch = hs.transpose(0, 2, 1, 3)
    pred_masks = seg_head.mask_predictor(hs_batch[-1], instance_embed)

    for i, feat in enumerate(backbone_out["backbone_fpn"]):
        np.save(args.output_prefix.with_name(f"{args.output_prefix.name}.fpn_{i}.npy"), np.ascontiguousarray(np.array(feat, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".memory.npy"), np.ascontiguousarray(np.array(encoder_out["encoder_hidden_states"], dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".prompt.npy"), np.ascontiguousarray(np.array(prompt, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".prompt_mask.npy"), np.ascontiguousarray(np.array(prompt_mask, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".hs_05.npy"), np.ascontiguousarray(np.array(hs[-1], dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".encoder_hidden_states.npy"), np.ascontiguousarray(np.array(enc_attended, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".encoder_feature_map.npy"), np.ascontiguousarray(np.array(encoder_feature_map, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".stage0_conv.npy"), np.ascontiguousarray(np.array(stage0_conv.transpose(0, 3, 1, 2), dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".stage0_output.npy"), np.ascontiguousarray(np.array(stage0_output.transpose(0, 3, 1, 2), dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".stage1_conv.npy"), np.ascontiguousarray(np.array(stage1_conv.transpose(0, 3, 1, 2), dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".pixel_embed.npy"), np.ascontiguousarray(np.array(pixel_embed, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".instance_embed.npy"), np.ascontiguousarray(np.array(instance_embed, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".semantic_seg.npy"), np.ascontiguousarray(np.array(semantic_seg, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".pred_masks.npy"), np.ascontiguousarray(np.array(pred_masks, dtype=np.float32)))

    print(f"image={args.image}")
    print(f"prompt={args.prompt!r}")
    print(f"output_prefix={args.output_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
