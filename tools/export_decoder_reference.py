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
    parser = argparse.ArgumentParser(description="Export MLX SAM3 decoder reference tensors.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "golden" / "decoder" / "sample")
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

    np.save(args.output_prefix.with_name(args.output_prefix.name + ".memory.npy"), np.ascontiguousarray(np.array(encoder_out["encoder_hidden_states"], dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".pos_embed.npy"), np.ascontiguousarray(np.array(encoder_out["pos_embed"], dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".prompt.npy"), np.ascontiguousarray(np.array(prompt, dtype=np.float32)))
    np.save(args.output_prefix.with_name(args.output_prefix.name + ".prompt_mask.npy"), np.ascontiguousarray(np.array(prompt_mask, dtype=np.float32)))

    bs = encoder_out["encoder_hidden_states"].shape[1]
    query_embed = model.transformer.decoder.query_embed.weight
    tgt = mx.tile(query_embed[:, None], (1, bs, 1))
    hs, reference_boxes, dec_presence_out, _ = model.transformer.decoder(
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

    hs = np.array(hs, dtype=np.float32)
    reference_boxes = np.array(reference_boxes, dtype=np.float32)
    dec_presence_out = np.array(dec_presence_out, dtype=np.float32)
    out = {}
    model._update_scores_and_boxes(
        out,
        mx.array(hs.transpose(0, 2, 1, 3)),
        mx.array(reference_boxes.transpose(0, 2, 1, 3)),
        prompt,
        prompt_mask,
        dec_presence_out=mx.array(dec_presence_out.transpose(0, 2, 1)),
    )
    for i in range(hs.shape[0]):
        np.save(args.output_prefix.with_name(f"{args.output_prefix.name}.hs_{i:02d}.npy"), np.ascontiguousarray(hs[i]))
        np.save(args.output_prefix.with_name(f"{args.output_prefix.name}.ref_{i:02d}.npy"), np.ascontiguousarray(reference_boxes[i]))
        np.save(args.output_prefix.with_name(f"{args.output_prefix.name}.presence_{i:02d}.npy"), np.ascontiguousarray(dec_presence_out[i]))
        np.save(args.output_prefix.with_name(f"{args.output_prefix.name}.pred_logits_{i:02d}.npy"), np.ascontiguousarray(np.array(out["aux_outputs"][i]["pred_logits"] if i + 1 < hs.shape[0] else out["pred_logits"], dtype=np.float32)))
        np.save(args.output_prefix.with_name(f"{args.output_prefix.name}.pred_boxes_{i:02d}.npy"), np.ascontiguousarray(np.array(out["aux_outputs"][i]["pred_boxes"] if i + 1 < hs.shape[0] else out["pred_boxes"], dtype=np.float32)))

    print(f"image={args.image}")
    print(f"prompt={args.prompt!r}")
    print(f"memory={args.output_prefix.with_name(args.output_prefix.name + '.memory.npy')}")
    print(f"pos_embed={args.output_prefix.with_name(args.output_prefix.name + '.pos_embed.npy')}")
    print(f"prompt_tensor={args.output_prefix.with_name(args.output_prefix.name + '.prompt.npy')}")
    print(f"prompt_mask={args.output_prefix.with_name(args.output_prefix.name + '.prompt_mask.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
