#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "mlx_sam3"))

from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model import box_ops  # noqa: E402
from sam3.model.data_misc import FindStage, interpolate  # noqa: E402


def transform_image(image: Image.Image, resolution: int) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[None]


def run_command(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def save_overlay(image: Image.Image, masks: np.ndarray, out_path: Path) -> None:
    base = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    overlay = base.copy()
    colors = [
        np.array([255, 0, 0, 110], dtype=np.uint8),
        np.array([0, 180, 255, 110], dtype=np.uint8),
        np.array([0, 220, 120, 110], dtype=np.uint8),
        np.array([255, 180, 0, 110], dtype=np.uint8),
    ]

    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        mask_bool = mask > 0.5
        overlay[mask_bool] = color

    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    blended = (overlay[..., :3].astype(np.float32) * alpha + base[..., :3].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    out = np.concatenate([blended, np.full((*blended.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
    Image.fromarray(out, mode="RGBA").save(out_path)


def compare_arrays(label: str, ref: np.ndarray, cand: np.ndarray) -> None:
    if ref.shape != cand.shape:
        print(f"{label}: shape mismatch reference={ref.shape} candidate={cand.shape}")
        return
    diff = ref.astype(np.float32) - cand.astype(np.float32)
    abs_diff = np.abs(diff)
    print(
        f"{label}: shape={ref.shape} max_abs={float(abs_diff.max(initial=0.0)):.9f} "
        f"mean_abs={float(abs_diff.mean() if abs_diff.size else 0.0):.9f} "
        f"rmse={float(np.sqrt(np.mean(diff ** 2)) if diff.size else 0.0):.9f}"
    )


def run_mlx_reference(model, image: Image.Image, prompt: str, width: int, height: int) -> dict[str, np.ndarray]:
    state = {
        "original_height": height,
        "original_width": width,
        "backbone_out": model.backbone.call_image(mx.array(transform_image(image, 1008))),
    }
    state["backbone_out"].update(model.backbone.call_text([prompt]))

    find_stage = FindStage(
        img_ids=mx.array([0], dtype=mx.int64),
        text_ids=mx.array([0], dtype=mx.int64),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )

    outputs = model.call_grounding(
        backbone_out=state["backbone_out"],
        find_input=find_stage,
        geometric_prompt=model._get_dummy_prompt(),
        find_target=None,
    )

    out_bbox = outputs["pred_boxes"]
    out_logits = outputs["pred_logits"]
    out_masks = outputs["pred_masks"]
    out_probs = mx.sigmoid(out_logits)
    presence_score = mx.sigmoid(outputs["presence_logit_dec"])[:, None]
    out_probs = (out_probs * presence_score).squeeze(-1)

    keep = out_probs > 0.5
    keep_idx = np.array(keep[0]).nonzero()[0]
    keep_idx_mx = mx.array(keep_idx)

    scores = np.array(out_probs[0][keep_idx_mx], dtype=np.float32)
    masks = out_masks[0][keep_idx_mx]
    boxes = out_bbox[0][keep_idx_mx]

    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    scale = mx.array([width, height, width, height])
    boxes = np.array(boxes * scale[None, :], dtype=np.float32)

    masks = interpolate(
        masks[:, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    masks = np.array(mx.sigmoid(masks), dtype=np.float32)

    return {
        "scores": scores,
        "boxes": boxes,
        "masks": masks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid SAM3 grounding using C++ backbone/text slices and MLX downstream.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--gguf", type=Path, default=ROOT / "models" / "sam3-image-f32.gguf")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "out" / "hybrid" / "sample")
    parser.add_argument("--compare-mlx", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Run C++ slices on CPU instead of preferred backend.")
    parser.add_argument("--cpp-image-backbone", action="store_true", help="Run the ViT trunk + neck in C++ from the preprocessed image tensor.")
    parser.add_argument("--cpp-prompt", action="store_true", help="Run the dummy geometry prompt encoder in C++ and append it to C++ text features.")
    parser.add_argument("--cpp-encoder", action="store_true", help="Run transformer.encoder in C++ after MLX prompt encoding.")
    parser.add_argument("--cpp-decoder", action="store_true", help="Run transformer.decoder and box/logit head in C++ after the C++ encoder.")
    parser.add_argument("--cpp-segmentation", action="store_true", help="Run the segmentation head in C++ after the decoder.")
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
    width, height = image.size
    image_arr = mx.array(transform_image(image, 1008))

    tokenizer = model.backbone.language_backbone.tokenizer
    tokenized = np.array(tokenizer([args.prompt], context_length=model.backbone.language_backbone.context_length)[0], dtype=np.int32)

    with tempfile.TemporaryDirectory(prefix="sam3hybrid_", dir=str(ROOT / "out")) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        image_tensor_path = tmp_dir / "image.npy"
        trunk_path = tmp_dir / "trunk.npy"
        tokens_path = tmp_dir / "tokens.txt"
        trunk_prefix = tmp_dir / "vision_trunk"
        call_image_prefix = tmp_dir / "call_image"
        text_prefix = tmp_dir / "text"

        np.save(image_tensor_path, np.array(image_arr, dtype=np.float32))
        np.savetxt(tokens_path, tokenized, fmt="%d")

        if args.cpp_image_backbone:
            trunk_cmd = [
                str(ROOT / "build" / "sam3-vision-trunk"),
                str(args.gguf),
                str(image_tensor_path),
                str(trunk_prefix),
            ]
            if args.cpu:
                trunk_cmd.append("--cpu")
            run_command(trunk_cmd, ROOT)
            trunk_path = trunk_prefix.with_name(trunk_prefix.name + ".trunk.npy")
        else:
            trunk_out = model.backbone.vision_backbone.trunk(image_arr)[-1]
            np.save(trunk_path, np.array(trunk_out, dtype=np.float32))

        call_image_cmd = [
            str(ROOT / "build" / "sam3-call-image"),
            str(args.gguf),
            str(trunk_path),
            str(call_image_prefix),
        ]
        text_cmd = [
            str(ROOT / "build" / "sam3-text-encode"),
            str(args.gguf),
            str(tokens_path),
            str(text_prefix),
        ]
        if args.cpu:
            call_image_cmd.append("--cpu")
            text_cmd.append("--cpu")

        run_command(call_image_cmd, ROOT)
        run_command(text_cmd, ROOT)

        backbone_fpn = []
        vision_pos_enc = []
        level = 0
        while (call_image_prefix.with_name(f"{call_image_prefix.name}.fpn_{level}.npy")).exists():
            backbone_fpn.append(mx.array(np.load(call_image_prefix.with_name(f"{call_image_prefix.name}.fpn_{level}.npy"))))
            vision_pos_enc.append(mx.array(np.load(call_image_prefix.with_name(f"{call_image_prefix.name}.pos_{level}.npy"))))
            level += 1

        language_features = mx.array(np.load(text_prefix.with_name(text_prefix.name + ".memory.npy")))
        language_mask = mx.array((tokenized == 0)[None])

        backbone_out = {
            "vision_features": mx.array(np.load(call_image_prefix.with_name(call_image_prefix.name + ".vision_features.npy"))),
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": backbone_fpn,
            "language_features": language_features,
            "language_mask": language_mask,
        }

        find_stage = FindStage(
            img_ids=mx.array([0], dtype=mx.int64),
            text_ids=mx.array([0], dtype=mx.int64),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        if args.cpp_prompt or args.cpp_encoder or args.cpp_decoder or args.cpp_segmentation:
            prompt_path = tmp_dir / "prompt.npy"
            prompt_mask_path = tmp_dir / "prompt_mask.npy"
            text_mask_path = tmp_dir / "text_mask.npy"
            np.save(text_mask_path, np.array(language_mask, dtype=np.float32))

            vision_features = np.load(call_image_prefix.with_name(call_image_prefix.name + ".vision_features.npy"))
            vis_feat_sizes = [tuple(int(v) for v in vision_features.shape[-2:])]

            if args.cpp_prompt:
                prompt_prefix = tmp_dir / "dummy_prompt"
                prompt_cmd = [
                    str(ROOT / "build" / "sam3-dummy-prompt"),
                    str(args.gguf),
                    str(call_image_prefix.with_name(call_image_prefix.name + ".vision_features.npy")),
                    str(call_image_prefix.with_name(f"{call_image_prefix.name}.pos_{len(vision_pos_enc) - 1}.npy")),
                    str(text_prefix.with_name(text_prefix.name + ".memory.npy")),
                    str(text_mask_path),
                    str(prompt_prefix),
                ]
                if args.cpu:
                    prompt_cmd.append("--cpu")
                run_command(prompt_cmd, ROOT)
                prompt_tensor = mx.array(np.load(prompt_prefix.with_name(prompt_prefix.name + ".prompt.npy")))
                prompt_mask = mx.array(np.load(prompt_prefix.with_name(prompt_prefix.name + ".prompt_mask.npy")))
            else:
                prompt_tensor, prompt_mask, backbone_out = model._encode_prompt(
                    backbone_out,
                    find_stage,
                    model._get_dummy_prompt(),
                )

            np.save(prompt_path, np.array(prompt_tensor, dtype=np.float32))
            np.save(prompt_mask_path, np.array(prompt_mask, dtype=np.float32))

            if args.cpp_encoder or args.cpp_decoder:
                encoder_prefix = tmp_dir / "encoder"
                pos_path = call_image_prefix.with_name(f"{call_image_prefix.name}.pos_{len(vision_pos_enc) - 1}.npy")
                encoder_cmd = [
                    str(ROOT / "build" / "sam3-encoder-fusion"),
                    str(args.gguf),
                    str(call_image_prefix.with_name(call_image_prefix.name + ".vision_features.npy")),
                    str(pos_path),
                    str(prompt_path),
                    str(prompt_mask_path),
                    str(encoder_prefix),
                ]
                if args.cpu:
                    encoder_cmd.append("--cpu")
                run_command(encoder_cmd, ROOT)

                encoder_out = {
                    "encoder_hidden_states": mx.array(np.load(encoder_prefix.with_name(encoder_prefix.name + ".memory.npy"))),
                    "pos_embed": mx.array(np.load(encoder_prefix.with_name(encoder_prefix.name + ".pos_embed.npy"))),
                    "padding_mask": None,
                    "level_start_index": mx.array([0], dtype=mx.int64),
                    "spatial_shapes": mx.array([list(vis_feat_sizes[0])], dtype=mx.int64),
                    "valid_ratios": mx.ones((1, 1, 2), dtype=mx.float32),
                    "vis_feat_sizes": vis_feat_sizes,
                    "prompt_before_enc": prompt_tensor,
                    "prompt_after_enc": prompt_tensor,
                    "prompt_mask": prompt_mask,
                }
            else:
                backbone_out, encoder_out, _ = model._run_encoder(
                    backbone_out,
                    find_stage,
                    prompt_tensor,
                    prompt_mask,
                )

            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": backbone_out,
                },
            }
            if args.cpp_decoder:
                decoder_prefix = tmp_dir / "decoder"
                head_prefix = tmp_dir / "grounding"
                decoder_cmd = [
                    str(ROOT / "build" / "sam3-decoder"),
                    str(args.gguf),
                    str(encoder_prefix.with_name(encoder_prefix.name + ".memory.npy")),
                    str(encoder_prefix.with_name(encoder_prefix.name + ".pos_embed.npy")),
                    str(prompt_path),
                    str(prompt_mask_path),
                    str(decoder_prefix),
                ]
                head_cmd = [
                    str(ROOT / "build" / "sam3-grounding-head"),
                    str(args.gguf),
                    str(decoder_prefix),
                    str(prompt_path),
                    str(prompt_mask_path),
                    str(head_prefix),
                ]
                if args.cpu:
                    decoder_cmd.append("--cpu")
                    head_cmd.append("--cpu")
                run_command(decoder_cmd, ROOT)
                run_command(head_cmd, ROOT)

                hs_layers = []
                for layer in range(6):
                    hs_layers.append(np.load(decoder_prefix.with_name(f"{decoder_prefix.name}.hs_{layer:02d}.npy")).transpose(1, 0, 2))
                hs = mx.array(np.stack(hs_layers, axis=0))

                out["pred_boxes"] = mx.array(np.load(head_prefix.with_name(f"{head_prefix.name}.pred_boxes_05.npy")))
                out["pred_logits"] = mx.array(np.load(head_prefix.with_name(f"{head_prefix.name}.pred_logits_05.npy")))
                out["presence_logit_dec"] = mx.array(np.load(decoder_prefix.with_name(f"{decoder_prefix.name}.presence_05.npy")).reshape(1, 1))
            else:
                out, hs = model._run_decoder(
                    memory=out["encoder_hidden_states"],
                    pos_embed=encoder_out["pos_embed"],
                    src_mask=encoder_out["padding_mask"],
                    out=out,
                    prompt=prompt_tensor,
                    prompt_mask=prompt_mask,
                    encoder_out=encoder_out,
                )

            if args.cpp_segmentation:
                seg_memory_path = tmp_dir / "seg_memory.npy"
                seg_hs_path = tmp_dir / "seg_hs.npy"
                seg_prefix = tmp_dir / "segmentation"
                np.save(seg_memory_path, np.array(out["encoder_hidden_states"], dtype=np.float32))
                np.save(seg_hs_path, np.ascontiguousarray(np.array(hs[-1].transpose(1, 0, 2), dtype=np.float32)))

                seg_cmd = [
                    str(ROOT / "build" / "sam3-segmentation-head"),
                    str(args.gguf),
                    str(call_image_prefix.with_name(call_image_prefix.name + ".fpn_0.npy")),
                    str(call_image_prefix.with_name(call_image_prefix.name + ".fpn_1.npy")),
                    str(call_image_prefix.with_name(call_image_prefix.name + ".fpn_2.npy")),
                    str(seg_memory_path),
                    str(prompt_path),
                    str(prompt_mask_path),
                    str(seg_hs_path),
                    str(seg_prefix),
                ]
                if args.cpu:
                    seg_cmd.append("--cpu")
                run_command(seg_cmd, ROOT)
                out["pred_masks"] = mx.array(np.load(seg_prefix.with_name(seg_prefix.name + ".pred_masks.npy")))
                out["semantic_seg"] = mx.array(np.load(seg_prefix.with_name(seg_prefix.name + ".semantic_seg.npy")))
            else:
                model._run_segmentation_heads(
                    out=out,
                    backbone_out=backbone_out,
                    img_ids=find_stage.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=out["encoder_hidden_states"],
                    prompt=prompt_tensor,
                    prompt_mask=prompt_mask,
                    hs=hs,
                )
            outputs = out
        else:
            outputs = model.call_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                geometric_prompt=model._get_dummy_prompt(),
                find_target=None,
            )

    out_bbox = outputs["pred_boxes"]
    out_logits = outputs["pred_logits"]
    out_masks = outputs["pred_masks"]
    out_probs = mx.sigmoid(out_logits)
    presence_score = mx.sigmoid(outputs["presence_logit_dec"])[:, None]
    out_probs = (out_probs * presence_score).squeeze(-1)

    keep = out_probs > 0.5
    keep_idx = np.array(keep[0]).nonzero()[0]
    keep_idx_mx = mx.array(keep_idx)

    scores = np.array(out_probs[0][keep_idx_mx], dtype=np.float32)
    masks = out_masks[0][keep_idx_mx]
    boxes = out_bbox[0][keep_idx_mx]

    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    scale = mx.array([width, height, width, height])
    boxes = np.array(boxes * scale[None, :], dtype=np.float32)

    masks = interpolate(
        masks[:, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    masks = np.array(mx.sigmoid(masks), dtype=np.float32)

    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        scores=scores,
        boxes=boxes,
        masks=masks,
    )
    save_overlay(image, masks[:, 0], args.output_prefix.with_name(args.output_prefix.name + ".overlay.png"))

    meta = {
        "image": str(args.image),
        "prompt": args.prompt,
        "weights": str(args.weights),
        "gguf": str(args.gguf),
        "cpu": args.cpu,
        "cpp_image_backbone": args.cpp_image_backbone,
        "cpp_encoder": args.cpp_encoder,
        "cpp_decoder": args.cpp_decoder,
        "count": int(scores.shape[0]),
        "hash": hashlib.sha256(f"{args.image}|{args.prompt}".encode()).hexdigest()[:16],
    }
    with args.output_prefix.with_suffix(".json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"image={args.image}")
    print(f"prompt={args.prompt!r}")
    print(f"detections={scores.shape[0]}")
    print(f"npz={args.output_prefix.with_suffix('.npz')}")
    print(f"overlay={args.output_prefix.with_name(args.output_prefix.name + '.overlay.png')}")

    if args.compare_mlx:
        ref = run_mlx_reference(model, image, args.prompt, width, height)
        compare_arrays("scores", ref["scores"], scores)
        compare_arrays("boxes", ref["boxes"], boxes)
        compare_arrays("masks", ref["masks"], masks)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
