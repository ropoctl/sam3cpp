#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable

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


def run_prompt(model, image_path: Path, prompt: str):
    import mlx.core as mx

    image = Image.open(image_path)
    width, height = image.size
    image_arr = mx.array(transform_image(image, 1008))

    state = {
        "original_height": height,
        "original_width": width,
        "backbone_out": model.backbone.call_image(image_arr),
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
    keep_idx = mx.array(keep_idx)

    out_probs = np.array(out_probs[0][keep_idx])
    out_masks = outputs["pred_masks"][0][keep_idx]
    out_bbox = out_bbox[0][keep_idx]

    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    scale = mx.array([width, height, width, height])
    boxes = np.array(boxes * scale[None, :])

    out_masks = interpolate(
        out_masks[:, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    out_masks = np.array(mx.sigmoid(out_masks))

    return {
        "scores": out_probs.astype(np.float32),
        "boxes": boxes.astype(np.float32),
        "masks": out_masks.astype(np.float32),
    }


def iter_images(downloads: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for path in downloads.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache MLX SAM3 golden outputs.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "mlx-sam3" / "model.safetensors")
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "golden" / "mlx")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prompt", action="append", default=[])
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt or ["object"]
    images = sorted(iter_images(args.downloads))
    if not images:
        raise RuntimeError(f"no images found under {args.downloads}")
    rng = random.Random(args.seed)
    images = rng.sample(images, min(args.sample_count, len(images)))

    model = build_sam3_image_model(
        checkpoint_path=str(args.weights),
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )

    manifest = []
    for image_path in images:
        for prompt in prompts:
            key = hashlib.sha256(f"{image_path}|{prompt}".encode()).hexdigest()[:16]
            out_file = args.cache_dir / f"{key}.npz"
            if not out_file.exists():
                result = run_prompt(model, image_path, prompt)
                np.savez_compressed(out_file, **result)
            manifest.append(
                {
                    "key": key,
                    "image": str(image_path),
                    "prompt": prompt,
                    "cache": str(out_file),
                }
            )

    with (args.cache_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
