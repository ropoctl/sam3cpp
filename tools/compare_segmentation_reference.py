#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compare(label: str, ref: np.ndarray, cand: np.ndarray) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MLX and C++ SAM3 segmentation head tensors.")
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--candidate-prefix", type=Path, required=True)
    parser.add_argument("--intermediates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    compare(
        "pred_masks",
        np.load(args.reference_prefix.with_name(args.reference_prefix.name + ".pred_masks.npy")),
        np.load(args.candidate_prefix.with_name(args.candidate_prefix.name + ".pred_masks.npy")),
    )
    compare(
        "semantic_seg",
        np.load(args.reference_prefix.with_name(args.reference_prefix.name + ".semantic_seg.npy")),
        np.load(args.candidate_prefix.with_name(args.candidate_prefix.name + ".semantic_seg.npy")),
    )

    if args.intermediates:
        for name in ("encoder_hidden_states", "encoder_feature_map", "stage0_conv", "stage0_output", "stage1_conv", "pixel_embed", "instance_embed"):
            compare(
                name,
                np.load(args.reference_prefix.with_name(f"{args.reference_prefix.name}.{name}.npy")),
                np.load(args.candidate_prefix.with_name(f"{args.candidate_prefix.name}.{name}.npy")),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
