#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_candidate_feature(prefix: Path, level: int) -> np.ndarray:
    for suffix in (f".fpn_{level}.npy", f".level_{level}.npy"):
        path = prefix.with_name(prefix.name + suffix)
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"missing candidate feature for level {level}")


def load_candidate_position(prefix: Path, level: int) -> np.ndarray:
    path = prefix.with_name(f"{prefix.name}.pos_{level}.npy")
    if path.exists():
        return np.load(path)
    raise FileNotFoundError(f"missing candidate position for level {level}")


def compare_arrays(label: str, ref: np.ndarray, cand: np.ndarray) -> None:
    diff = ref - cand
    abs_diff = np.abs(diff)
    idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    print(label)
    print(f"  reference_shape={ref.shape} candidate_shape={cand.shape}")
    print(f"  max_abs={float(abs_diff.max()):.9f}")
    print(f"  mean_abs={float(abs_diff.mean()):.9f}")
    print(f"  rmse={float(np.sqrt(np.mean(diff ** 2))):.9f}")
    print(f"  worst_idx={idx} ref={float(ref[idx]):.9f} cand={float(cand[idx]):.9f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SAM3 backbone.call_image() reference tensors against C++ neck outputs.")
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--candidate-prefix", type=Path, required=True)
    parser.add_argument("--levels", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.levels is None:
        levels = 0
        while args.reference_prefix.with_name(f"{args.reference_prefix.name}.fpn_{levels}.npy").exists():
            levels += 1
    else:
        levels = args.levels

    for level in range(levels):
        ref_fpn = np.load(args.reference_prefix.with_name(f"{args.reference_prefix.name}.fpn_{level}.npy"))
        cand_level = load_candidate_feature(args.candidate_prefix, level)
        compare_arrays(f"fpn_{level}", ref_fpn, cand_level)

        ref_pos = np.load(args.reference_prefix.with_name(f"{args.reference_prefix.name}.pos_{level}.npy"))
        cand_pos = load_candidate_position(args.candidate_prefix, level)
        compare_arrays(f"pos_{level}", ref_pos, cand_pos)

    ref_vision_features = np.load(args.reference_prefix.with_name(args.reference_prefix.name + ".vision_features.npy"))
    vision_feature_path = args.candidate_prefix.with_name(args.candidate_prefix.name + ".vision_features.npy")
    cand_vision_features = (
        np.load(vision_feature_path)
        if vision_feature_path.exists()
        else load_candidate_feature(args.candidate_prefix, levels - 1)
    )
    compare_arrays("vision_features", ref_vision_features, cand_vision_features)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
