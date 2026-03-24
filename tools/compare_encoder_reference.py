#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare encoder-fusion outputs against MLX reference.")
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--candidate-prefix", type=Path, required=True)
    parser.add_argument("--layers", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    compare_arrays(
        "memory",
        np.load(args.reference_prefix.with_name(args.reference_prefix.name + ".memory.npy")),
        np.load(args.candidate_prefix.with_name(args.candidate_prefix.name + ".memory.npy")),
    )
    compare_arrays(
        "pos_embed",
        np.load(args.reference_prefix.with_name(args.reference_prefix.name + ".pos_embed.npy")),
        np.load(args.candidate_prefix.with_name(args.candidate_prefix.name + ".pos_embed.npy")),
    )

    if args.layers:
        layer_idx = 0
        while True:
            ref_path = args.reference_prefix.with_name(f"{args.reference_prefix.name}.layer_{layer_idx:02d}.npy")
            cand_path = args.candidate_prefix.with_name(f"{args.candidate_prefix.name}.layer_{layer_idx:02d}.npy")
            if not ref_path.exists() or not cand_path.exists():
                break
            compare_arrays(f"layer_{layer_idx:02d}", np.load(ref_path), np.load(cand_path))
            layer_idx += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
