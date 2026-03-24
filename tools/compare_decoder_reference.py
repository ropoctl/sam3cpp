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
    parser = argparse.ArgumentParser(description="Compare decoder outputs against MLX reference.")
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--candidate-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for i in range(6):
        compare_arrays(
            f"hs_{i:02d}",
            np.load(args.reference_prefix.with_name(f"{args.reference_prefix.name}.hs_{i:02d}.npy")),
            np.load(args.candidate_prefix.with_name(f"{args.candidate_prefix.name}.hs_{i:02d}.npy")),
        )
        compare_arrays(
            f"ref_{i:02d}",
            np.load(args.reference_prefix.with_name(f"{args.reference_prefix.name}.ref_{i:02d}.npy")),
            np.load(args.candidate_prefix.with_name(f"{args.candidate_prefix.name}.ref_{i:02d}.npy")),
        )
        compare_arrays(
            f"presence_{i:02d}",
            np.load(args.reference_prefix.with_name(f"{args.reference_prefix.name}.presence_{i:02d}.npy")).reshape(1, 1),
            np.load(args.candidate_prefix.with_name(f"{args.candidate_prefix.name}.presence_{i:02d}.npy")).reshape(1, 1),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
