#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare C++ text-encoder output against MLX reference.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ref = np.load(args.reference)
    cand = np.load(args.candidate)
    diff = ref - cand
    abs_diff = np.abs(diff)
    idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

    print(f"reference_shape={ref.shape} candidate_shape={cand.shape}")
    print(f"max_abs={float(abs_diff.max()):.9f}")
    print(f"mean_abs={float(abs_diff.mean()):.9f}")
    print(f"rmse={float(np.sqrt(np.mean(diff ** 2))):.9f}")
    print(
        "worst_idx="
        f"{idx} ref={float(ref[idx]):.9f} cand={float(cand[idx]):.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
