#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare C++ vision-neck outputs against MLX reference.")
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--candidate-prefix", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for level in range(args.levels):
        ref = np.load(args.reference_prefix.with_name(f"{args.reference_prefix.name}.level_{level}.npy"))
        cand = np.load(args.candidate_prefix.with_name(f"{args.candidate_prefix.name}.level_{level}.npy"))
        diff = ref - cand
        abs_diff = np.abs(diff)
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

        print(f"level_{level}")
        print(f"  reference_shape={ref.shape} candidate_shape={cand.shape}")
        print(f"  max_abs={float(abs_diff.max()):.9f}")
        print(f"  mean_abs={float(abs_diff.mean()):.9f}")
        print(f"  rmse={float(np.sqrt(np.mean(diff ** 2))):.9f}")
        print(
            "  worst_idx="
            f"{idx} ref={float(ref[idx]):.9f} cand={float(cand[idx]):.9f}"
        )

        ref_pos_path = args.reference_prefix.with_name(f"{args.reference_prefix.name}.pos_{level}.npy")
        cand_pos_path = args.candidate_prefix.with_name(f"{args.candidate_prefix.name}.pos_{level}.npy")
        if ref_pos_path.exists() and cand_pos_path.exists():
            ref_pos = np.load(ref_pos_path)
            cand_pos = np.load(cand_pos_path)
            pos_diff = ref_pos - cand_pos
            pos_abs_diff = np.abs(pos_diff)
            pos_idx = np.unravel_index(np.argmax(pos_abs_diff), pos_abs_diff.shape)
            print(f"  pos_shape={ref_pos.shape} candidate_pos_shape={cand_pos.shape}")
            print(f"  pos_max_abs={float(pos_abs_diff.max()):.9f}")
            print(f"  pos_mean_abs={float(pos_abs_diff.mean()):.9f}")
            print(f"  pos_rmse={float(np.sqrt(np.mean(pos_diff ** 2))):.9f}")
            print(
                "  pos_worst_idx="
                f"{pos_idx} ref={float(ref_pos[pos_idx]):.9f} cand={float(cand_pos[pos_idx]):.9f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
