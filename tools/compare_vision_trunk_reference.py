#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compare(label: str, reference_path: Path, candidate_path: Path) -> None:
    ref = np.load(reference_path)
    cand = np.load(candidate_path)
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
    parser = argparse.ArgumentParser(description="Compare SAM3 vision-trunk reference tensors.")
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--candidate-prefix", type=Path, required=True)
    parser.add_argument("--layers", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare("trunk", args.reference_prefix.with_name(args.reference_prefix.name + ".trunk.npy"), args.candidate_prefix.with_name(args.candidate_prefix.name + ".trunk.npy"))

    stem_ref = args.reference_prefix.with_name(args.reference_prefix.name + ".stem.npy")
    stem_cand = args.candidate_prefix.with_name(args.candidate_prefix.name + ".stem.npy")
    if stem_ref.exists() and stem_cand.exists():
        compare("stem", stem_ref, stem_cand)

    ln_pre_ref = args.reference_prefix.with_name(args.reference_prefix.name + ".ln_pre.npy")
    ln_pre_cand = args.candidate_prefix.with_name(args.candidate_prefix.name + ".ln_pre.npy")
    if ln_pre_ref.exists() and ln_pre_cand.exists():
        compare("ln_pre", ln_pre_ref, ln_pre_cand)

    if args.layers:
        for layer in range(32):
            ref = args.reference_prefix.with_name(f"{args.reference_prefix.name}.layer_{layer:02d}.npy")
            cand = args.candidate_prefix.with_name(f"{args.candidate_prefix.name}.layer_{layer:02d}.npy")
            if ref.exists() and cand.exists():
                compare(f"layer_{layer:02d}", ref, cand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
