#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def compare_arrays(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    if ref.shape != cand.shape:
        return {
            "shape_match": 0.0,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
        }
    diff = np.abs(ref.astype(np.float32) - cand.astype(np.float32))
    return {
        "shape_match": 1.0,
        "max_abs": float(diff.max(initial=0.0)),
        "mean_abs": float(diff.mean() if diff.size else 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare cached SAM3 outputs.")
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.reference_manifest.open() as f:
        manifest = json.load(f)

    summary = []
    for item in manifest:
        ref_file = Path(item["cache"])
        cand_file = args.candidate_dir / ref_file.name
        if not cand_file.exists():
            summary.append(
                {
                    "key": item["key"],
                    "image": item["image"],
                    "prompt": item["prompt"],
                    "status": "missing",
                }
            )
            continue

        ref = load_npz(ref_file)
        cand = load_npz(cand_file)
        scores = compare_arrays(ref["scores"], cand["scores"])
        boxes = compare_arrays(ref["boxes"], cand["boxes"])
        masks = compare_arrays(ref["masks"], cand["masks"])
        summary.append(
            {
                "key": item["key"],
                "image": item["image"],
                "prompt": item["prompt"],
                "status": "ok",
                "scores": scores,
                "boxes": boxes,
                "masks": masks,
            }
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
