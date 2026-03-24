#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "llama.cpp" / "gguf-py"))

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFWriter, quantize  # noqa: E402

DEFAULT_OUTPUT = ROOT / "models" / "sam3-image-f16.gguf"
DEFAULT_INDEX = ROOT / "models" / "mlx-sam3" / "model.safetensors.index.json"
DEFAULT_INPUT = ROOT / "models" / "mlx-sam3" / "model.safetensors"

QTYPE_BY_NAME = {
    "F16": GGMLQuantizationType.F16,
    "F32": GGMLQuantizationType.F32,
    "Q4_0": GGMLQuantizationType.Q4_0,
    "Q4_1": GGMLQuantizationType.Q4_1,
    "Q5_0": GGMLQuantizationType.Q5_0,
    "Q5_1": GGMLQuantizationType.Q5_1,
    "Q8_0": GGMLQuantizationType.Q8_0,
}
FLOAT_QTYPES = {
    GGMLQuantizationType.F16,
    GGMLQuantizationType.F32,
}
SCHEMES = {
    "f16_ref",
    "f32_ref",
    "q8_0",
    "balanced",
    "small",
    "aggressive",
}
STAGES = (
    "text",
    "vision_trunk",
    "vision_convs",
    "encoder",
    "decoder",
    "segmentation",
    "grounding",
    "geometry",
    "other",
)
SENSITIVE_SUBSTRINGS = (
    "embedding",
    "positional",
    "pos_embed",
    "freqs_cis",
    "query_embed",
    "reference_points",
    "presence_token",
    "boxrpb",
    "bbox_embed",
    "dot_prod_scoring",
    "level_embed",
    "mask_token",
    "mask_tokens",
    "cls_embed",
    "iou_token",
    "text_projection",
)


@dataclass(frozen=True)
class RegexOverride:
    pattern: str
    regex: re.Pattern[str]
    qtype: GGMLQuantizationType


@dataclass(frozen=True)
class QuantDecision:
    requested: GGMLQuantizationType
    assigned: GGMLQuantizationType
    reason: str
    quantized: bool


def parse_qtype(name: str) -> GGMLQuantizationType:
    key = name.strip().upper()
    try:
        return QTYPE_BY_NAME[key]
    except KeyError as exc:
        choices = ", ".join(sorted(QTYPE_BY_NAME))
        raise argparse.ArgumentTypeError(f"unsupported qtype {name!r}; choose from {choices}") from exc


def parse_stage_override(spec: str) -> tuple[str, GGMLQuantizationType]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("stage override must use STAGE=QTYPE")
    stage, qtype_name = spec.split("=", 1)
    stage = stage.strip()
    if stage not in STAGES:
        raise argparse.ArgumentTypeError(
            f"unknown stage {stage!r}; choose from {', '.join(STAGES)}"
        )
    return stage, parse_qtype(qtype_name)


def parse_regex_override(spec: str) -> RegexOverride:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("override must use REGEX=QTYPE")
    pattern, qtype_name = spec.rsplit("=", 1)
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regex {pattern!r}: {exc}") from exc
    return RegexOverride(pattern=pattern, regex=regex, qtype=parse_qtype(qtype_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert public MLX SAM3 image weights into GGUF."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to model.safetensors.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Path to model.safetensors.index.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GGUF path. Defaults to models/sam3-image-<scheme>.gguf.",
    )
    parser.add_argument(
        "--outtype",
        choices=["f16", "f32"],
        default="f16",
        help="Floating point format for tensors that remain unquantized.",
    )
    parser.add_argument(
        "--scheme",
        choices=sorted(SCHEMES),
        default="f16_ref",
        help="Named quantization scheme. f16_ref stays full float16 by default.",
    )
    parser.add_argument(
        "--stage-qtype",
        action="append",
        default=[],
        metavar="STAGE=QTYPE",
        help=(
            "Override a logical stage qtype. Stages: "
            + ", ".join(STAGES)
            + ". Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="REGEX=QTYPE",
        help="Override qtype for matching tensor names. Later overrides win.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Analyze tensor assignments and estimated sizes without writing a GGUF.",
    )
    args = parser.parse_args()
    args.stage_qtype = [parse_stage_override(spec) for spec in args.stage_qtype]
    args.override = [parse_regex_override(spec) for spec in args.override]
    return args


def to_numpy_dtype(outtype: str) -> np.dtype:
    if outtype == "f16":
        return np.float16
    return np.float32


def to_float_qtype(outtype: str) -> GGMLQuantizationType:
    if outtype == "f16":
        return GGMLQuantizationType.F16
    return GGMLQuantizationType.F32


def infer_stage(name: str) -> str:
    if name.startswith("backbone.language_backbone.encoder"):
        return "text"
    if name.startswith("backbone.vision_backbone.trunk"):
        return "vision_trunk"
    if name.startswith("backbone.vision_backbone.convs") or name.startswith(
        "backbone.vision_backbone.sam2_convs"
    ):
        return "vision_convs"
    if name.startswith("transformer.encoder"):
        return "encoder"
    if name.startswith("transformer.decoder"):
        return "decoder"
    if name.startswith("segmentation_head"):
        return "segmentation"
    if name.startswith("dot_prod_scoring"):
        return "grounding"
    if name.startswith("geometry_encoder"):
        return "geometry"
    return "other"


def is_sensitive_tensor(name: str) -> bool:
    lower = name.lower()
    if name.endswith(".bias"):
        return True
    if "norm" in lower:
        return True
    return any(token in lower for token in SENSITIVE_SUBSTRINGS)


def is_float_tensor(tensor: np.ndarray) -> bool:
    bfloat16_dtype = getattr(np, "bfloat16", None)
    if bfloat16_dtype is not None and tensor.dtype == bfloat16_dtype:
        return True
    return np.issubdtype(tensor.dtype, np.floating)


def is_quantizable_tensor(tensor: np.ndarray) -> bool:
    return tensor.ndim == 2 and is_float_tensor(tensor)


def preferred_float_qtype(
    name: str,
    base_float_qtype: GGMLQuantizationType,
) -> GGMLQuantizationType:
    if base_float_qtype == GGMLQuantizationType.F32:
        return GGMLQuantizationType.F32
    if is_sensitive_tensor(name):
        return GGMLQuantizationType.F32
    return base_float_qtype


def scheme_qtype(
    scheme: str,
    *,
    name: str,
    stage: str,
    tensor: np.ndarray,
    float_qtype: GGMLQuantizationType,
) -> GGMLQuantizationType:
    if scheme == "f16_ref":
        return float_qtype
    if scheme == "f32_ref":
        return GGMLQuantizationType.F32
    if not is_quantizable_tensor(tensor) or is_sensitive_tensor(name):
        return float_qtype
    if scheme == "q8_0":
        return GGMLQuantizationType.Q8_0
    if scheme == "balanced":
        if stage in {"text", "segmentation", "grounding", "geometry"}:
            return GGMLQuantizationType.Q8_0
        if stage in {"vision_trunk", "encoder", "decoder"}:
            return GGMLQuantizationType.Q5_0
        return float_qtype
    if scheme == "small":
        if stage in {"text", "segmentation", "grounding", "geometry"}:
            return GGMLQuantizationType.Q5_0
        if stage in {"vision_trunk", "encoder", "decoder"}:
            return GGMLQuantizationType.Q4_1
        return float_qtype
    if scheme == "aggressive":
        if stage in {"text", "segmentation", "grounding", "geometry"}:
            return GGMLQuantizationType.Q4_1
        if stage in {"vision_trunk", "encoder", "decoder"}:
            return GGMLQuantizationType.Q4_0
        return float_qtype
    raise ValueError(f"unsupported scheme {scheme!r}")


def qtype_fallback_chain(
    requested: GGMLQuantizationType,
    base_float_qtype: GGMLQuantizationType,
) -> list[GGMLQuantizationType]:
    if requested in FLOAT_QTYPES:
        return [requested]
    if requested == GGMLQuantizationType.Q8_0:
        return [requested, base_float_qtype]
    return [requested, GGMLQuantizationType.Q8_0, base_float_qtype]


def can_store_as_qtype(tensor: np.ndarray, qtype: GGMLQuantizationType) -> bool:
    if qtype in FLOAT_QTYPES:
        return is_float_tensor(tensor)
    if not is_quantizable_tensor(tensor):
        return False
    block_size, _ = GGML_QUANT_SIZES[qtype]
    return tensor.shape[-1] % block_size == 0


def choose_qtype(
    requested: GGMLQuantizationType,
    *,
    tensor: np.ndarray,
    float_qtype: GGMLQuantizationType,
) -> QuantDecision:
    for candidate in qtype_fallback_chain(requested, float_qtype):
        if can_store_as_qtype(tensor, candidate):
            if candidate == requested:
                return QuantDecision(
                    requested=requested,
                    assigned=candidate,
                    reason="requested",
                    quantized=candidate not in FLOAT_QTYPES,
                )
            return QuantDecision(
                requested=requested,
                assigned=candidate,
                reason="fallback",
                quantized=candidate not in FLOAT_QTYPES,
            )
    return QuantDecision(
        requested=requested,
        assigned=float_qtype,
        reason="float_fallback",
        quantized=False,
    )


def materialize_tensor(
    tensor: np.ndarray,
    qtype: GGMLQuantizationType,
    base_float_dtype: np.dtype,
) -> np.ndarray:
    del base_float_dtype
    if qtype == GGMLQuantizationType.F16:
        with np.errstate(over="ignore"):
            return tensor.astype(np.float16, copy=False)
    if qtype == GGMLQuantizationType.F32:
        return tensor.astype(np.float32, copy=False)
    return quantize(tensor.astype(np.float32, copy=False), qtype)


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    if args.scheme == "f16_ref" and not args.stage_qtype and not args.override and args.outtype == "f16":
        return DEFAULT_OUTPUT
    if args.scheme.endswith("_ref"):
        suffix = args.outtype
    else:
        suffix = args.scheme
    if args.stage_qtype or args.override:
        suffix = f"{suffix}-custom"
    return ROOT / "models" / f"sam3-image-{suffix}.gguf"


def human_size(nbytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(nbytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{nbytes} B"


def shorten_tensor_name(name: str) -> str:
    digest = hashlib.sha1(name.encode()).hexdigest()[:12]
    return f"t_{digest}"


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.index.exists():
        raise FileNotFoundError(args.index)

    with args.index.open() as f:
        index = json.load(f)

    output_path = resolve_output_path(args)
    base_float_dtype = to_numpy_dtype(args.outtype)
    base_float_qtype = to_float_qtype(args.outtype)
    stage_overrides = dict(args.stage_qtype)
    regex_overrides = args.override

    writer = None
    if not args.report_only:
        writer = GGUFWriter(str(output_path), "sam3")
        writer.add_name("sam3-image")
        writer.add_string("sam3.source_repo", "mlx-community/sam3-image")
        writer.add_string("sam3.source_impl", "Deekshith-Dade/mlx_sam3")
        writer.add_string("sam3.task", "image-segmentation")
        writer.add_string("sam3.reference_path", "text-only-detector")
        writer.add_string("general.quantized_by", "sam3cpp/tools/convert_mlx_sam3_to_gguf.py")
        writer.add_string("sam3.quant_scheme", args.scheme)
        writer.add_string("sam3.quant_base_outtype", args.outtype)
        writer.add_string(
            "sam3.quant_stage_overrides",
            json.dumps({stage: qtype.name for stage, qtype in stage_overrides.items()}, sort_keys=True),
        )
        writer.add_string(
            "sam3.quant_regex_overrides",
            json.dumps(
                [{"pattern": item.pattern, "qtype": item.qtype.name} for item in regex_overrides],
                indent=2,
                sort_keys=True,
            ),
        )
        writer.add_int32("sam3.image_size", 1008)
        writer.add_int32("sam3.patch_size", 14)
        writer.add_int32("sam3.vision_layers", 32)
        writer.add_int32("sam3.text_layers", 24)
        writer.add_uint64("sam3.tensor_count", len(index["weight_map"]))
        writer.add_uint64("sam3.total_size", int(index["metadata"]["total_size"]))
        writer.add_bool("sam3.regenerate_freqs_cis", True)

    skipped = []
    tensor_name_map: dict[str, str] = {}
    tensor_info_map: dict[str, dict[str, object]] = {}
    by_qtype_count: Counter[str] = Counter()
    by_qtype_bytes: Counter[str] = Counter()
    by_stage_count: Counter[str] = Counter()
    by_stage_bytes: Counter[str] = Counter()
    by_stage_quantized_count: Counter[str] = Counter()
    fallback_samples: list[dict[str, object]] = []
    requested_vs_assigned: Counter[tuple[str, str]] = Counter()
    total_source_bytes = 0
    total_reference_bytes = 0
    total_output_bytes = 0
    quantized_tensor_count = 0

    with safe_open(args.input, framework="np") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            total_source_bytes += tensor.nbytes
            if tensor.dtype == np.complex64:
                skipped.append(name)
                continue
            stage = infer_stage(name)
            float_qtype = preferred_float_qtype(name, base_float_qtype)
            requested = scheme_qtype(
                args.scheme,
                name=name,
                stage=stage,
                tensor=tensor,
                float_qtype=float_qtype,
            )
            if stage in stage_overrides:
                requested = stage_overrides[stage]
            for override in regex_overrides:
                if override.regex.search(name):
                    requested = override.qtype

            decision = choose_qtype(
                requested,
                tensor=tensor,
                float_qtype=float_qtype,
            )
            stored = materialize_tensor(tensor, decision.assigned, base_float_dtype)
            if is_float_tensor(tensor):
                with np.errstate(over="ignore"):
                    reference_nbytes = tensor.astype(base_float_dtype, copy=False).nbytes
            else:
                reference_nbytes = tensor.nbytes
            total_reference_bytes += reference_nbytes
            total_output_bytes += stored.nbytes
            by_qtype_count[decision.assigned.name] += 1
            by_qtype_bytes[decision.assigned.name] += stored.nbytes
            by_stage_count[stage] += 1
            by_stage_bytes[stage] += stored.nbytes
            if decision.quantized:
                quantized_tensor_count += 1
                by_stage_quantized_count[stage] += 1
            requested_vs_assigned[(decision.requested.name, decision.assigned.name)] += 1

            short_name = shorten_tensor_name(name)
            tensor_name_map[short_name] = name
            tensor_info_map[short_name] = {
                "name": name,
                "shape": list(tensor.shape),
                "stage": stage,
                "requested_qtype": decision.requested.name,
                "assigned_qtype": decision.assigned.name,
            }
            if decision.reason != "requested":
                fallback_samples.append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "requested_qtype": decision.requested.name,
                        "assigned_qtype": decision.assigned.name,
                        "reason": decision.reason,
                    }
                )

            if writer is not None:
                if decision.assigned in FLOAT_QTYPES:
                    writer.add_tensor(short_name, stored)
                else:
                    writer.add_tensor(short_name, stored, raw_dtype=decision.assigned)

    report = {
        "config": {
            "scheme": args.scheme,
            "outtype": args.outtype,
            "input": str(args.input),
            "index": str(args.index),
            "output": str(output_path),
            "report_only": args.report_only,
            "stage_overrides": {stage: qtype.name for stage, qtype in stage_overrides.items()},
            "regex_overrides": [
                {"pattern": item.pattern, "qtype": item.qtype.name} for item in regex_overrides
            ],
        },
        "totals": {
            "tensor_count": len(index["weight_map"]),
            "written_tensor_count": sum(by_qtype_count.values()),
            "skipped_tensor_count": len(skipped),
            "quantized_tensor_count": quantized_tensor_count,
            "source_bytes": total_source_bytes,
            "reference_bytes": total_reference_bytes,
            "output_bytes": total_output_bytes,
            "compression_vs_source": (
                float(total_output_bytes) / float(total_source_bytes) if total_source_bytes else 0.0
            ),
            "compression_vs_reference": (
                float(total_output_bytes) / float(total_reference_bytes) if total_reference_bytes else 0.0
            ),
        },
        "by_qtype": {
            qtype: {
                "tensor_count": by_qtype_count[qtype],
                "bytes": by_qtype_bytes[qtype],
            }
            for qtype in sorted(by_qtype_count)
        },
        "by_stage": {
            stage: {
                "tensor_count": by_stage_count[stage],
                "quantized_tensor_count": by_stage_quantized_count[stage],
                "bytes": by_stage_bytes[stage],
            }
            for stage in STAGES
            if by_stage_count[stage]
        },
        "requested_vs_assigned": {
            f"{requested}->{assigned}": count
            for (requested, assigned), count in sorted(requested_vs_assigned.items())
        },
        "fallback_samples": fallback_samples[:64],
        "skipped_tensors": skipped,
    }

    print(
        json.dumps(
            {
                "scheme": args.scheme,
                "output": str(output_path),
                "quantized_tensors": quantized_tensor_count,
                "source_size": human_size(total_source_bytes),
                "reference_size": human_size(total_reference_bytes),
                "output_size": human_size(total_output_bytes),
                "compression_vs_reference": report["totals"]["compression_vs_reference"],
                "by_qtype": report["by_qtype"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    if writer is not None:
        writer.add_uint64("sam3.skipped_tensor_count", len(skipped))
        writer.add_uint64("sam3.quantized_tensor_count", quantized_tensor_count)
        writer.add_string("sam3.quant_report_json", json.dumps(report, sort_keys=True))
        if skipped:
            writer.add_array("sam3.skipped_tensors", skipped)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        sidecar = output_path.with_suffix(output_path.suffix + ".tensor_map.json")
        with sidecar.open("w") as f:
            json.dump(tensor_name_map, f, indent=2, sort_keys=True)

        tensor_info_path = output_path.with_suffix(output_path.suffix + ".tensor_info.json")
        with tensor_info_path.open("w") as f:
            json.dump(tensor_info_map, f, indent=2, sort_keys=True)

        quant_report_path = output_path.with_suffix(output_path.suffix + ".quant_report.json")
        with quant_report_path.open("w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
