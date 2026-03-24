import os
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_metallib = _pkg_dir / "default.metallib"
if _metallib.exists():
    os.environ.setdefault("GGML_METAL_PATH_RESOURCES", str(_pkg_dir))

from ._sam3cpp import Prediction
from ._sam3cpp import Sam3Model as _Sam3Model

HF_REPO = "rob-laz/sam3-gguf"
HF_DEFAULT_MODEL = "sam3-image-f16.gguf"


def _cache_dir() -> Path:
    """Return the cache directory for downloaded models."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "sam3cpp"


def resolve_model_path(path: str) -> str:
    """Resolve a model path, downloading from HuggingFace if needed.

    Accepts:
      - Local file paths (returned as-is)
      - ``hf://repo/filename`` URIs (downloaded via HTTPS, no extra deps)
      - Bare filenames like ``sam3-image-f16.gguf`` (resolved against rob-laz/sam3-gguf)
    """
    if os.path.isfile(path):
        return path

    repo_id = HF_REPO
    filename = path

    if path.startswith("hf://"):
        parts = path[5:].split("/", 2)
        if len(parts) == 3:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = parts[2]
        else:
            filename = parts[-1]

    cache = _cache_dir() / repo_id.replace("/", "--")
    cached = cache / filename
    if cached.is_file():
        return str(cached)

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".part")

    import urllib.request
    import sys

    print(f"Downloading {filename} from {repo_id}...", file=sys.stderr)
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", file=sys.stderr)
            if total:
                print(file=sys.stderr)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.rename(cached)
    print(f"Cached at {cached}", file=sys.stderr)
    return str(cached)


class Sam3Model(_Sam3Model):
    """SAM3 model with automatic HuggingFace weight downloading."""

    def __init__(self, gguf_path: str = None, prefer_gpu: bool = True, bpe_path: str = ""):
        if gguf_path is None:
            gguf_path = f"hf://{HF_REPO}/{HF_DEFAULT_MODEL}"
        resolved = resolve_model_path(gguf_path)
        super().__init__(resolved, prefer_gpu, bpe_path)

import numpy as np
from PIL import Image, ImageDraw


def draw_overlay(
    image_path,
    prediction,
    mask_index=0,
    color=(255, 64, 64),
    alpha=0.45,
    draw_box=True,
    draw_label=True,
):
    image = Image.open(image_path).convert("RGBA")
    mask = np.clip(prediction.mask(mask_index), 0.0, 1.0)

    overlay = np.zeros((prediction.height, prediction.width, 4), dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = np.clip(mask * int(255 * alpha), 0, 255).astype(np.uint8)

    composed = Image.alpha_composite(image, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(composed)

    if draw_box:
        x0, y0, x1, y1 = prediction.boxes_xyxy[mask_index]
        draw.rectangle((float(x0), float(y0), float(x1), float(y1)), outline=color, width=3)

    if draw_label:
        score = float(prediction.scores[mask_index])
        draw.text((12, 12), f"mask={mask_index} score={score:.4f}", fill=color)

    return composed


def draw_side_by_side(
    image_path,
    left_prediction,
    right_prediction,
    left_label="left",
    right_label="right",
    mask_index=0,
    color=(255, 64, 64),
    alpha=0.45,
):
    left = draw_overlay(image_path, left_prediction, mask_index=mask_index, color=color, alpha=alpha)
    right = draw_overlay(image_path, right_prediction, mask_index=mask_index, color=color, alpha=alpha)

    width = left.width + right.width
    height = max(left.height, right.height) + 36
    canvas = Image.new("RGBA", (width, height), (18, 20, 24, 255))
    canvas.paste(left, (0, 36))
    canvas.paste(right, (left.width, 36))

    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), left_label, fill=(255, 255, 255, 255))
    draw.text((left.width + 12, 10), right_label, fill=(255, 255, 255, 255))
    return canvas


__all__ = [
    "Prediction",
    "Sam3Model",
    "draw_overlay",
    "draw_side_by_side",
]
