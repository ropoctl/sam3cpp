# sam3.cpp

C++ / ggml inference engine for **SAM 3** (Segment Anything Model 3). Runs on Metal, CUDA, Vulkan, and CPU.

## Quickstart

```bash
# Build
cmake -S . -B build && cmake --build build -j

# Run (auto-downloads weights from HuggingFace on first use, no extra deps)
python3 -c "
import sam3cpp
model = sam3cpp.Sam3Model('hf://rob-laz/sam3-gguf/sam3-image-f16.gguf')
pred = model.predict('photo.jpg', 'person')
print(f'{pred.count} detections')
"
```

Or use the CLI tools directly with a local GGUF:

```bash
./build/sam3-vision-trunk models/sam3-image-f16.gguf image.jpg output_prefix
```

## Pre-trained Weights

Available on HuggingFace: [rob-laz/sam3-gguf](https://huggingface.co/rob-laz/sam3-gguf)

| Variant | Size | Inference (M4 Max) |
|---------|------|--------------------|
| [`sam3-image-f32.gguf`](https://huggingface.co/rob-laz/sam3-gguf/resolve/main/sam3-image-f32.gguf) | 3.2 GB | 3.5s |
| [`sam3-image-f16.gguf`](https://huggingface.co/rob-laz/sam3-gguf/resolve/main/sam3-image-f16.gguf) | 1.7 GB | 3.5s |
| [`sam3-image-q8_0.gguf`](https://huggingface.co/rob-laz/sam3-gguf/resolve/main/sam3-image-q8_0.gguf) | 1.0 GB | 3.2s |

The Python binding accepts `hf://rob-laz/sam3-gguf/<filename>` paths and will download + cache automatically (no extra dependencies — uses stdlib `urllib`).

Based on [SAM 3](https://github.com/facebookresearch/sam3) by Meta FAIR.

## Architecture

- **Vision backbone**: 32-layer ViT with windowed + global attention (1008x1008 input)
- **Text encoder**: CLIP-based language backbone
- **Encoder fusion**: 6-layer transformer with self + cross-attention
- **Decoder**: 6-layer transformer decoder with iterative box refinement
- **Heads**: Grounding head (detection) + segmentation head (masks)

## Building

```bash
cmake -S . -B build
cmake --build build -j
```

Metal is enabled by default on macOS. All other GPU backends are disabled but can be toggled via CMake options.

## Python Binding

```bash
uv pip install pybind11 scikit-build-core
uv pip install -e bindings/python

python3 -c "
import sam3cpp

# Auto-download F16 weights from HuggingFace
model = sam3cpp.Sam3Model('hf://rob-laz/sam3-gguf/sam3-image-f16.gguf')
pred = model.predict('photo.jpg', 'person')

# pred.scores, pred.boxes_xyxy, pred.masks are numpy arrays
for i in range(pred.count):
    print(f'score={pred.scores[i]:.3f} box={pred.boxes_xyxy[i]}')

# Overlay visualization
overlay = sam3cpp.draw_overlay('photo.jpg', pred, mask_index=0)
overlay.save('output.png')
"
```

## Rust Binding

```bash
cd bindings/rust
cargo build --example predict
cargo run --example predict
```

## CLI Tools

| Tool | Description |
|------|-------------|
| `sam3-inspect` | Report model metadata and tensor inventory |
| `sam3-vision-trunk` | Run the ViT backbone on an image |
| `sam3-call-image` | Produce the FPN pyramid from a trunk tensor |
| `sam3-text-encode` | Encode text tokens |
| `sam3-encoder-fusion` | Run the encoder fusion transformer |
| `sam3-decoder` | Run the decoder with box refinement |
| `sam3-grounding-head` | Produce detection boxes and logits |
| `sam3-segmentation-head` | Produce segmentation masks |
| `sam3-profile` | Run full pipeline with per-stage timing |

All tools accept `--cpu` to force CPU execution. Default is GPU.

## Converting Weights

```bash
python3 tools/convert_mlx_sam3_to_gguf.py \
  --input models/mlx-sam3/model.safetensors \
  --index models/mlx-sam3/model.safetensors.index.json \
  --output models/sam3-image-f32.gguf

# Quantize
python3 tools/convert_mlx_sam3_to_gguf.py \
  --input models/mlx-sam3/model.safetensors \
  --index models/mlx-sam3/model.safetensors.index.json \
  --output models/sam3-image-q8_0.gguf \
  --quantize q8_0
```

## Performance

End-to-end pipeline benchmark on Apple M4 Max (1008x1008 input, Metal GPU):

| Stage | F32 | F16 | Q8_0 |
|-------|-----|-----|------|
| Vision trunk (32-layer ViT) | 1.9s | 2.1s | 1.8s |
| Vision neck (FPN deconv) | 0.4s | 0.4s | 0.4s |
| Text encoder | 0.2s | 0.1s | 0.1s |
| Encoder fusion (6-layer) | 0.06s | 0.07s | 0.06s |
| Decoder + heads | 0.8s | 0.8s | 0.8s |
| **Total** | **3.5s** | **3.5s** | **3.2s** |

Key GPU optimizations:
- **Vision neck**: transposed convolution decomposed into matmul + pixel shuffle (was 166s with CPU-only `conv_transpose_2d`)
- **Encoder fusion**: flash attention via `ggml_flash_attn_ext` (was 1.8s with materialized O(n²) score matrices)
- **Segmentation head**: group norm with interleaved channels runs on GPU (was 0.6s on CPU)
- **Vision trunk**: window partition/unpartition via reshape+permute (replaces CPU-only `ggml_win_part`)

## License

Apache-2.0
