#include "sam3/vision_trunk.h"

#include "ggml-alloc.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kImageSize = 1008;
constexpr int32_t kPatchSize = 14;
constexpr int32_t kGridSize = kImageSize / kPatchSize;
constexpr int32_t kPretrainGridSize = 336 / kPatchSize;
constexpr int32_t kEmbedDim = 1024;
constexpr int32_t kHeads = 16;
constexpr int32_t kHeadDim = kEmbedDim / kHeads;
constexpr int32_t kLayers = 32;
constexpr int32_t kWindowSize = 24;
constexpr float kLayerNormEps = 1e-5f;
constexpr float kRopeTheta = 10000.0f;
constexpr float kGlobalRopeScale = static_cast<float>(kWindowSize) / static_cast<float>(kGridSize);

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * tensor = model.find_weight(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return tensor;
}

ggml_tensor * ensure_f32(ggml_context * ctx, ggml_tensor * tensor) {
    if (tensor->type == GGML_TYPE_F32) {
        return tensor;
    }
    return ggml_cast(ctx, tensor, GGML_TYPE_F32);
}

std::string block_prefix(int layer) {
    return "backbone.vision_backbone.trunk.blocks." + std::to_string(layer);
}

bool is_global_attention_block(int layer) {
    return layer == 7 || layer == 15 || layer == 23 || layer == 31;
}

ggml_tensor * linear(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * bias,
    ggml_tensor * x
) {
    ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    y = ensure_f32(ctx, y);
    if (bias != nullptr) {
        y = ggml_add(ctx, y, ggml_repeat(ctx, ensure_f32(ctx, bias), y));
    }
    return y;
}

ggml_tensor * layer_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias
) {
    ggml_tensor * y = ggml_norm(ctx, ensure_f32(ctx, x), kLayerNormEps);
    y = ggml_mul(ctx, y, ggml_repeat(ctx, ensure_f32(ctx, weight), y));
    y = ggml_add(ctx, y, ggml_repeat(ctx, ensure_f32(ctx, bias), y));
    return y;
}

ggml_tensor * linear_weight_chunk(
    ggml_context * ctx,
    ggml_tensor * weight,
    int64_t out_offset,
    int64_t out_size
) {
    return ggml_view_2d(
        ctx,
        weight,
        weight->ne[0],
        out_size,
        weight->nb[1],
        static_cast<size_t>(out_offset) * weight->nb[1]);
}

ggml_tensor * linear_bias_chunk(
    ggml_context * ctx,
    ggml_tensor * bias,
    int64_t out_offset,
    int64_t out_size
) {
    return ggml_view_1d(
        ctx,
        bias,
        out_size,
        static_cast<size_t>(out_offset) * ggml_element_size(bias));
}

std::vector<float> load_tensor_f32(const GgufModel & model, const std::string & name) {
    ggml_tensor * tensor = require_tensor(model, name);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> make_tiled_abs_pos(const GgufModel & model) {
    const std::vector<float> pos = load_tensor_f32(model, "backbone.vision_backbone.trunk.pos_embed");
    if (pos.size() != static_cast<size_t>((kPretrainGridSize * kPretrainGridSize + 1) * kEmbedDim)) {
        throw std::runtime_error("unexpected trunk.pos_embed size");
    }

    std::vector<float> out(static_cast<size_t>(kEmbedDim * kGridSize * kGridSize));
    for (int64_t y = 0; y < kGridSize; ++y) {
        const int64_t src_y = y % kPretrainGridSize;
        for (int64_t x = 0; x < kGridSize; ++x) {
            const int64_t src_x = x % kPretrainGridSize;
            const int64_t src_token = 1 + src_y * kPretrainGridSize + src_x;
            const int64_t dst_base = (y * kGridSize + x) * kEmbedDim;
            const int64_t src_base = src_token * kEmbedDim;
            for (int64_t c = 0; c < kEmbedDim; ++c) {
                out[static_cast<size_t>(dst_base + c)] = pos[static_cast<size_t>(src_base + c)];
            }
        }
    }

    return out;
}

std::vector<int32_t> make_axis_positions(int grid_size, bool x_axis) {
    const int64_t tokens = static_cast<int64_t>(grid_size) * static_cast<int64_t>(grid_size);
    std::vector<int32_t> out(static_cast<size_t>(tokens));
    for (int64_t idx = 0; idx < tokens; ++idx) {
        const int32_t x = static_cast<int32_t>(idx % grid_size);
        const int32_t y = static_cast<int32_t>(idx / grid_size);
        out[static_cast<size_t>(idx)] = x_axis ? x : y;
    }
    return out;
}

ggml_tensor * apply_axial_rope(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * pos_x,
    ggml_tensor * pos_y,
    float freq_scale
) {
    const int64_t half_dim = x->ne[0] / 2;
    ggml_tensor * first = ggml_view_4d(
        ctx,
        x,
        half_dim,
        x->ne[1],
        x->ne[2],
        x->ne[3],
        x->nb[1],
        x->nb[2],
        x->nb[3],
        0);
    ggml_tensor * second = ggml_view_4d(
        ctx,
        x,
        half_dim,
        x->ne[1],
        x->ne[2],
        x->ne[3],
        x->nb[1],
        x->nb[2],
        x->nb[3],
        static_cast<size_t>(half_dim) * ggml_element_size(x));

    first = ggml_rope_ext(
        ctx,
        first,
        pos_x,
        nullptr,
        half_dim,
        0,
        32768,
        kRopeTheta,
        freq_scale,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    second = ggml_rope_ext(
        ctx,
        second,
        pos_y,
        nullptr,
        half_dim,
        0,
        32768,
        kRopeTheta,
        freq_scale,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    return ggml_concat(ctx, first, second, 0);
}

ggml_tensor * apply_attention(
    ggml_context * ctx,
    ggml_tensor * q_cur,
    ggml_tensor * k_cur,
    ggml_tensor * v_cur
) {
    ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(ctx, k_cur, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx, v_cur, 0, 2, 1, 3);

    ggml_tensor * cur = ggml_flash_attn_ext(
        ctx,
        q,
        k,
        v,
        nullptr,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)),
        0.0f,
        0.0f);
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
    return ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
}

// GPU-friendly window partition: [C, W, H, 1] → [C, win, win, num_windows].
// Uses only reshape + permute + cont (GPU-native ops) to avoid CPU fallback.
// Optimized to use a single cont (memory copy) by keeping the natural
// nW_y-major batch ordering rather than transposing to nW_x-major.
// Requires W and H to be exact multiples of window_size.
ggml_tensor * gpu_win_part(
    ggml_context * ctx,
    ggml_tensor * x,
    int window_size
) {
    const int64_t C  = x->ne[0];
    const int64_t W  = x->ne[1];
    const int64_t H  = x->ne[2];
    const int64_t nW_x = W / window_size;
    const int64_t nW_y = H / window_size;

    // [C, W, H, 1] → [C, win, nW_x, H]        split width into (win, nW_x)
    ggml_tensor * cur = ggml_reshape_4d(ctx, x, C, window_size, nW_x, H);
    // → [C, win, H, nW_x]                       move H before nW_x
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 1, 3, 2));
    // → [C*win, win, nW_y, nW_x]                fold C*local_x, split H into (win, nW_y)
    cur = ggml_reshape_4d(ctx, cur, C * window_size, window_size, nW_y, nW_x);
    // → [C, win, win, nW_y*nW_x]                unfold C*local_x, flatten batch
    // Batch ordering is nW_y-major (natural from H split); attention is
    // per-window so ordering within the batch dimension doesn't matter.
    cur = ggml_reshape_4d(ctx, cur, C, window_size, window_size, nW_y * nW_x);
    return cur;
}

// GPU-friendly window unpartition: [C, win, win, num_windows] → [C, W, H, 1].
// Reverses gpu_win_part using the same nW_y-major batch convention.
// Single cont operation.
ggml_tensor * gpu_win_unpart(
    ggml_context * ctx,
    ggml_tensor * x,
    int orig_w,
    int orig_h,
    int window_size
) {
    const int64_t C    = x->ne[0];
    const int64_t nW_x = orig_w / window_size;
    const int64_t nW_y = orig_h / window_size;

    // [C, win, win, nW_y*nW_x] → [C*win, win, nW_y, nW_x]  fold C*lx, split batch
    ggml_tensor * cur = ggml_reshape_4d(ctx, x, C * window_size, window_size, nW_y, nW_x);
    // → [C, win, H, nW_x]                       unfold C*lx, merge local_y*nW_y → H
    cur = ggml_reshape_4d(ctx, cur, C, window_size, static_cast<int64_t>(window_size) * nW_y, nW_x);
    // → [C, win, nW_x, H]                       move nW_x before H
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 1, 3, 2));
    // → [C, W, H, 1]                            merge local_x*nW_x → W
    cur = ggml_reshape_4d(ctx, cur, C, static_cast<int64_t>(window_size) * nW_x, static_cast<int64_t>(window_size) * nW_y, 1);
    return cur;
}

ggml_tensor * conv2d_patch_embed(
    ggml_context * ctx,
    ggml_tensor * w,
    ggml_tensor * x
) {
    ggml_tensor * w_ggml = ggml_cont(ctx, ggml_permute(ctx, w, 2, 0, 1, 3));
    return ggml_conv_2d(ctx, w_ggml, x, kPatchSize, kPatchSize, 0, 0, 1, 1);
}

}  // namespace

VisionTrunk::VisionTrunk(const GgufModel & model) : model_(model) {}

VisionTrunkOutput VisionTrunk::run(
    const std::vector<float> & image_nchw,
    const std::vector<int64_t> & shape_nchw
) const {
    if (shape_nchw != std::vector<int64_t>({1, 3, kImageSize, kImageSize})) {
        throw std::runtime_error("vision trunk expects NCHW image shape [1, 3, 1008, 1008]");
    }
    if (static_cast<int64_t>(image_nchw.size()) != 1LL * 3 * kImageSize * kImageSize) {
        throw std::runtime_error("vision trunk input payload size mismatch");
    }

    const std::vector<float> abs_pos = make_tiled_abs_pos(model_);
    const std::vector<int32_t> rope_pos_x_local = make_axis_positions(kWindowSize, true);
    const std::vector<int32_t> rope_pos_y_local = make_axis_positions(kWindowSize, false);
    const std::vector<int32_t> rope_pos_x_global = make_axis_positions(kGridSize, true);
    const std::vector<int32_t> rope_pos_y_global = make_axis_positions(kGridSize, false);

    const size_t graph_size = 262144;
    const size_t ctx_size =
        ggml_tensor_overhead() * graph_size +
        ggml_graph_overhead_custom(graph_size, false);
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_init_params params{
        /*.mem_size =*/ ctx_size,
        /*.mem_buffer =*/ ctx_buf.data(),
        /*.no_alloc =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_backend_t backend = model_.backend();
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t backends[2] = {backend, nullptr};
    int n_backends = 1;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_backend = model_.cpu_backend();
        if (cpu_backend != nullptr) {
            backends[1] = cpu_backend;
            n_backends = 2;
        }
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
    if (sched == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to create backend scheduler");
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    ggml_tensor * image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kImageSize, kImageSize, 3, 1);
    ggml_tensor * abs_pos_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kEmbedDim, kGridSize, kGridSize, 1);
    ggml_tensor * rope_local_x = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(rope_pos_x_local.size()));
    ggml_tensor * rope_local_y = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(rope_pos_y_local.size()));
    ggml_tensor * rope_global_x = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(rope_pos_x_global.size()));
    ggml_tensor * rope_global_y = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(rope_pos_y_global.size()));

    ggml_backend_sched_set_tensor_backend(sched, image, backend);
    ggml_backend_sched_set_tensor_backend(sched, abs_pos_tensor, backend);
    ggml_backend_sched_set_tensor_backend(sched, rope_local_x, backend);
    ggml_backend_sched_set_tensor_backend(sched, rope_local_y, backend);
    ggml_backend_sched_set_tensor_backend(sched, rope_global_x, backend);
    ggml_backend_sched_set_tensor_backend(sched, rope_global_y, backend);

    ggml_tensor * cur = conv2d_patch_embed(
        ctx,
        require_tensor(model_, "backbone.vision_backbone.trunk.patch_embed.proj.weight"),
        image);
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 2, 0, 3));
    cur = ggml_add(ctx, cur, abs_pos_tensor);
    cur = layer_norm(
        ctx,
        cur,
        require_tensor(model_, "backbone.vision_backbone.trunk.ln_pre.weight"),
        require_tensor(model_, "backbone.vision_backbone.trunk.ln_pre.bias"));

    ggml_tensor * inp = cur;
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = block_prefix(layer);

        cur = layer_norm(
            ctx,
            inp,
            require_tensor(model_, prefix + ".norm1.weight"),
            require_tensor(model_, prefix + ".norm1.bias"));

        const int64_t w0 = cur->ne[1];
        const int64_t h0 = cur->ne[2];
        const bool global_attn = is_global_attention_block(layer);
        if (!global_attn) {
            cur = gpu_win_part(ctx, cur, kWindowSize);
        }

        const int64_t w = cur->ne[1];
        const int64_t h = cur->ne[2];
        const int64_t b = cur->ne[3];
        const int64_t tokens = w * h;

        ggml_tensor * qkv_w = require_tensor(model_, prefix + ".attn.qkv.weight");
        ggml_tensor * qkv_b = require_tensor(model_, prefix + ".attn.qkv.bias");

        ggml_tensor * q = linear(
            ctx,
            linear_weight_chunk(ctx, qkv_w, 0, kEmbedDim),
            linear_bias_chunk(ctx, qkv_b, 0, kEmbedDim),
            cur);
        ggml_tensor * k = linear(
            ctx,
            linear_weight_chunk(ctx, qkv_w, kEmbedDim, kEmbedDim),
            linear_bias_chunk(ctx, qkv_b, kEmbedDim, kEmbedDim),
            cur);
        ggml_tensor * v = linear(
            ctx,
            linear_weight_chunk(ctx, qkv_w, kEmbedDim * 2, kEmbedDim),
            linear_bias_chunk(ctx, qkv_b, kEmbedDim * 2, kEmbedDim),
            cur);

        q = ggml_reshape_4d(ctx, q, kHeadDim, kHeads, tokens, b);
        k = ggml_reshape_4d(ctx, k, kHeadDim, kHeads, tokens, b);
        v = ggml_reshape_4d(ctx, v, kHeadDim, kHeads, tokens, b);

        ggml_tensor * rope_x = global_attn ? rope_global_x : rope_local_x;
        ggml_tensor * rope_y = global_attn ? rope_global_y : rope_local_y;
        const float rope_scale = global_attn ? kGlobalRopeScale : 1.0f;
        q = apply_axial_rope(ctx, q, rope_x, rope_y, rope_scale);
        k = apply_axial_rope(ctx, k, rope_x, rope_y, rope_scale);

        ggml_tensor * attn = apply_attention(ctx, q, k, v);
        attn = ggml_reshape_4d(ctx, attn, kEmbedDim, w, h, b);
        attn = linear(
            ctx,
            require_tensor(model_, prefix + ".attn.proj.weight"),
            require_tensor(model_, prefix + ".attn.proj.bias"),
            attn);

        if (!global_attn) {
            attn = gpu_win_unpart(ctx, attn, static_cast<int>(w0), static_cast<int>(h0), kWindowSize);
        }

        cur = ggml_add(ctx, inp, attn);

        ggml_tensor * ffn = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".norm2.weight"),
            require_tensor(model_, prefix + ".norm2.bias"));
        ffn = linear(
            ctx,
            require_tensor(model_, prefix + ".mlp.fc1.weight"),
            require_tensor(model_, prefix + ".mlp.fc1.bias"),
            ffn);
        ffn = ggml_gelu_erf(ctx, ffn);
        ffn = linear(
            ctx,
            require_tensor(model_, prefix + ".mlp.fc2.weight"),
            require_tensor(model_, prefix + ".mlp.fc2.bias"),
            ffn);

        inp = ggml_add(ctx, cur, ffn);
    }

    ggml_tensor * trunk = ggml_cont(ctx, ggml_permute(ctx, inp, 2, 0, 1, 3));
    ggml_build_forward_expand(gf, trunk);

    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(image, image_nchw.data(), 0, image_nchw.size() * sizeof(float));
    ggml_backend_tensor_set(abs_pos_tensor, abs_pos.data(), 0, abs_pos.size() * sizeof(float));
    ggml_backend_tensor_set(rope_local_x, rope_pos_x_local.data(), 0, rope_pos_x_local.size() * sizeof(int32_t));
    ggml_backend_tensor_set(rope_local_y, rope_pos_y_local.data(), 0, rope_pos_y_local.size() * sizeof(int32_t));
    ggml_backend_tensor_set(rope_global_x, rope_pos_x_global.data(), 0, rope_pos_x_global.size() * sizeof(int32_t));
    ggml_backend_tensor_set(rope_global_y, rope_pos_y_global.data(), 0, rope_pos_y_global.size() * sizeof(int32_t));

    const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        throw std::runtime_error("vision trunk graph compute failed");
    }

    VisionTrunkOutput out;
    out.shape_nchw = {trunk->ne[3], trunk->ne[2], trunk->ne[1], trunk->ne[0]};
    out.trunk_nchw.resize(static_cast<size_t>(ggml_nelements(trunk)));
    ggml_backend_tensor_get(trunk, out.trunk_nchw.data(), 0, out.trunk_nchw.size() * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);

    return out;
}

}  // namespace sam3
