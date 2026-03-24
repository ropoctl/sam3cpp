#include "sam3/vision_neck.h"

#include "ggml-alloc.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * tensor = model.find_weight(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return tensor;
}

// Add bias along the channel dimension.
// spatial_first = true  → x is [W, H, C, 1], bias broadcast as [1, 1, C, 1]
// spatial_first = false → x is [C, W, H, 1], bias broadcast as [C, 1, 1, 1]
ggml_tensor * add_bias_1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias, int64_t channels,
                          bool spatial_first = true) {
    ggml_tensor * b = bias;
    if (b->type != GGML_TYPE_F32) {
        b = ggml_cast(ctx, b, GGML_TYPE_F32);
    }
    if (spatial_first) {
        b = ggml_reshape_4d(ctx, b, 1, 1, channels, 1);
    } else {
        b = ggml_reshape_4d(ctx, b, channels, 1, 1, 1);
    }
    return ggml_add(ctx, x, b);
}

ggml_tensor * conv2d_bias(
    ggml_context * ctx,
    ggml_tensor * w,
    ggml_tensor * b,
    ggml_tensor * x,
    int s0,
    int s1,
    int p0,
    int p1
) {
    // MLX stores Conv2d weights as [OC, KH, KW, IC], which lands in ggml as
    // ne = [IC, KW, KH, OC]. ggml_conv_2d expects ne = [KW, KH, IC, OC].
    ggml_tensor * w_ggml = ggml_cont(ctx, ggml_permute(ctx, w, 2, 0, 1, 3));
    ggml_tensor * y = ggml_conv_2d(ctx, w_ggml, x, s0, s1, p0, p1, 1, 1);
    return add_bias_1d(ctx, y, b, w_ggml->ne[3]);
}

// GPU-friendly transposed conv for kernel=2x2, stride=2.
// Decompose into matmul (IC → OC*4) + pixel shuffle (depth-to-space).
//
// Weight w:   [IC, kw=2, kh=2, OC] — ggml layout of MLX [OC, KH, KW, IC].
// Input x:    [W, H, IC, 1] if input_ch_first == false (spatial-first, ggml default)
//             [IC, W, H, 1] if input_ch_first == true  (channel-first)
// Output:     [OC, W*2, H*2, 1] (channel-first)
//
// The channel-first output avoids one GPU memory copy (permute+cont) per call.
// Callers that need spatial-first [W*2, H*2, OC, 1] should do one explicit
// cont(permute) after the last deconv in a chain.
ggml_tensor * deconv2d_bias(
    ggml_context * ctx,
    ggml_tensor * w,
    ggml_tensor * b,
    ggml_tensor * x,
    int stride,
    bool input_ch_first = false
) {
    const int64_t IC = w->ne[0];
    const int64_t OC = w->ne[3];
    const int64_t S  = static_cast<int64_t>(stride);

    int64_t W, H;
    if (input_ch_first) {
        // x is [IC, W, H, 1]
        W = x->ne[1];
        H = x->ne[2];
    } else {
        // x is [W, H, IC, 1]
        W = x->ne[0];
        H = x->ne[1];
    }

    // Weight: w.ne = [IC, kw, kh, OC]. Permute to [IC, OC, kw, kh] so that when
    // flattened to [IC, OC*kw*kh], OC varies fastest for the pixel shuffle.
    ggml_tensor * w_p = ggml_cont(ctx, ggml_permute(ctx, w, 0, 2, 3, 1));
    ggml_tensor * w_flat = ggml_reshape_2d(ctx, w_p, IC, OC * S * S);
    // w_flat: [IC, OC*4] where idx = oc + kw*OC + kh*OC*S

    // Prepare input as [IC, W*H]:
    ggml_tensor * x_2d;
    if (input_ch_first) {
        // x is [IC, W, H, 1] — already channel-first, just reshape
        x_2d = ggml_reshape_2d(ctx, x, IC, W * H);
    } else {
        // x is [W, H, IC, 1] — permute to [IC, W, H, 1], then reshape
        x_2d = ggml_reshape_2d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)),
            IC, W * H);
    }

    // matmul: w_flat[IC, OC*4] @ x_2d[IC, W*H] → y[OC*4, W*H]
    ggml_tensor * y = ggml_mul_mat(ctx, w_flat, x_2d);

    // Pixel shuffle: y[OC*4, W*H] → [OC, W*S, H*S, 1]  (channel-first output)
    //
    // Matmul output index: y[oc + kw*OC + kh*OC*S, w + h*W]
    // Target:              out[oc, w*S+kw, h*S+kh, 0]
    //
    // Reshape to [OC*S, S, W, H] = [OC_kw, kh, W, H]
    // where OC_kw = oc + kw*OC (oc fastest), kh in ne[1].
    y = ggml_reshape_4d(ctx, y, OC * S, S, W, H);

    // Interleave kh with H: permute to [OC_kw, W, kh, H], then merge [kh, H]
    // into h_out = kh + h*S (kh varies fastest in the merged dim).
    y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3)); // [OC_kw, W, kh, H]
    y = ggml_reshape_4d(ctx, y, OC * S, W, S * H, 1);     // [OC_kw, W, H_out, 1]

    // Split OC_kw = [OC, kw], then merge [kw, W] into w_out = kw + w*S.
    y = ggml_reshape_4d(ctx, y, OC, S, W, S * H);         // [OC, kw, W, H_out]
    y = ggml_reshape_4d(ctx, y, OC, S * W, S * H, 1);     // [OC, W_out, H_out, 1]

    return add_bias_1d(ctx, y, b, OC, /*spatial_first=*/false);
}

std::vector<float> nchw_to_whcn(const std::vector<float> & input, int64_t n, int64_t c, int64_t h, int64_t w) {
    (void) n;
    (void) c;
    (void) h;
    (void) w;
    // NCHW contiguous storage is byte-compatible with a ggml tensor created as
    // ggml_new_tensor_4d(..., W, H, C, N). No data reorder is required.
    return input;
}

std::vector<float> whcn_to_nchw(const std::vector<float> & input, int64_t w, int64_t h, int64_t c, int64_t n) {
    (void) w;
    (void) h;
    (void) c;
    (void) n;
    return input;
}

std::vector<float> make_position_encoding_nchw(int64_t n, int64_t c, int64_t h, int64_t w) {
    if (c % 4 != 0) {
        throw std::runtime_error("position encoding expects channels divisible by 4");
    }

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kScale = 2.0f * kPi;
    constexpr float kTemperature = 10000.0f;
    constexpr float kEps = 1e-6f;

    const int64_t num_pos_feats = c / 2;
    const int64_t half_pairs = num_pos_feats / 2;

    std::vector<float> dim_t(static_cast<size_t>(num_pos_feats));
    for (int64_t i = 0; i < num_pos_feats; ++i) {
        const float exponent = 2.0f * static_cast<float>(i / 2) / static_cast<float>(num_pos_feats);
        dim_t[static_cast<size_t>(i)] = std::pow(kTemperature, exponent);
    }

    std::vector<float> out(static_cast<size_t>(n * c * h * w));
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t yi = 0; yi < h; ++yi) {
            const float y_embed = (static_cast<float>(yi + 1) / (static_cast<float>(h) + kEps)) * kScale;
            for (int64_t xi = 0; xi < w; ++xi) {
                const float x_embed = (static_cast<float>(xi + 1) / (static_cast<float>(w) + kEps)) * kScale;

                for (int64_t pair = 0; pair < half_pairs; ++pair) {
                    const int64_t even_idx = pair * 2;
                    const int64_t odd_idx = even_idx + 1;
                    const float y_scaled = y_embed / dim_t[static_cast<size_t>(even_idx)];
                    const float x_scaled = x_embed / dim_t[static_cast<size_t>(even_idx)];

                    const int64_t cy_sin = even_idx;
                    const int64_t cy_cos = odd_idx;
                    const int64_t cx_sin = num_pos_feats + even_idx;
                    const int64_t cx_cos = num_pos_feats + odd_idx;

                    out[static_cast<size_t>(((ni * c + cy_sin) * h + yi) * w + xi)] = std::sin(y_scaled);
                    out[static_cast<size_t>(((ni * c + cy_cos) * h + yi) * w + xi)] = std::cos(y_scaled);
                    out[static_cast<size_t>(((ni * c + cx_sin) * h + yi) * w + xi)] = std::sin(x_scaled);
                    out[static_cast<size_t>(((ni * c + cx_cos) * h + yi) * w + xi)] = std::cos(x_scaled);
                }
            }
        }
    }

    return out;
}

}  // namespace

VisionNeck::VisionNeck(const GgufModel & model) : model_(model) {}

VisionNeckOutput VisionNeck::run(
    const std::vector<float> & trunk_nchw,
    const std::vector<int64_t> & shape_nchw,
    std::optional<int> only_level
) const {
    if (shape_nchw.size() != 4) {
        throw std::runtime_error("expected trunk input shape NCHW");
    }
    if (only_level.has_value() && (*only_level < 0 || *only_level > 3)) {
        throw std::runtime_error("only_level must be in [0, 3]");
    }
    const int64_t n = shape_nchw[0];
    const int64_t c = shape_nchw[1];
    const int64_t h = shape_nchw[2];
    const int64_t w = shape_nchw[3];

    const size_t graph_size = 8192;
    const size_t ctx_size =
        ggml_tensor_overhead() * graph_size +
        ggml_graph_overhead_custom(graph_size, false);
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_init_params params {
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
    ggml_backend_t backends[2] = { backend, nullptr };
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
    ggml_tensor * trunk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, c, n);
    ggml_backend_sched_set_tensor_backend(sched, trunk, backend);

    std::vector<ggml_tensor *> outputs;
    std::vector<std::vector<int64_t>> out_shapes;

    if (!only_level.has_value() || *only_level == 0) {
        // deconv0: trunk [W,H,IC,1] → channel-first [OC,W*2,H*2,1]
        ggml_tensor * y0 = deconv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.0.dconv_2x2_0.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.0.dconv_2x2_0.bias"),
            trunk,
            2, /*input_ch_first=*/false);
        y0 = ggml_gelu_erf(ctx, y0);  // element-wise, layout-agnostic
        // deconv1: channel-first [IC,W,H,1] → channel-first [OC,W*2,H*2,1]
        y0 = deconv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.0.dconv_2x2_1.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.0.dconv_2x2_1.bias"),
            y0,
            2, /*input_ch_first=*/true);
        // Permute from channel-first [OC,W,H,1] to spatial-first [W,H,OC,1] for conv2d
        y0 = ggml_cont(ctx, ggml_permute(ctx, y0, 2, 0, 1, 3));
        y0 = conv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.0.conv_1x1.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.0.conv_1x1.bias"),
            y0,
            1, 1, 0, 0);
        y0 = conv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.0.conv_3x3.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.0.conv_3x3.bias"),
            y0,
            1, 1, 1, 1);
        outputs.push_back(y0);
        out_shapes.push_back({n, y0->ne[2], y0->ne[1], y0->ne[0]});
    }

    if (!only_level.has_value() || *only_level == 1) {
        // deconv: trunk [W,H,IC,1] → channel-first [OC,W*2,H*2,1]
        ggml_tensor * y1 = deconv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.1.dconv_2x2.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.1.dconv_2x2.bias"),
            trunk,
            2, /*input_ch_first=*/false);
        // Permute from channel-first [OC,W,H,1] to spatial-first [W,H,OC,1] for conv2d
        y1 = ggml_cont(ctx, ggml_permute(ctx, y1, 2, 0, 1, 3));
        y1 = conv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.1.conv_1x1.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.1.conv_1x1.bias"),
            y1,
            1, 1, 0, 0);
        y1 = conv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.1.conv_3x3.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.1.conv_3x3.bias"),
            y1,
            1, 1, 1, 1);
        outputs.push_back(y1);
        out_shapes.push_back({n, y1->ne[2], y1->ne[1], y1->ne[0]});
    }

    if (!only_level.has_value() || *only_level == 2) {
        ggml_tensor * y2 = conv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.2.conv_1x1.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.2.conv_1x1.bias"),
            trunk,
            1, 1, 0, 0);
        y2 = conv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.2.conv_3x3.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.2.conv_3x3.bias"),
            y2,
            1, 1, 1, 1);
        outputs.push_back(y2);
        out_shapes.push_back({n, y2->ne[2], y2->ne[1], y2->ne[0]});
    }

    if (!only_level.has_value() || *only_level == 3) {
        ggml_tensor * y3 = ggml_pool_2d(ctx, trunk, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        y3 = conv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.3.conv_1x1.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.3.conv_1x1.bias"),
            y3,
            1, 1, 0, 0);
        y3 = conv2d_bias(
            ctx,
            require_tensor(model_, "backbone.vision_backbone.convs.3.conv_3x3.weight"),
            require_tensor(model_, "backbone.vision_backbone.convs.3.conv_3x3.bias"),
            y3,
            1, 1, 1, 1);
        outputs.push_back(y3);
        out_shapes.push_back({n, y3->ne[2], y3->ne[1], y3->ne[0]});
    }

    for (ggml_tensor * out : outputs) {
        ggml_build_forward_expand(gf, out);
    }

    ggml_backend_sched_alloc_graph(sched, gf);
    const std::vector<float> whcn = nchw_to_whcn(trunk_nchw, n, c, h, w);
    ggml_backend_tensor_set(trunk, whcn.data(), 0, whcn.size() * sizeof(float));

    const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        throw std::runtime_error("vision neck graph compute failed");
    }

    VisionNeckOutput result;
    result.shapes = out_shapes;
    result.levels.resize(outputs.size());
    result.position_shapes = out_shapes;
    result.positions.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::vector<float> tmp(static_cast<size_t>(ggml_nelements(outputs[i])));
        ggml_backend_tensor_get(outputs[i], tmp.data(), 0, ggml_nbytes(outputs[i]));
        result.levels[i] = whcn_to_nchw(tmp, outputs[i]->ne[0], outputs[i]->ne[1], outputs[i]->ne[2], outputs[i]->ne[3]);
        result.positions[i] = make_position_encoding_nchw(
            out_shapes[i][0],
            out_shapes[i][1],
            out_shapes[i][2],
            out_shapes[i][3]);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

}  // namespace sam3
