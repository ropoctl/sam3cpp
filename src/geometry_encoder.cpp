#include "sam3/geometry_encoder.h"

#include "ggml-alloc.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kModelDim = 256;
constexpr int32_t kHeads = 8;
constexpr int32_t kHeadDim = kModelDim / kHeads;
constexpr int32_t kLayers = 3;
constexpr float kLayerNormEps = 1e-5f;

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

ggml_tensor * layer_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias
) {
    ggml_tensor * y = ggml_norm(ctx, ensure_f32(ctx, x), kLayerNormEps);
    y = ggml_mul(ctx, y, ensure_f32(ctx, weight));
    y = ggml_add(ctx, y, ensure_f32(ctx, bias));
    return y;
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
        y = ggml_add(ctx, y, ensure_f32(ctx, bias));
    }
    return y;
}

ggml_tensor * mha(
    ggml_context * ctx,
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    int32_t q_len,
    int32_t kv_len,
    ggml_tensor * mask
) {
    ggml_tensor * qh = ggml_permute(ctx, ggml_cont_3d(ctx, q, kHeadDim, kHeads, q_len), 0, 2, 1, 3);
    ggml_tensor * kh = ggml_permute(ctx, ggml_cont_3d(ctx, k, kHeadDim, kHeads, kv_len), 0, 2, 1, 3);
    ggml_tensor * vh = ggml_cont_3d(
        ctx,
        ggml_permute(ctx, ggml_cont_3d(ctx, v, kHeadDim, kHeads, kv_len), 1, 2, 0, 3),
        kv_len,
        kHeadDim,
        kHeads
    );

    ggml_tensor * scores = ensure_f32(ctx, ggml_mul_mat(ctx, kh, qh));
    ggml_tensor * probs = ggml_soft_max_ext(
        ctx,
        scores,
        mask,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)),
        0.0f
    );
    ggml_tensor * attn = ensure_f32(ctx, ggml_mul_mat(ctx, vh, probs));
    ggml_tensor * merged = ggml_permute(ctx, attn, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, merged, kModelDim, q_len);
}

std::vector<float> flatten_nchw_to_seq(
    const std::vector<float> & input,
    int64_t c,
    int64_t h,
    int64_t w
) {
    std::vector<float> out(static_cast<size_t>(c * h * w));
    for (int64_t yi = 0; yi < h; ++yi) {
        for (int64_t xi = 0; xi < w; ++xi) {
            const int64_t seq = yi * w + xi;
            for (int64_t ci = 0; ci < c; ++ci) {
                const size_t src = static_cast<size_t>(((ci * h) + yi) * w + xi);
                const size_t dst = static_cast<size_t>(seq * c + ci);
                out[dst] = input[src];
            }
        }
    }
    return out;
}

std::string layer_prefix(int layer) {
    return "geometry_encoder.encode." + std::to_string(layer);
}

// Sinusoidal position encoding for a single coordinate value.
// Returns num_pos_feats floats: [sin(x/t0), cos(x/t0), sin(x/t1), cos(x/t1), ...]
// interleaved as [sin_0, sin_1, ...sin_{n/2-1}, cos_0, cos_1, ...cos_{n/2-1}]
// after the stack+flatten in the upstream code.
constexpr int32_t kNumPosFeats = 128;
constexpr float kPosEncTemperature = 10000.0f;
constexpr float kPosEncScale = 2.0f * 3.14159265358979323846f;

void encode_1d_sinusoidal(float coord, float * out) {
    const float scaled = coord * kPosEncScale;
    // dim_t[i] = temperature^(2*(i//2)/num_pos_feats)
    // pos = scaled / dim_t[i]
    // output: stack(sin(pos[0::2]), cos(pos[1::2])).flatten
    // In practice: sin for even indices, cos for odd, then interleave sin/cos pairs
    // The upstream does: sin(pos[:,0::2]), cos(pos[:,1::2]) then stack+flatten
    // Which produces: [sin_0, cos_0, sin_1, cos_1, ...]? No:
    // stack((sin_half, cos_half), axis=2).flatten(1)
    // sin_half = [sin(pos[0]), sin(pos[2]), ...] (even indices only)
    // cos_half = [cos(pos[1]), cos(pos[3]), ...] (odd indices only)
    // stack axis=2 on shape [N, half]: [[sin0,cos0],[sin1,cos1],...]
    // flatten: [sin0, cos0, sin1, cos1, ...]
    const int32_t half = kNumPosFeats / 2;
    for (int32_t i = 0; i < half; ++i) {
        const float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(kNumPosFeats);
        const float dim_t_even = std::pow(kPosEncTemperature, exponent);
        const float dim_t_odd = std::pow(kPosEncTemperature, exponent);  // same exponent for 2*(i//2) when i is even or odd pair
        const float pos_even = scaled / dim_t_even;
        const float pos_odd = scaled / dim_t_odd;
        out[i * 2 + 0] = std::sin(pos_even);
        out[i * 2 + 1] = std::cos(pos_odd);
    }
}

// Encode points: returns [point_count, 256] where each row is concat(pos_y, pos_x)
// (upstream uses (pos_y, pos_x) order for points)
std::vector<float> encode_points_pos(const float * points_xy, int32_t point_count) {
    std::vector<float> out(static_cast<size_t>(point_count) * kNumPosFeats * 2);
    for (int32_t i = 0; i < point_count; ++i) {
        float * row = out.data() + static_cast<size_t>(i) * kNumPosFeats * 2;
        encode_1d_sinusoidal(points_xy[i * 2 + 1], row);                    // pos_y first
        encode_1d_sinusoidal(points_xy[i * 2 + 0], row + kNumPosFeats);     // then pos_x
    }
    return out;
}

// Encode boxes: returns [box_count, 258] where each row is concat(pos_y, pos_x, h, w)
std::vector<float> encode_boxes_pos(const float * boxes_cxcywh, int32_t box_count) {
    std::vector<float> out(static_cast<size_t>(box_count) * (kNumPosFeats * 2 + 2));
    for (int32_t i = 0; i < box_count; ++i) {
        float * row = out.data() + static_cast<size_t>(i) * (kNumPosFeats * 2 + 2);
        const float cx = boxes_cxcywh[i * 4 + 0];
        const float cy = boxes_cxcywh[i * 4 + 1];
        const float w = boxes_cxcywh[i * 4 + 2];
        const float h = boxes_cxcywh[i * 4 + 3];
        encode_1d_sinusoidal(cy, row);                                       // pos_y first
        encode_1d_sinusoidal(cx, row + kNumPosFeats);                        // then pos_x
        row[kNumPosFeats * 2 + 0] = h;
        row[kNumPosFeats * 2 + 1] = w;
    }
    return out;
}

// Bilinear sample from NCHW feature map at normalized (x, y) in [0, 1].
// Returns C-dim vector.
void bilinear_sample_nchw(
    const float * nchw, int64_t c, int64_t h, int64_t w,
    float x_norm, float y_norm,
    float * out
) {
    const float fx = x_norm * static_cast<float>(w) - 0.5f;
    const float fy = y_norm * static_cast<float>(h) - 0.5f;
    const int x0 = std::max(0, std::min(static_cast<int>(std::floor(fx)), static_cast<int>(w) - 1));
    const int x1 = std::min(x0 + 1, static_cast<int>(w) - 1);
    const int y0 = std::max(0, std::min(static_cast<int>(std::floor(fy)), static_cast<int>(h) - 1));
    const int y1 = std::min(y0 + 1, static_cast<int>(h) - 1);
    const float wx = fx - static_cast<float>(x0);
    const float wy = fy - static_cast<float>(y0);

    for (int64_t ci = 0; ci < c; ++ci) {
        const float * plane = nchw + ci * h * w;
        const float v00 = plane[y0 * w + x0];
        const float v01 = plane[y0 * w + x1];
        const float v10 = plane[y1 * w + x0];
        const float v11 = plane[y1 * w + x1];
        const float top = v00 + (v01 - v00) * wx;
        const float bot = v10 + (v11 - v10) * wx;
        out[ci] = top + (bot - top) * wy;
    }
}

// Apply layer norm on CPU to NCHW feature map along channel dim.
// Input/output: [C, H, W] (N=1 assumed)
void cpu_layer_norm_nchw(
    const float * input, float * output,
    const float * weight, const float * bias,
    int64_t c, int64_t h, int64_t w, float eps
) {
    const int64_t spatial = h * w;
    for (int64_t s = 0; s < spatial; ++s) {
        // Gather channel values at this spatial position
        float mean = 0.0f;
        for (int64_t ci = 0; ci < c; ++ci) {
            mean += input[ci * spatial + s];
        }
        mean /= static_cast<float>(c);

        float var = 0.0f;
        for (int64_t ci = 0; ci < c; ++ci) {
            const float d = input[ci * spatial + s] - mean;
            var += d * d;
        }
        var /= static_cast<float>(c);
        const float inv_std = 1.0f / std::sqrt(var + eps);

        for (int64_t ci = 0; ci < c; ++ci) {
            const float normed = (input[ci * spatial + s] - mean) * inv_std;
            output[ci * spatial + s] = normed * weight[ci] + bias[ci];
        }
    }
}

}  // namespace

GeometryEncoder::GeometryEncoder(const GgufModel & model) : model_(model) {}

GeometryEncoderOutput GeometryEncoder::run(
    const float * points_xy, const int32_t * labels, int32_t point_count,
    const float * boxes_cxcywh, const int32_t * box_labels, int32_t box_count,
    const std::vector<float> & image_nchw,
    const std::vector<int64_t> & image_shape_nchw,
    const std::vector<float> & pos_nchw,
    const std::vector<int64_t> & pos_shape_nchw
) const {
    if (point_count <= 0 && box_count <= 0) {
        throw std::runtime_error("geometry encoder requires at least one point or box");
    }
    if (image_shape_nchw.size() != 4 || image_shape_nchw[0] != 1 || image_shape_nchw[1] != kModelDim) {
        throw std::runtime_error("expected image shape [1, 256, H, W]");
    }
    if (pos_shape_nchw != image_shape_nchw) {
        throw std::runtime_error("pos shape must match image shape");
    }

    const int64_t h = image_shape_nchw[2];
    const int64_t w = image_shape_nchw[3];
    const int32_t image_seq_len = static_cast<int32_t>(h * w);
    const int32_t geo_token_count = point_count + box_count + 1;  // +1 for CLS

    const std::vector<float> image_seq = flatten_nchw_to_seq(image_nchw, kModelDim, h, w);
    const std::vector<float> pos_seq = flatten_nchw_to_seq(pos_nchw, kModelDim, h, w);

    // Build the initial embedding sequence on CPU:
    // For each point: direct_project([x, y]) + label_embed(label)
    // For each box:   direct_project([cx, cy, w, h]) + label_embed(label)
    // Then CLS token appended.
    // We'll feed this as an input tensor and do the projections in the ggml graph.

    // Prepare raw coordinate data: [point_count, 2] and [box_count, 4]
    // and label indices: [point_count] and [box_count]
    std::vector<float> point_coords(static_cast<size_t>(point_count) * 2);
    std::vector<int32_t> point_labels(static_cast<size_t>(point_count));
    for (int32_t i = 0; i < point_count; ++i) {
        point_coords[static_cast<size_t>(i) * 2 + 0] = points_xy[i * 2 + 0];
        point_coords[static_cast<size_t>(i) * 2 + 1] = points_xy[i * 2 + 1];
        point_labels[static_cast<size_t>(i)] = labels[i];
    }

    std::vector<float> box_coords(static_cast<size_t>(box_count) * 4);
    std::vector<int32_t> box_label_vec(static_cast<size_t>(box_count));
    for (int32_t i = 0; i < box_count; ++i) {
        box_coords[static_cast<size_t>(i) * 4 + 0] = boxes_cxcywh[i * 4 + 0];
        box_coords[static_cast<size_t>(i) * 4 + 1] = boxes_cxcywh[i * 4 + 1];
        box_coords[static_cast<size_t>(i) * 4 + 2] = boxes_cxcywh[i * 4 + 2];
        box_coords[static_cast<size_t>(i) * 4 + 3] = boxes_cxcywh[i * 4 + 3];
        box_label_vec[static_cast<size_t>(i)] = box_labels[i];
    }

    const size_t graph_size = 32768;
    const size_t ctx_size =
        ggml_tensor_overhead() * graph_size +
        ggml_graph_overhead_custom(graph_size, false);
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_context * ctx = ggml_init({
        /*.mem_size =*/ ctx_size,
        /*.mem_buffer =*/ ctx_buf.data(),
        /*.no_alloc =*/ true,
    });
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

    // Input tensors
    ggml_tensor * image = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, image_seq_len);
    ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, image_seq_len);
    ggml_backend_sched_set_tensor_backend(sched, image, backend);
    ggml_backend_sched_set_tensor_backend(sched, pos, backend);

    // Precompute sinusoidal position encodings on CPU
    std::vector<float> point_pos_enc_data;
    std::vector<float> box_pos_enc_data;
    if (point_count > 0) {
        point_pos_enc_data = encode_points_pos(points_xy, point_count);
    }
    if (box_count > 0) {
        box_pos_enc_data = encode_boxes_pos(boxes_cxcywh, box_count);
    }

    // Pool encoding: bilinear sample from layer-normed image features at point/box locations.
    // Done on CPU since it's just a few lookups.
    std::vector<float> point_pool_data;
    std::vector<float> normed_image;
    if (point_count > 0) {
        // Read img_pre_norm weights
        ggml_tensor * norm_w = require_tensor(model_, "geometry_encoder.img_pre_norm.weight");
        ggml_tensor * norm_b = require_tensor(model_, "geometry_encoder.img_pre_norm.bias");
        std::vector<float> nw(static_cast<size_t>(kModelDim));
        std::vector<float> nb(static_cast<size_t>(kModelDim));
        ggml_backend_tensor_get(norm_w, nw.data(), 0, nw.size() * sizeof(float));
        ggml_backend_tensor_get(norm_b, nb.data(), 0, nb.size() * sizeof(float));

        // Apply layer norm to image NCHW features
        normed_image.resize(image_nchw.size());
        cpu_layer_norm_nchw(image_nchw.data(), normed_image.data(),
            nw.data(), nb.data(), kModelDim, h, w, kLayerNormEps);

        // Bilinear sample at each point
        point_pool_data.resize(static_cast<size_t>(point_count) * kModelDim);
        for (int32_t i = 0; i < point_count; ++i) {
            bilinear_sample_nchw(
                normed_image.data(), kModelDim, h, w,
                points_xy[i * 2 + 0], points_xy[i * 2 + 1],
                point_pool_data.data() + static_cast<size_t>(i) * kModelDim);
        }
    }

    // Build point embeddings: direct_project + pos_enc_project + pool_project + label_embed
    ggml_tensor * point_embeds = nullptr;
    ggml_tensor * point_coords_t = nullptr;
    ggml_tensor * point_labels_t = nullptr;
    ggml_tensor * point_pos_enc_t = nullptr;
    ggml_tensor * point_pool_t = nullptr;
    if (point_count > 0) {
        point_coords_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, point_count);
        point_labels_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, point_count);
        point_pos_enc_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kNumPosFeats * 2, point_count);
        point_pool_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, point_count);
        ggml_backend_sched_set_tensor_backend(sched, point_coords_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, point_labels_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, point_pos_enc_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, point_pool_t, backend);

        // direct_project
        ggml_tensor * proj = linear(
            ctx,
            require_tensor(model_, "geometry_encoder.points_direct_project.weight"),
            require_tensor(model_, "geometry_encoder.points_direct_project.bias"),
            point_coords_t
        );
        // pos_enc_project
        ggml_tensor * pos_proj = linear(
            ctx,
            require_tensor(model_, "geometry_encoder.points_pos_enc_project.weight"),
            require_tensor(model_, "geometry_encoder.points_pos_enc_project.bias"),
            point_pos_enc_t
        );
        proj = ggml_add(ctx, proj, pos_proj);
        // pool_project (bilinear-sampled image features)
        ggml_tensor * pool_proj = linear(
            ctx,
            require_tensor(model_, "geometry_encoder.points_pool_project.weight"),
            require_tensor(model_, "geometry_encoder.points_pool_project.bias"),
            point_pool_t
        );
        proj = ggml_add(ctx, proj, pool_proj);

        ggml_tensor * label_emb = ggml_get_rows(ctx,
            ensure_f32(ctx, require_tensor(model_, "geometry_encoder.label_embed.weight")),
            point_labels_t);
        point_embeds = ggml_add(ctx, proj, label_emb);
    }

    // Build box embeddings: direct_project(coords) + pos_enc_project(sinusoidal) + label_embed(label)
    ggml_tensor * box_embeds = nullptr;
    ggml_tensor * box_coords_t = nullptr;
    ggml_tensor * box_labels_t = nullptr;
    ggml_tensor * box_pos_enc_t = nullptr;
    if (box_count > 0) {
        box_coords_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, box_count);
        box_labels_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, box_count);
        box_pos_enc_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kNumPosFeats * 2 + 2, box_count);
        ggml_backend_sched_set_tensor_backend(sched, box_coords_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, box_labels_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, box_pos_enc_t, backend);

        // direct_project
        ggml_tensor * proj = linear(
            ctx,
            require_tensor(model_, "geometry_encoder.boxes_direct_project.weight"),
            require_tensor(model_, "geometry_encoder.boxes_direct_project.bias"),
            box_coords_t
        );
        // pos_enc_project
        ggml_tensor * pos_proj = linear(
            ctx,
            require_tensor(model_, "geometry_encoder.boxes_pos_enc_project.weight"),
            require_tensor(model_, "geometry_encoder.boxes_pos_enc_project.bias"),
            box_pos_enc_t
        );
        proj = ggml_add(ctx, proj, pos_proj);

        ggml_tensor * label_emb = ggml_get_rows(ctx,
            ensure_f32(ctx, require_tensor(model_, "geometry_encoder.label_embed.weight")),
            box_labels_t);
        box_embeds = ggml_add(ctx, proj, label_emb);
    }

    // Concatenate: [point_embeds; box_embeds; cls_embed] -> [kModelDim, geo_token_count]
    ggml_tensor * cls = ggml_view_2d(ctx,
        require_tensor(model_, "geometry_encoder.cls_embed.weight"),
        kModelDim, 1,
        require_tensor(model_, "geometry_encoder.cls_embed.weight")->nb[1], 0);

    // Build the concatenated sequence
    ggml_tensor * cur = nullptr;
    if (point_embeds != nullptr && box_embeds != nullptr) {
        cur = ggml_concat(ctx, ggml_concat(ctx, point_embeds, box_embeds, 1), cls, 1);
    } else if (point_embeds != nullptr) {
        cur = ggml_concat(ctx, point_embeds, cls, 1);
    } else {
        cur = ggml_concat(ctx, box_embeds, cls, 1);
    }

    // final_proj + norm (same as dummy_prompt)
    cur = linear(
        ctx,
        require_tensor(model_, "geometry_encoder.final_proj.weight"),
        require_tensor(model_, "geometry_encoder.final_proj.bias"),
        cur
    );
    cur = layer_norm(
        ctx,
        cur,
        require_tensor(model_, "geometry_encoder.norm.weight"),
        require_tensor(model_, "geometry_encoder.norm.bias")
    );

    // 3-layer transformer: self-attn + cross-attn-to-image + FFN (identical to dummy_prompt)
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = layer_prefix(layer);

        // Self-attention
        ggml_tensor * norm1 = layer_norm(
            ctx, cur,
            require_tensor(model_, prefix + ".norm1.weight"),
            require_tensor(model_, prefix + ".norm1.bias")
        );
        ggml_tensor * self_out = mha(
            ctx,
            linear(ctx, require_tensor(model_, prefix + ".self_attn.query_proj.weight"), require_tensor(model_, prefix + ".self_attn.query_proj.bias"), norm1),
            linear(ctx, require_tensor(model_, prefix + ".self_attn.key_proj.weight"), require_tensor(model_, prefix + ".self_attn.key_proj.bias"), norm1),
            linear(ctx, require_tensor(model_, prefix + ".self_attn.value_proj.weight"), require_tensor(model_, prefix + ".self_attn.value_proj.bias"), norm1),
            geo_token_count,
            geo_token_count,
            nullptr
        );
        self_out = linear(ctx, require_tensor(model_, prefix + ".self_attn.out_proj.weight"), require_tensor(model_, prefix + ".self_attn.out_proj.bias"), self_out);
        cur = ggml_add(ctx, cur, self_out);

        // Cross-attention to image
        ggml_tensor * norm2 = layer_norm(
            ctx, cur,
            require_tensor(model_, prefix + ".norm2.weight"),
            require_tensor(model_, prefix + ".norm2.bias")
        );
        ggml_tensor * cross_out = mha(
            ctx,
            linear(ctx, require_tensor(model_, prefix + ".cross_attn_image.query_proj.weight"), require_tensor(model_, prefix + ".cross_attn_image.query_proj.bias"), norm2),
            linear(ctx, require_tensor(model_, prefix + ".cross_attn_image.key_proj.weight"), require_tensor(model_, prefix + ".cross_attn_image.key_proj.bias"), ggml_add(ctx, image, pos)),
            linear(ctx, require_tensor(model_, prefix + ".cross_attn_image.value_proj.weight"), require_tensor(model_, prefix + ".cross_attn_image.value_proj.bias"), image),
            geo_token_count,
            image_seq_len,
            nullptr
        );
        cross_out = linear(ctx, require_tensor(model_, prefix + ".cross_attn_image.out_proj.weight"), require_tensor(model_, prefix + ".cross_attn_image.out_proj.bias"), cross_out);
        cur = ggml_add(ctx, cur, cross_out);

        // FFN
        ggml_tensor * norm3 = layer_norm(
            ctx, cur,
            require_tensor(model_, prefix + ".norm3.weight"),
            require_tensor(model_, prefix + ".norm3.bias")
        );
        ggml_tensor * ffn = linear(ctx, require_tensor(model_, prefix + ".linear1.weight"), require_tensor(model_, prefix + ".linear1.bias"), norm3);
        ffn = ggml_relu(ctx, ffn);
        ffn = linear(ctx, require_tensor(model_, prefix + ".linear2.weight"), require_tensor(model_, prefix + ".linear2.bias"), ffn);
        cur = ggml_add(ctx, cur, ffn);
    }

    // encode_norm
    cur = layer_norm(
        ctx,
        cur,
        require_tensor(model_, "geometry_encoder.encode_norm.weight"),
        require_tensor(model_, "geometry_encoder.encode_norm.bias")
    );

    ggml_tensor * capture = ggml_cont(ctx, cur);
    ggml_build_forward_expand(gf, capture);

    ggml_backend_sched_alloc_graph(sched, gf);

    // Set input data
    ggml_backend_tensor_set(image, image_seq.data(), 0, image_seq.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_seq.data(), 0, pos_seq.size() * sizeof(float));
    if (point_coords_t != nullptr) {
        ggml_backend_tensor_set(point_coords_t, point_coords.data(), 0, point_coords.size() * sizeof(float));
        ggml_backend_tensor_set(point_labels_t, point_labels.data(), 0, point_labels.size() * sizeof(int32_t));
        ggml_backend_tensor_set(point_pos_enc_t, point_pos_enc_data.data(), 0, point_pos_enc_data.size() * sizeof(float));
        ggml_backend_tensor_set(point_pool_t, point_pool_data.data(), 0, point_pool_data.size() * sizeof(float));
    }
    if (box_coords_t != nullptr) {
        ggml_backend_tensor_set(box_coords_t, box_coords.data(), 0, box_coords.size() * sizeof(float));
        ggml_backend_tensor_set(box_labels_t, box_label_vec.data(), 0, box_label_vec.size() * sizeof(int32_t));
        ggml_backend_tensor_set(box_pos_enc_t, box_pos_enc_data.data(), 0, box_pos_enc_data.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        throw std::runtime_error("geometry encoder graph compute failed");
    }

    GeometryEncoderOutput out;
    out.geo_seq_len = geo_token_count;
    out.hidden_dim = kModelDim;
    out.geo_tokens.resize(static_cast<size_t>(geo_token_count * kModelDim));
    ggml_backend_tensor_get(capture, out.geo_tokens.data(), 0, out.geo_tokens.size() * sizeof(float));

    // All geometry tokens are valid (no padding)
    out.geo_mask.resize(static_cast<size_t>(geo_token_count), 0.0f);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return out;
}

}  // namespace sam3
