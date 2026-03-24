#include "sam3/decoder.h"

#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kModelDim = 256;
constexpr int32_t kHeads = 8;
constexpr int32_t kHeadDim = kModelDim / kHeads;
constexpr int32_t kLayers = 6;
constexpr int32_t kNumQueries = 200;
constexpr int32_t kPromptSineDim = 512;
constexpr int32_t kFfnDim = 2048;
constexpr float kLayerNormEps = 1e-5f;
constexpr float kInvSigmoidEps = 1e-3f;
constexpr int32_t kResolution = 1008;
constexpr int32_t kStride = 14;

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

ggml_tensor * relu(ggml_context * ctx, ggml_tensor * x) {
    return ggml_relu(ctx, x);
}

ggml_tensor * mlp(
    ggml_context * ctx,
    const GgufModel & model,
    const std::string & prefix,
    int layers,
    ggml_tensor * x
) {
    ggml_tensor * cur = x;
    for (int i = 0; i < layers; ++i) {
        cur = linear(
            ctx,
            require_tensor(model, prefix + ".layers." + std::to_string(i) + ".weight"),
            require_tensor(model, prefix + ".layers." + std::to_string(i) + ".bias"),
            cur
        );
        if (i + 1 != layers) {
            cur = relu(ctx, cur);
        }
    }
    return cur;
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

std::string layer_prefix(int layer) {
    return "transformer.decoder.layers." + std::to_string(layer);
}

std::vector<ggml_fp16_t> make_prompt_mask_f16(
    const std::vector<float> & prompt_mask,
    int64_t src_len,
    int64_t tgt_len
) {
    std::vector<ggml_fp16_t> out(static_cast<size_t>(src_len * tgt_len));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t tq = 0; tq < tgt_len; ++tq) {
        for (int64_t sk = 0; sk < src_len; ++sk) {
            const float masked = prompt_mask[static_cast<size_t>(sk)] > 0.5f ? neg_inf : 0.0f;
            out[static_cast<size_t>(tq * src_len + sk)] = ggml_fp32_to_fp16(masked);
        }
    }
    return out;
}

std::vector<float> gen_sineembed_for_position(const std::vector<float> & pos, int64_t seq_len) {
    const int64_t input_dim = 4;
    const int64_t half_feats = kModelDim / 2;
    const float scale = 2.0f * static_cast<float>(M_PI);

    std::vector<float> dim_t(static_cast<size_t>(half_feats));
    for (int64_t i = 0; i < half_feats; ++i) {
        const float exponent = 2.0f * std::floor(static_cast<float>(i) / 2.0f) / static_cast<float>(half_feats);
        dim_t[static_cast<size_t>(i)] = std::pow(10000.0f, exponent);
    }

    std::vector<float> out(static_cast<size_t>(seq_len * kPromptSineDim));
    for (int64_t s = 0; s < seq_len; ++s) {
        for (int coord = 0; coord < input_dim; ++coord) {
            const float embed = pos[static_cast<size_t>(s * input_dim + coord)] * scale;
            for (int64_t i = 0; i < half_feats; ++i) {
                const float v = embed / dim_t[static_cast<size_t>(i)];
                const float trig = (i % 2 == 0) ? std::sin(v) : std::cos(v);
                const size_t offset = static_cast<size_t>(s * kPromptSineDim + coord * half_feats + i);
                out[offset] = trig;
            }
        }
    }

    // reorder y, x, w, h blocks to match MLX concat order
    std::vector<float> reordered(static_cast<size_t>(seq_len * kPromptSineDim));
    for (int64_t s = 0; s < seq_len; ++s) {
        const float * src = out.data() + s * kPromptSineDim;
        float * dst = reordered.data() + s * kPromptSineDim;
        std::memcpy(dst + 0 * half_feats, src + 1 * half_feats, static_cast<size_t>(half_feats) * sizeof(float));
        std::memcpy(dst + 1 * half_feats, src + 0 * half_feats, static_cast<size_t>(half_feats) * sizeof(float));
        std::memcpy(dst + 2 * half_feats, src + 2 * half_feats, static_cast<size_t>(half_feats) * sizeof(float));
        std::memcpy(dst + 3 * half_feats, src + 3 * half_feats, static_cast<size_t>(half_feats) * sizeof(float));
    }
    return reordered;
}

std::vector<float> sigmoid_vec(const std::vector<float> & x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    return out;
}

std::vector<float> inverse_sigmoid_vec(const std::vector<float> & x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float clipped = std::min(1.0f - kInvSigmoidEps, std::max(kInvSigmoidEps, x[i]));
        out[i] = std::log(clipped / (1.0f - clipped));
    }
    return out;
}

std::vector<float> build_box_rpb_bias(
    const std::vector<float> & reference_boxes,
    int64_t num_queries,
    int64_t feat_h,
    int64_t feat_w,
    const std::vector<float> & mlp_x_0_w,
    const std::vector<float> & mlp_x_0_b,
    const std::vector<float> & mlp_x_1_w,
    const std::vector<float> & mlp_x_1_b,
    const std::vector<float> & mlp_y_0_w,
    const std::vector<float> & mlp_y_0_b,
    const std::vector<float> & mlp_y_1_w,
    const std::vector<float> & mlp_y_1_b
) {
    std::vector<float> coords_h(static_cast<size_t>(feat_h));
    std::vector<float> coords_w(static_cast<size_t>(feat_w));
    for (int64_t i = 0; i < feat_h; ++i) {
        coords_h[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(feat_h);
    }
    for (int64_t i = 0; i < feat_w; ++i) {
        coords_w[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(feat_w);
    }

    auto mlp2 = [](const std::vector<float> & in, const std::vector<float> & w0, const std::vector<float> & b0, const std::vector<float> & w1, const std::vector<float> & b1, int64_t out0, int64_t out1) {
        std::vector<float> h(static_cast<size_t>(out0));
        for (int64_t o = 0; o < out0; ++o) {
            float sum = b0[static_cast<size_t>(o)];
            for (size_t i = 0; i < in.size(); ++i) {
                sum += w0[static_cast<size_t>(o * static_cast<int64_t>(in.size()) + static_cast<int64_t>(i))] * in[i];
            }
            h[static_cast<size_t>(o)] = std::max(0.0f, sum);
        }
        std::vector<float> out(static_cast<size_t>(out1));
        for (int64_t o = 0; o < out1; ++o) {
            float sum = b1[static_cast<size_t>(o)];
            for (int64_t i = 0; i < out0; ++i) {
                sum += w1[static_cast<size_t>(o * out0 + i)] * h[static_cast<size_t>(i)];
            }
            out[static_cast<size_t>(o)] = sum;
        }
        return out;
    };

    std::vector<float> out(static_cast<size_t>(kHeads * (num_queries + 1) * feat_h * feat_w), 0.0f);
    for (int64_t q = 0; q < num_queries; ++q) {
        const float cx = reference_boxes[static_cast<size_t>(q * 4 + 0)];
        const float cy = reference_boxes[static_cast<size_t>(q * 4 + 1)];
        const float bw = reference_boxes[static_cast<size_t>(q * 4 + 2)];
        const float bh = reference_boxes[static_cast<size_t>(q * 4 + 3)];
        const float x0 = cx - 0.5f * bw;
        const float x1 = cx + 0.5f * bw;
        const float y0 = cy - 0.5f * bh;
        const float y1 = cy + 0.5f * bh;

        std::vector<std::vector<float>> dx(static_cast<size_t>(feat_w));
        std::vector<std::vector<float>> dy(static_cast<size_t>(feat_h));
        for (int64_t x = 0; x < feat_w; ++x) {
            std::vector<float> in = {
                coords_w[static_cast<size_t>(x)] - x0,
                coords_w[static_cast<size_t>(x)] - x1,
            };
            for (float & v : in) {
                v *= 8.0f;
                v = std::copysign(std::log2(std::fabs(v) + 1.0f) / std::log2(8.0f), v);
            }
            dx[static_cast<size_t>(x)] = mlp2(in, mlp_x_0_w, mlp_x_0_b, mlp_x_1_w, mlp_x_1_b, kModelDim, kHeads);
        }
        for (int64_t y = 0; y < feat_h; ++y) {
            std::vector<float> in = {
                coords_h[static_cast<size_t>(y)] - y0,
                coords_h[static_cast<size_t>(y)] - y1,
            };
            for (float & v : in) {
                v *= 8.0f;
                v = std::copysign(std::log2(std::fabs(v) + 1.0f) / std::log2(8.0f), v);
            }
            dy[static_cast<size_t>(y)] = mlp2(in, mlp_y_0_w, mlp_y_0_b, mlp_y_1_w, mlp_y_1_b, kModelDim, kHeads);
        }

        for (int64_t h = 0; h < kHeads; ++h) {
            for (int64_t y = 0; y < feat_h; ++y) {
                for (int64_t x = 0; x < feat_w; ++x) {
                    const int64_t src = y * feat_w + x;
                    const size_t idx = static_cast<size_t>(((h * (num_queries + 1) + (q + 1)) * (feat_h * feat_w)) + src);
                    out[idx] = dy[static_cast<size_t>(y)][static_cast<size_t>(h)] + dx[static_cast<size_t>(x)][static_cast<size_t>(h)];
                }
            }
        }
    }
    return out;
}

}  // namespace

Decoder::Decoder(const GgufModel & model) : model_(model) {}

DecoderOutput Decoder::run(
    const std::vector<float> & memory,
    const std::vector<int64_t> & memory_shape,
    const std::vector<float> & pos_embed,
    const std::vector<int64_t> & pos_shape,
    const std::vector<float> & prompt,
    const std::vector<int64_t> & prompt_shape,
    const std::vector<float> & prompt_mask,
    const std::vector<int64_t> & prompt_mask_shape
) const {
    if (memory_shape != pos_shape || memory_shape.size() != 3 || prompt_shape.size() != 3 || prompt_mask_shape.size() != 2) {
        throw std::runtime_error("invalid decoder input shapes");
    }
    if (memory_shape[1] != 1 || prompt_shape[1] != 1 || prompt_mask_shape[0] != 1) {
        throw std::runtime_error("decoder currently expects batch size 1");
    }
    if (memory_shape[2] != kModelDim || prompt_shape[2] != kModelDim) {
        throw std::runtime_error("decoder expects hidden_dim=256");
    }

    const int32_t hw = static_cast<int32_t>(memory_shape[0]);
    const int32_t prompt_seq = static_cast<int32_t>(prompt_shape[0]);
    const int32_t feat_h = kResolution / kStride;
    const int32_t feat_w = kResolution / kStride;
    if (hw != feat_h * feat_w) {
        throw std::runtime_error("decoder expects single 72x72 feature level");
    }

    const std::vector<ggml_fp16_t> prompt_mask_f16 = make_prompt_mask_f16(prompt_mask, prompt_seq, kNumQueries + 1);
    ggml_backend_t backend = model_.backend();
    ggml_backend_t cpu_backend = nullptr;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_backend = model_.cpu_backend();
    }

    std::vector<float> query_embed(static_cast<size_t>(kNumQueries * kModelDim));
    std::vector<float> ref_points_weight(static_cast<size_t>(kNumQueries * 4));
    std::vector<float> presence_token_weight(static_cast<size_t>(kModelDim));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.query_embed.weight"), query_embed.data(), 0, query_embed.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.reference_points.weight"), ref_points_weight.data(), 0, ref_points_weight.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.presence_token.weight"), presence_token_weight.data(), 0, presence_token_weight.size() * sizeof(float));
    std::vector<float> current_output = query_embed;
    std::vector<float> current_ref = sigmoid_vec(ref_points_weight);
    std::vector<float> current_presence = presence_token_weight;

    std::vector<float> rpb_x0_w(static_cast<size_t>(kModelDim * 2));
    std::vector<float> rpb_x0_b(static_cast<size_t>(kModelDim));
    std::vector<float> rpb_x1_w(static_cast<size_t>(kHeads * kModelDim));
    std::vector<float> rpb_x1_b(static_cast<size_t>(kHeads));
    std::vector<float> rpb_y0_w(static_cast<size_t>(kModelDim * 2));
    std::vector<float> rpb_y0_b(static_cast<size_t>(kModelDim));
    std::vector<float> rpb_y1_w(static_cast<size_t>(kHeads * kModelDim));
    std::vector<float> rpb_y1_b(static_cast<size_t>(kHeads));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.boxRPB_embed_x.layers.0.weight"), rpb_x0_w.data(), 0, rpb_x0_w.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.boxRPB_embed_x.layers.0.bias"), rpb_x0_b.data(), 0, rpb_x0_b.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.boxRPB_embed_x.layers.1.weight"), rpb_x1_w.data(), 0, rpb_x1_w.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.boxRPB_embed_x.layers.1.bias"), rpb_x1_b.data(), 0, rpb_x1_b.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.boxRPB_embed_y.layers.0.weight"), rpb_y0_w.data(), 0, rpb_y0_w.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.boxRPB_embed_y.layers.0.bias"), rpb_y0_b.data(), 0, rpb_y0_b.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.boxRPB_embed_y.layers.1.weight"), rpb_y1_w.data(), 0, rpb_y1_w.size() * sizeof(float));
    ggml_backend_tensor_get(require_tensor(model_, "transformer.decoder.boxRPB_embed_y.layers.1.bias"), rpb_y1_b.data(), 0, rpb_y1_b.size() * sizeof(float));

    DecoderOutput out;
    out.num_layers = kLayers;
    out.num_queries = kNumQueries;
    out.hidden_dim = kModelDim;
    out.hs.resize(kLayers);
    out.reference_boxes.resize(kLayers);
    out.presence_logits.resize(kLayers);

    for (int layer = 0; layer < kLayers; ++layer) {
        const size_t graph_size = 65536;
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
            if (cpu_backend != nullptr) {

            }
            throw std::runtime_error("ggml_init failed");
        }

        ggml_backend_t backends[2] = { backend, cpu_backend };
        const int n_backends = cpu_backend != nullptr ? 2 : 1;
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
        if (sched == nullptr) {
            ggml_free(ctx);
            if (cpu_backend != nullptr) {

            }
            throw std::runtime_error("failed to create decoder scheduler");
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

        const std::string prefix = layer_prefix(layer);
        const std::vector<float> ref_inv_host = inverse_sigmoid_vec(current_ref);
        const std::vector<float> sine_host = gen_sineembed_for_position(current_ref, kNumQueries);
        const std::vector<float> image_bias = build_box_rpb_bias(
            current_ref,
            kNumQueries,
            feat_h,
            feat_w,
            rpb_x0_w, rpb_x0_b, rpb_x1_w, rpb_x1_b,
            rpb_y0_w, rpb_y0_b, rpb_y1_w, rpb_y1_b
        );
        const std::vector<float> zero_presence_pos(static_cast<size_t>(kModelDim), 0.0f);

        ggml_tensor * memory_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, hw);
        ggml_tensor * pos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, hw);
        ggml_tensor * prompt_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, prompt_seq);
        ggml_tensor * prompt_mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, prompt_seq, kNumQueries + 1);
        ggml_tensor * output_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, kNumQueries);
        ggml_tensor * presence_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, 1);
        ggml_tensor * query_sine_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kPromptSineDim, kNumQueries);
        ggml_tensor * image_bias_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hw, kNumQueries + 1, kHeads, 1);
        ggml_tensor * ref_inv_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, kNumQueries);
        ggml_tensor * zero_presence_pos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, 1);

        for (ggml_tensor * t : {memory_t, pos_t, prompt_t, prompt_mask_t, output_t, presence_t, query_sine_t, image_bias_t, ref_inv_t, zero_presence_pos_t}) {
            ggml_backend_sched_set_tensor_backend(sched, t, backend);
        }

        ggml_tensor * query_pos = mlp(ctx, model_, "transformer.decoder.ref_point_head", 2, query_sine_t);
        ggml_tensor * self_q_in = ggml_concat(ctx, zero_presence_pos_t, query_pos, 1);
        ggml_tensor * self_kv_in = ggml_concat(ctx, presence_t, output_t, 1);
        ggml_tensor * self_qk = ggml_add(ctx, self_kv_in, self_q_in);
        ggml_tensor * self_out = mha(
            ctx,
            linear(ctx, require_tensor(model_, prefix + ".self_attn.query_proj.weight"), require_tensor(model_, prefix + ".self_attn.query_proj.bias"), self_qk),
            linear(ctx, require_tensor(model_, prefix + ".self_attn.key_proj.weight"), require_tensor(model_, prefix + ".self_attn.key_proj.bias"), self_qk),
            linear(ctx, require_tensor(model_, prefix + ".self_attn.value_proj.weight"), require_tensor(model_, prefix + ".self_attn.value_proj.bias"), self_kv_in),
            kNumQueries + 1,
            kNumQueries + 1,
            nullptr
        );
        self_out = linear(ctx, require_tensor(model_, prefix + ".self_attn.out_proj.weight"), require_tensor(model_, prefix + ".self_attn.out_proj.bias"), self_out);
        ggml_tensor * tgt = layer_norm(ctx, ggml_add(ctx, self_kv_in, self_out), require_tensor(model_, prefix + ".norm2.weight"), require_tensor(model_, prefix + ".norm2.bias"));

        ggml_tensor * tgt_q = ggml_add(ctx, tgt, self_q_in);
        ggml_tensor * text_out = mha(
            ctx,
            linear(ctx, require_tensor(model_, prefix + ".ca_text.query_proj.weight"), require_tensor(model_, prefix + ".ca_text.query_proj.bias"), tgt_q),
            linear(ctx, require_tensor(model_, prefix + ".ca_text.key_proj.weight"), require_tensor(model_, prefix + ".ca_text.key_proj.bias"), prompt_t),
            linear(ctx, require_tensor(model_, prefix + ".ca_text.value_proj.weight"), require_tensor(model_, prefix + ".ca_text.value_proj.bias"), prompt_t),
            kNumQueries + 1,
            prompt_seq,
            prompt_mask_t
        );
        text_out = linear(ctx, require_tensor(model_, prefix + ".ca_text.out_proj.weight"), require_tensor(model_, prefix + ".ca_text.out_proj.bias"), text_out);
        tgt = layer_norm(ctx, ggml_add(ctx, tgt, text_out), require_tensor(model_, prefix + ".catext_norm.weight"), require_tensor(model_, prefix + ".catext_norm.bias"));

        ggml_tensor * image_q = ggml_add(ctx, tgt, self_q_in);
        ggml_tensor * image_k = ggml_add(ctx, memory_t, pos_t);
        ggml_tensor * image_out = mha(
            ctx,
            linear(ctx, require_tensor(model_, prefix + ".cross_attn.query_proj.weight"), require_tensor(model_, prefix + ".cross_attn.query_proj.bias"), image_q),
            linear(ctx, require_tensor(model_, prefix + ".cross_attn.key_proj.weight"), require_tensor(model_, prefix + ".cross_attn.key_proj.bias"), image_k),
            linear(ctx, require_tensor(model_, prefix + ".cross_attn.value_proj.weight"), require_tensor(model_, prefix + ".cross_attn.value_proj.bias"), memory_t),
            kNumQueries + 1,
            hw,
            image_bias_t
        );
        image_out = linear(ctx, require_tensor(model_, prefix + ".cross_attn.out_proj.weight"), require_tensor(model_, prefix + ".cross_attn.out_proj.bias"), image_out);
        tgt = layer_norm(ctx, ggml_add(ctx, tgt, image_out), require_tensor(model_, prefix + ".norm1.weight"), require_tensor(model_, prefix + ".norm1.bias"));

        ggml_tensor * ffn = linear(ctx, require_tensor(model_, prefix + ".linear1.weight"), require_tensor(model_, prefix + ".linear1.bias"), tgt);
        ffn = relu(ctx, ffn);
        ffn = linear(ctx, require_tensor(model_, prefix + ".linear2.weight"), require_tensor(model_, prefix + ".linear2.bias"), ffn);
        tgt = layer_norm(ctx, ggml_add(ctx, tgt, ffn), require_tensor(model_, prefix + ".norm3.weight"), require_tensor(model_, prefix + ".norm3.bias"));

        ggml_tensor * next_presence = ggml_view_2d(ctx, tgt, kModelDim, 1, tgt->nb[1], 0);
        ggml_tensor * next_output = ggml_view_2d(ctx, tgt, kModelDim, kNumQueries, tgt->nb[1], tgt->nb[1]);
        ggml_tensor * output_norm = layer_norm(ctx, next_output, require_tensor(model_, "transformer.decoder.norm.weight"), require_tensor(model_, "transformer.decoder.norm.bias"));
        ggml_tensor * delta = mlp(ctx, model_, "transformer.decoder.bbox_embed", 3, output_norm);
        ggml_tensor * next_ref = ggml_sigmoid(ctx, ggml_add(ctx, ref_inv_t, delta));
        ggml_tensor * presence_norm = layer_norm(ctx, next_presence, require_tensor(model_, "transformer.decoder.presence_token_out_norm.weight"), require_tensor(model_, "transformer.decoder.presence_token_out_norm.bias"));
        ggml_tensor * presence_logit = mlp(ctx, model_, "transformer.decoder.presence_token_head", 3, presence_norm);

        ggml_tensor * output_capture = ggml_cont(ctx, next_output);
        ggml_tensor * presence_capture = ggml_cont(ctx, next_presence);
        ggml_tensor * hs_capture = ggml_cont(ctx, output_norm);
        ggml_tensor * ref_capture = ggml_cont(ctx, next_ref);
        ggml_tensor * presence_logit_capture = ggml_cont(ctx, presence_logit);

        ggml_build_forward_expand(gf, output_capture);
        ggml_build_forward_expand(gf, presence_capture);
        ggml_build_forward_expand(gf, hs_capture);
        ggml_build_forward_expand(gf, ref_capture);
        ggml_build_forward_expand(gf, presence_logit_capture);

        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_tensor_set(memory_t, memory.data(), 0, memory.size() * sizeof(float));
        ggml_backend_tensor_set(pos_t, pos_embed.data(), 0, pos_embed.size() * sizeof(float));
        ggml_backend_tensor_set(prompt_t, prompt.data(), 0, prompt.size() * sizeof(float));
        ggml_backend_tensor_set(prompt_mask_t, prompt_mask_f16.data(), 0, prompt_mask_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(output_t, current_output.data(), 0, current_output.size() * sizeof(float));
        ggml_backend_tensor_set(presence_t, current_presence.data(), 0, current_presence.size() * sizeof(float));
        ggml_backend_tensor_set(query_sine_t, sine_host.data(), 0, sine_host.size() * sizeof(float));
        ggml_backend_tensor_set(image_bias_t, image_bias.data(), 0, image_bias.size() * sizeof(float));
        ggml_backend_tensor_set(ref_inv_t, ref_inv_host.data(), 0, ref_inv_host.size() * sizeof(float));
        ggml_backend_tensor_set(zero_presence_pos_t, zero_presence_pos.data(), 0, zero_presence_pos.size() * sizeof(float));

        const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            if (cpu_backend != nullptr) {

            }
            throw std::runtime_error("decoder layer graph compute failed");
        }

        out.hs[static_cast<size_t>(layer)].resize(static_cast<size_t>(kNumQueries * kModelDim));
        out.reference_boxes[static_cast<size_t>(layer)] = current_ref;
        out.presence_logits[static_cast<size_t>(layer)].resize(1);
        ggml_backend_tensor_get(hs_capture, out.hs[static_cast<size_t>(layer)].data(), 0, out.hs[static_cast<size_t>(layer)].size() * sizeof(float));
        ggml_backend_tensor_get(presence_logit_capture, out.presence_logits[static_cast<size_t>(layer)].data(), 0, sizeof(float));

        current_output.resize(static_cast<size_t>(kNumQueries * kModelDim));
        current_presence.resize(static_cast<size_t>(kModelDim));
        current_ref.resize(static_cast<size_t>(kNumQueries * 4));
        ggml_backend_tensor_get(output_capture, current_output.data(), 0, current_output.size() * sizeof(float));
        ggml_backend_tensor_get(presence_capture, current_presence.data(), 0, current_presence.size() * sizeof(float));
        ggml_backend_tensor_get(ref_capture, current_ref.data(), 0, current_ref.size() * sizeof(float));

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    return out;
}

}  // namespace sam3
