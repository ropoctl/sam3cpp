#include "sam3/segmentation_head.h"

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
constexpr int32_t kNumQueries = 200;
constexpr int32_t kGroupNormGroups = 8;
constexpr float kLayerNormEps = 1e-5f;
constexpr float kGroupNormEps = 1e-5f;

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

ggml_tensor * add_bias_1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias, int64_t channels) {
    ggml_tensor * b = ggml_reshape_4d(ctx, ensure_f32(ctx, bias), 1, 1, channels, 1);
    return ggml_add(ctx, x, b);
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
    ggml_tensor * w_ggml = ggml_cont(ctx, ggml_permute(ctx, w, 2, 0, 1, 3));
    ggml_tensor * y = ggml_conv_2d(ctx, w_ggml, x, s0, s1, p0, p1, 1, 1);
    return add_bias_1d(ctx, y, b, w_ggml->ne[3]);
}

ggml_tensor * group_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias
) {
    ggml_tensor * y = ggml_group_norm(ctx, x, kGroupNormGroups, kGroupNormEps);
    ggml_tensor * w = ggml_reshape_4d(ctx, ensure_f32(ctx, weight), 1, 1, x->ne[2], 1);
    ggml_tensor * b = ggml_reshape_4d(ctx, ensure_f32(ctx, bias), 1, 1, x->ne[2], 1);
    y = ggml_add(ctx, ggml_mul(ctx, y, w), b);
    return y;
}

// Rearrange a 1-D parameter vector [C] from interleaved channel order
// (c = ci*G + g) to standard group-norm order (c = g*(C/G) + ci).
ggml_tensor * interleaved_to_standard_1d(
    ggml_context * ctx,
    ggml_tensor * param,
    int groups
) {
    // param: [C]
    const int64_t C = param->ne[0];
    const int64_t cpg = C / groups;
    // Reshape to [G, cpg] then transpose to [cpg, G] then flatten to [C].
    ggml_tensor * p2d = ggml_reshape_2d(ctx, ensure_f32(ctx, param), groups, cpg);
    ggml_tensor * pt  = ggml_cont(ctx, ggml_transpose(ctx, p2d));
    return ggml_reshape_1d(ctx, pt, C);
}

// Rearrange channels of a 4-D tensor [W, H, C, 1] from interleaved
// (c = ci*G + g) to standard group-norm order (c = g*(C/G) + ci).
ggml_tensor * interleaved_to_standard_4d(
    ggml_context * ctx,
    ggml_tensor * x,
    int groups
) {
    const int64_t W   = x->ne[0];
    const int64_t H   = x->ne[1];
    const int64_t C   = x->ne[2];
    const int64_t cpg = C / groups;
    // [W, H, C, 1] → [W, H, G, cpg] (G fastest in dim2)
    ggml_tensor * r = ggml_reshape_4d(ctx, x, W, H, groups, cpg);
    // permute(0,1,3,2) → [W, H, cpg, G]
    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
    return ggml_reshape_4d(ctx, r, W, H, C, 1);
}

// Reverse: standard → interleaved.
ggml_tensor * standard_to_interleaved_4d(
    ggml_context * ctx,
    ggml_tensor * x,
    int groups
) {
    const int64_t W   = x->ne[0];
    const int64_t H   = x->ne[1];
    const int64_t C   = x->ne[2];
    const int64_t cpg = C / groups;
    // [W, H, C, 1] → [W, H, cpg, G]
    ggml_tensor * r = ggml_reshape_4d(ctx, x, W, H, cpg, groups);
    // permute(0,1,3,2) → [W, H, G, cpg]
    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
    return ggml_reshape_4d(ctx, r, W, H, C, 1);
}

// GPU-accelerated group-norm with interleaved channel layout + ReLU.
// Equivalent to the old CPU mlx_group_norm_interleaved_relu but runs
// entirely as ggml graph ops (Metal on Apple Silicon).
ggml_tensor * group_norm_interleaved_relu(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias,
    int groups
) {
    // 1. Rearrange channels: interleaved → standard
    ggml_tensor * xs = interleaved_to_standard_4d(ctx, x, groups);

    // 2. Standard group norm (runs on GPU via Metal)
    ggml_tensor * y = ggml_group_norm(ctx, xs, groups, kGroupNormEps);

    // 3. Affine transform with rearranged weight/bias
    ggml_tensor * ws = interleaved_to_standard_1d(ctx, weight, groups);
    ggml_tensor * bs = interleaved_to_standard_1d(ctx, bias, groups);
    ggml_tensor * w4 = ggml_reshape_4d(ctx, ws, 1, 1, x->ne[2], 1);
    ggml_tensor * b4 = ggml_reshape_4d(ctx, bs, 1, 1, x->ne[2], 1);
    y = ggml_add(ctx, ggml_mul(ctx, y, w4), b4);

    // 4. ReLU
    y = ggml_relu(ctx, y);

    // 5. Rearrange channels back: standard → interleaved
    y = standard_to_interleaved_4d(ctx, y, groups);

    return y;
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

std::vector<ggml_fp16_t> make_prompt_mask_f16(
    const std::vector<float> & prompt_mask,
    int64_t src_len,
    int64_t tgt_len
) {
    std::vector<ggml_fp16_t> out(static_cast<size_t>(src_len * tgt_len));
    const float neg_inf = -INFINITY;
    for (int64_t tq = 0; tq < tgt_len; ++tq) {
        for (int64_t sk = 0; sk < src_len; ++sk) {
            const float masked = prompt_mask[static_cast<size_t>(sk)] > 0.5f ? neg_inf : 0.0f;
            out[static_cast<size_t>(tq * src_len + sk)] = ggml_fp32_to_fp16(masked);
        }
    }
    return out;
}

ggml_tensor * encode_memory_as_feature_map(
    ggml_context * ctx,
    ggml_tensor * encoder_hidden_states,
    int64_t feat_w,
    int64_t feat_h
) {
    ggml_tensor * transposed = ggml_cont(ctx, ggml_transpose(ctx, encoder_hidden_states));
    return ggml_reshape_4d(ctx, transposed, feat_w, feat_h, kModelDim, 1);
}

}  // namespace

SegmentationHead::SegmentationHead(const GgufModel & model) : model_(model) {}

SegmentationHeadOutput SegmentationHead::run(
    const std::vector<std::vector<float>> & backbone_fpn,
    const std::vector<std::vector<int64_t>> & backbone_fpn_shapes,
    const std::vector<float> & encoder_hidden_states,
    const std::vector<int64_t> & encoder_hidden_states_shape,
    const std::vector<float> & prompt,
    const std::vector<int64_t> & prompt_shape,
    const std::vector<float> & prompt_mask,
    const std::vector<int64_t> & prompt_mask_shape,
    const std::vector<float> & obj_queries,
    const std::vector<int64_t> & obj_queries_shape
) const {
    if (backbone_fpn.size() != 3 || backbone_fpn_shapes.size() != 3) {
        throw std::runtime_error("segmentation head expects exactly 3 backbone FPN levels");
    }
    if (encoder_hidden_states_shape.size() != 3 || prompt_shape.size() != 3 || prompt_mask_shape.size() != 2 || obj_queries_shape.size() != 3) {
        throw std::runtime_error("invalid segmentation input shapes");
    }
    if (encoder_hidden_states_shape[1] != 1 || prompt_shape[1] != 1 || prompt_mask_shape[0] != 1 || obj_queries_shape[1] != 1) {
        throw std::runtime_error("segmentation head currently expects batch size 1");
    }
    if (encoder_hidden_states_shape[2] != kModelDim || prompt_shape[2] != kModelDim || obj_queries_shape[2] != kModelDim) {
        throw std::runtime_error("segmentation head expects hidden_dim=256");
    }
    if (obj_queries_shape[0] != kNumQueries) {
        throw std::runtime_error("segmentation head expects 200 object queries");
    }

    for (size_t i = 0; i < backbone_fpn_shapes.size(); ++i) {
        const auto & shape = backbone_fpn_shapes[i];
        if (shape.size() != 4 || shape[0] != 1 || shape[1] != kModelDim) {
            throw std::runtime_error("segmentation head expects backbone FPN tensors with shape [1, 256, H, W]");
        }
    }

    const int64_t prompt_seq = prompt_shape[0];
    const int64_t feat_h = backbone_fpn_shapes.back()[2];
    const int64_t feat_w = backbone_fpn_shapes.back()[3];
    const int64_t hw = feat_h * feat_w;
    if (encoder_hidden_states_shape[0] != hw) {
        throw std::runtime_error("encoder_hidden_states shape does not match final backbone feature size");
    }

    const int64_t out_h = backbone_fpn_shapes.front()[2];
    const int64_t out_w = backbone_fpn_shapes.front()[3];

    const std::vector<ggml_fp16_t> prompt_mask_f16 = make_prompt_mask_f16(prompt_mask, prompt_seq, hw);

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
    SegmentationHeadOutput out;
    out.batch = 1;
    out.num_queries = kNumQueries;
    out.hidden_dim = kModelDim;
    out.height = out_h;
    out.width = out_w;

    std::vector<float> stage0_output_host;
    std::vector<float> pixel_embed_host;

    // --- Stage 0: cross-attention + conv + group_norm_interleaved_relu ---
    {
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

        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
        if (sched == nullptr) {
            ggml_free(ctx);
            throw std::runtime_error("failed to create segmentation scheduler");
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

        ggml_tensor * fpn1_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, backbone_fpn_shapes[1][3], backbone_fpn_shapes[1][2], backbone_fpn_shapes[1][1], backbone_fpn_shapes[1][0]);
        ggml_tensor * encoder_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, hw);
        ggml_tensor * prompt_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, prompt_seq);
        ggml_tensor * prompt_mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, prompt_seq, hw);
        for (ggml_tensor * t : {fpn1_t, encoder_t, prompt_t, prompt_mask_t}) {
            ggml_backend_sched_set_tensor_backend(sched, t, backend);
        }

        ggml_tensor * enc_norm = layer_norm(
            ctx,
            encoder_t,
            require_tensor(model_, "segmentation_head.cross_attn_norm.weight"),
            require_tensor(model_, "segmentation_head.cross_attn_norm.bias"));
        ggml_tensor * prompt_attn = mha(
            ctx,
            linear(ctx,
                   require_tensor(model_, "segmentation_head.cross_attend_prompt.query_proj.weight"),
                   require_tensor(model_, "segmentation_head.cross_attend_prompt.query_proj.bias"),
                   enc_norm),
            linear(ctx,
                   require_tensor(model_, "segmentation_head.cross_attend_prompt.key_proj.weight"),
                   require_tensor(model_, "segmentation_head.cross_attend_prompt.key_proj.bias"),
                   prompt_t),
            linear(ctx,
                   require_tensor(model_, "segmentation_head.cross_attend_prompt.value_proj.weight"),
                   require_tensor(model_, "segmentation_head.cross_attend_prompt.value_proj.bias"),
                   prompt_t),
            static_cast<int32_t>(hw),
            static_cast<int32_t>(prompt_seq),
            prompt_mask_t);
        prompt_attn = linear(
            ctx,
            require_tensor(model_, "segmentation_head.cross_attend_prompt.out_proj.weight"),
            require_tensor(model_, "segmentation_head.cross_attend_prompt.out_proj.bias"),
            prompt_attn);
        ggml_tensor * enc_attended = ggml_add(ctx, encoder_t, prompt_attn);
        ggml_tensor * fpn2_t = encode_memory_as_feature_map(ctx, enc_attended, feat_w, feat_h);
        ggml_tensor * stage0_in = ggml_add(ctx, fpn1_t, ggml_upscale(ctx, fpn2_t, 2, GGML_SCALE_MODE_NEAREST));
        ggml_tensor * stage0_conv = conv2d_bias(
            ctx,
            require_tensor(model_, "segmentation_head.pixel_decoder.conv_layers.0.weight"),
            require_tensor(model_, "segmentation_head.pixel_decoder.conv_layers.0.bias"),
            stage0_in,
            1, 1, 1, 1);

        // Group norm with interleaved channel layout + ReLU (runs on GPU)
        ggml_tensor * stage0_out = group_norm_interleaved_relu(
            ctx,
            stage0_conv,
            require_tensor(model_, "segmentation_head.pixel_decoder.norms.0.weight"),
            require_tensor(model_, "segmentation_head.pixel_decoder.norms.0.bias"),
            kGroupNormGroups);

        ggml_tensor * stage0_capture = ggml_cont(ctx, stage0_out);
        ggml_build_forward_expand(gf, stage0_capture);

        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_tensor_set(fpn1_t, backbone_fpn[1].data(), 0, backbone_fpn[1].size() * sizeof(float));
        ggml_backend_tensor_set(encoder_t, encoder_hidden_states.data(), 0, encoder_hidden_states.size() * sizeof(float));
        ggml_backend_tensor_set(prompt_t, prompt.data(), 0, prompt.size() * sizeof(float));
        ggml_backend_tensor_set(prompt_mask_t, prompt_mask_f16.data(), 0, prompt_mask_f16.size() * sizeof(ggml_fp16_t));

        const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            throw std::runtime_error("segmentation stage 0 graph compute failed");
        }

        stage0_output_host.resize(static_cast<size_t>(ggml_nelements(stage0_capture)));
        ggml_backend_tensor_get(stage0_capture, stage0_output_host.data(), 0, stage0_output_host.size() * sizeof(float));

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    // --- Stage 1: upscale + add fpn0 + conv + group_norm_interleaved_relu ---
    {
        const size_t graph_size = 16384;
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

        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
        if (sched == nullptr) {
            ggml_free(ctx);
            throw std::runtime_error("failed to create segmentation scheduler");
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

        ggml_tensor * fpn0_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, backbone_fpn_shapes[0][3], backbone_fpn_shapes[0][2], backbone_fpn_shapes[0][1], backbone_fpn_shapes[0][0]);
        ggml_tensor * stage0_out_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, backbone_fpn_shapes[1][3], backbone_fpn_shapes[1][2], backbone_fpn_shapes[1][1], backbone_fpn_shapes[1][0]);
        ggml_backend_sched_set_tensor_backend(sched, fpn0_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, stage0_out_t, backend);

        ggml_tensor * stage1_in = ggml_add(ctx, fpn0_t, ggml_upscale(ctx, stage0_out_t, 2, GGML_SCALE_MODE_NEAREST));
        ggml_tensor * stage1_conv = conv2d_bias(
            ctx,
            require_tensor(model_, "segmentation_head.pixel_decoder.conv_layers.1.weight"),
            require_tensor(model_, "segmentation_head.pixel_decoder.conv_layers.1.bias"),
            stage1_in,
            1, 1, 1, 1);

        // Group norm with interleaved channel layout + ReLU (runs on GPU)
        ggml_tensor * pixel_embed = group_norm_interleaved_relu(
            ctx,
            stage1_conv,
            require_tensor(model_, "segmentation_head.pixel_decoder.norms.1.weight"),
            require_tensor(model_, "segmentation_head.pixel_decoder.norms.1.bias"),
            kGroupNormGroups);

        ggml_tensor * pixel_embed_capture = ggml_cont(ctx, pixel_embed);
        ggml_build_forward_expand(gf, pixel_embed_capture);

        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_tensor_set(fpn0_t, backbone_fpn[0].data(), 0, backbone_fpn[0].size() * sizeof(float));
        ggml_backend_tensor_set(stage0_out_t, stage0_output_host.data(), 0, stage0_output_host.size() * sizeof(float));

        const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            throw std::runtime_error("segmentation stage 1 graph compute failed");
        }

        pixel_embed_host.resize(static_cast<size_t>(ggml_nelements(pixel_embed_capture)));
        ggml_backend_tensor_get(pixel_embed_capture, pixel_embed_host.data(), 0, pixel_embed_host.size() * sizeof(float));

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    // --- Stage 2: final segmentation outputs ---
    {
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

        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
        if (sched == nullptr) {
            ggml_free(ctx);
            throw std::runtime_error("failed to create segmentation scheduler");
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

        ggml_tensor * pixel_embed_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, out_w, out_h, kModelDim, 1);
        ggml_tensor * queries_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, kNumQueries);
        ggml_backend_sched_set_tensor_backend(sched, pixel_embed_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, queries_t, backend);

        ggml_tensor * semantic_seg = conv2d_bias(
            ctx,
            require_tensor(model_, "segmentation_head.semantic_seg_head.weight"),
            require_tensor(model_, "segmentation_head.semantic_seg_head.bias"),
            pixel_embed_t,
            1, 1, 0, 0);
        ggml_tensor * instance_embed = conv2d_bias(
            ctx,
            require_tensor(model_, "segmentation_head.instance_seg_head.weight"),
            require_tensor(model_, "segmentation_head.instance_seg_head.bias"),
            pixel_embed_t,
            1, 1, 0, 0);
        ggml_tensor * mask_embed = mlp(ctx, model_, "segmentation_head.mask_predictor.mask_embed", 3, queries_t);
        ggml_tensor * pixel_mat = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, instance_embed, out_h * out_w, kModelDim)));
        ggml_tensor * pred_masks_mat = ggml_mul_mat(ctx, pixel_mat, mask_embed);
        ggml_tensor * pred_masks = ggml_reshape_4d(ctx, pred_masks_mat, out_w, out_h, kNumQueries, 1);

        ggml_tensor * pred_masks_capture = ggml_cont(ctx, pred_masks);
        ggml_tensor * semantic_capture = ggml_cont(ctx, semantic_seg);
        ggml_build_forward_expand(gf, pred_masks_capture);
        ggml_build_forward_expand(gf, semantic_capture);

        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_tensor_set(pixel_embed_t, pixel_embed_host.data(), 0, pixel_embed_host.size() * sizeof(float));
        ggml_backend_tensor_set(queries_t, obj_queries.data(), 0, obj_queries.size() * sizeof(float));

        const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            throw std::runtime_error("segmentation mask graph compute failed");
        }

        out.pred_masks.resize(static_cast<size_t>(ggml_nelements(pred_masks_capture)));
        out.semantic_seg.resize(static_cast<size_t>(ggml_nelements(semantic_capture)));
        ggml_backend_tensor_get(pred_masks_capture, out.pred_masks.data(), 0, out.pred_masks.size() * sizeof(float));
        ggml_backend_tensor_get(semantic_capture, out.semantic_seg.data(), 0, out.semantic_seg.size() * sizeof(float));

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }


    return out;
}

}  // namespace sam3
