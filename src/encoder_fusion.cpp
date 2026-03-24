#include "sam3/encoder_fusion.h"

#include "ggml-alloc.h"

#include <cmath>
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
constexpr int32_t kFfnDim = 2048;
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
    // flash_attn_ext expects:
    //   q: [head_dim, q_len,  n_heads, 1]
    //   k: [head_dim, kv_len, n_heads, 1]
    //   v: [head_dim, kv_len, n_heads, 1]  (NOT transposed)
    //   mask: [kv_len, q_len, 1, 1]  (broadcasts across heads)
    //   result: [head_dim, n_heads, q_len, 1]
    ggml_tensor * qh = ggml_permute(ctx, ggml_reshape_3d(ctx, q, kHeadDim, kHeads, q_len), 0, 2, 1, 3);
    ggml_tensor * kh = ggml_permute(ctx, ggml_reshape_3d(ctx, k, kHeadDim, kHeads, kv_len), 0, 2, 1, 3);
    ggml_tensor * vh = ggml_permute(ctx, ggml_reshape_3d(ctx, v, kHeadDim, kHeads, kv_len), 0, 2, 1, 3);

    ggml_tensor * cur = ggml_flash_attn_ext(
        ctx, qh, kh, vh, mask,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)),
        0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
    // result is [head_dim, n_heads, q_len, 1] -> reshape to [model_dim, q_len]
    return ggml_reshape_2d(ctx, cur, kModelDim, q_len);
}

std::string layer_prefix(int layer) {
    return "transformer.encoder.layers." + std::to_string(layer);
}

std::vector<float> flatten_nchw_to_seq(
    const std::vector<float> & input,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w
) {
    if (n != 1) {
        throw std::runtime_error("encoder_fusion currently expects batch size 1");
    }

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

}  // namespace

EncoderFusion::EncoderFusion(const GgufModel & model) : model_(model) {}

EncoderFusionOutput EncoderFusion::run(
    const std::vector<float> & image_nchw,
    const std::vector<int64_t> & image_shape_nchw,
    const std::vector<float> & pos_nchw,
    const std::vector<int64_t> & pos_shape_nchw,
    const std::vector<float> & prompt,
    const std::vector<int64_t> & prompt_shape,
    const std::vector<float> & prompt_mask,
    const std::vector<int64_t> & prompt_mask_shape
) const {
    if (image_shape_nchw.size() != 4 || pos_shape_nchw.size() != 4) {
        throw std::runtime_error("expected NCHW image and position tensors");
    }
    if (prompt_shape.size() != 3 || prompt_mask_shape.size() != 2) {
        throw std::runtime_error("expected prompt shape [seq, batch, dim] and mask shape [batch, seq]");
    }
    if (image_shape_nchw != pos_shape_nchw) {
        throw std::runtime_error("image and position tensor shapes must match");
    }
    if (image_shape_nchw[0] != 1 || prompt_shape[1] != 1 || prompt_mask_shape[0] != 1) {
        throw std::runtime_error("encoder_fusion currently expects batch size 1");
    }
    if (image_shape_nchw[1] != kModelDim || prompt_shape[2] != kModelDim) {
        throw std::runtime_error("encoder_fusion expects hidden_dim=256");
    }
    if (prompt_shape[0] != prompt_mask_shape[1]) {
        throw std::runtime_error("prompt and prompt_mask sequence lengths must match");
    }
    if (static_cast<int64_t>(image_nchw.size()) != image_shape_nchw[0] * image_shape_nchw[1] * image_shape_nchw[2] * image_shape_nchw[3]) {
        throw std::runtime_error("image_nchw payload size mismatch");
    }
    if (pos_nchw.size() != image_nchw.size()) {
        throw std::runtime_error("position payload size mismatch");
    }
    if (static_cast<int64_t>(prompt.size()) != prompt_shape[0] * prompt_shape[1] * prompt_shape[2]) {
        throw std::runtime_error("prompt payload size mismatch");
    }
    if (static_cast<int64_t>(prompt_mask.size()) != prompt_mask_shape[0] * prompt_mask_shape[1]) {
        throw std::runtime_error("prompt_mask payload size mismatch");
    }

    const int64_t h = image_shape_nchw[2];
    const int64_t w = image_shape_nchw[3];
    const int32_t image_seq_len = static_cast<int32_t>(h * w);
    const int32_t prompt_seq_len = static_cast<int32_t>(prompt_shape[0]);

    const std::vector<float> image_seq = flatten_nchw_to_seq(image_nchw, 1, kModelDim, h, w);
    const std::vector<float> pos_seq = flatten_nchw_to_seq(pos_nchw, 1, kModelDim, h, w);
    const std::vector<ggml_fp16_t> prompt_mask_f16 = make_prompt_mask_f16(prompt_mask, prompt_seq_len, image_seq_len);

    const size_t graph_size = 131072;
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

    ggml_tensor * image = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, image_seq_len);
    ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, image_seq_len);
    ggml_tensor * prompt_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, prompt_seq_len);
    ggml_tensor * prompt_mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, prompt_seq_len, image_seq_len);

    ggml_backend_sched_set_tensor_backend(sched, image, backend);
    ggml_backend_sched_set_tensor_backend(sched, pos, backend);
    ggml_backend_sched_set_tensor_backend(sched, prompt_in, backend);
    ggml_backend_sched_set_tensor_backend(sched, prompt_mask_in, backend);

    ggml_tensor * cur = image;

    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = layer_prefix(layer);

        ggml_tensor * norm1 = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".norm1.weight"),
            require_tensor(model_, prefix + ".norm1.bias")
        );
        ggml_tensor * self_src = ggml_add(ctx, norm1, pos);
        ggml_tensor * self_q = linear(
            ctx,
            require_tensor(model_, prefix + ".self_attn.query_proj.weight"),
            require_tensor(model_, prefix + ".self_attn.query_proj.bias"),
            self_src
        );
        ggml_tensor * self_k = linear(
            ctx,
            require_tensor(model_, prefix + ".self_attn.key_proj.weight"),
            require_tensor(model_, prefix + ".self_attn.key_proj.bias"),
            self_src
        );
        ggml_tensor * self_v = linear(
            ctx,
            require_tensor(model_, prefix + ".self_attn.value_proj.weight"),
            require_tensor(model_, prefix + ".self_attn.value_proj.bias"),
            norm1
        );
        ggml_tensor * self_attn = mha(ctx, self_q, self_k, self_v, image_seq_len, image_seq_len, nullptr);
        self_attn = linear(
            ctx,
            require_tensor(model_, prefix + ".self_attn.out_proj.weight"),
            require_tensor(model_, prefix + ".self_attn.out_proj.bias"),
            self_attn
        );
        cur = ggml_add(ctx, cur, self_attn);

        ggml_tensor * norm2 = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".norm2.weight"),
            require_tensor(model_, prefix + ".norm2.bias")
        );
        ggml_tensor * cross_q = linear(
            ctx,
            require_tensor(model_, prefix + ".cross_attn_image.query_proj.weight"),
            require_tensor(model_, prefix + ".cross_attn_image.query_proj.bias"),
            norm2
        );
        ggml_tensor * cross_k = linear(
            ctx,
            require_tensor(model_, prefix + ".cross_attn_image.key_proj.weight"),
            require_tensor(model_, prefix + ".cross_attn_image.key_proj.bias"),
            prompt_in
        );
        ggml_tensor * cross_v = linear(
            ctx,
            require_tensor(model_, prefix + ".cross_attn_image.value_proj.weight"),
            require_tensor(model_, prefix + ".cross_attn_image.value_proj.bias"),
            prompt_in
        );
        ggml_tensor * cross_attn = mha(ctx, cross_q, cross_k, cross_v, image_seq_len, prompt_seq_len, prompt_mask_in);
        cross_attn = linear(
            ctx,
            require_tensor(model_, prefix + ".cross_attn_image.out_proj.weight"),
            require_tensor(model_, prefix + ".cross_attn_image.out_proj.bias"),
            cross_attn
        );
        cur = ggml_add(ctx, cur, cross_attn);

        ggml_tensor * norm3 = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".norm3.weight"),
            require_tensor(model_, prefix + ".norm3.bias")
        );
        ggml_tensor * ffn = linear(
            ctx,
            require_tensor(model_, prefix + ".linear1.weight"),
            require_tensor(model_, prefix + ".linear1.bias"),
            norm3
        );
        ffn = ggml_relu(ctx, ffn);
        ffn = linear(
            ctx,
            require_tensor(model_, prefix + ".linear2.weight"),
            require_tensor(model_, prefix + ".linear2.bias"),
            ffn
        );
        cur = ggml_add(ctx, cur, ffn);

    }

    ggml_tensor * memory = ggml_cont(ctx, cur);
    ggml_tensor * pos_capture = ggml_cont(ctx, pos);

    ggml_build_forward_expand(gf, pos_capture);
    ggml_build_forward_expand(gf, memory);

    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(image, image_seq.data(), 0, image_seq.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_seq.data(), 0, pos_seq.size() * sizeof(float));
    ggml_backend_tensor_set(prompt_in, prompt.data(), 0, prompt.size() * sizeof(float));
    ggml_backend_tensor_set(prompt_mask_in, prompt_mask_f16.data(), 0, prompt_mask_f16.size() * sizeof(ggml_fp16_t));

    const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        throw std::runtime_error("encoder_fusion graph compute failed");
    }

    EncoderFusionOutput out;
    out.image_seq_len = image_seq_len;
    out.prompt_seq_len = prompt_seq_len;
    out.hidden_dim = kModelDim;
    out.memory.resize(static_cast<size_t>(kModelDim * image_seq_len));
    out.pos_embed.resize(static_cast<size_t>(kModelDim * image_seq_len));
    ggml_backend_tensor_get(memory, out.memory.data(), 0, ggml_nbytes(memory));
    ggml_backend_tensor_get(pos_capture, out.pos_embed.data(), 0, ggml_nbytes(pos_capture));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return out;
}

}  // namespace sam3
