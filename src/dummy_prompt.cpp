#include "sam3/dummy_prompt.h"

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

std::vector<float> flatten_nchw_to_seq(
    const std::vector<float> & input,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w
) {
    if (n != 1) {
        throw std::runtime_error("dummy prompt currently expects batch size 1");
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

std::string layer_prefix(int layer) {
    return "geometry_encoder.encode." + std::to_string(layer);
}

}  // namespace

DummyPromptEncoder::DummyPromptEncoder(const GgufModel & model) : model_(model) {}

DummyPromptOutput DummyPromptEncoder::run(
    const std::vector<float> & image_nchw,
    const std::vector<int64_t> & image_shape_nchw,
    const std::vector<float> & pos_nchw,
    const std::vector<int64_t> & pos_shape_nchw,
    const std::vector<float> & text,
    const std::vector<int64_t> & text_shape,
    const std::vector<float> & text_mask,
    const std::vector<int64_t> & text_mask_shape
) const {
    if (image_shape_nchw.size() != 4 || pos_shape_nchw.size() != 4) {
        throw std::runtime_error("expected NCHW image and position tensors");
    }
    if (image_shape_nchw != pos_shape_nchw) {
        throw std::runtime_error("image and position tensor shapes must match");
    }
    if (image_shape_nchw[0] != 1 || image_shape_nchw[1] != kModelDim) {
        throw std::runtime_error("dummy prompt expects image tensors with shape [1, 256, H, W]");
    }
    if (text_shape.size() != 3 || text_mask_shape.size() != 2) {
        throw std::runtime_error("expected text shape [seq, batch, dim] and mask shape [batch, seq]");
    }
    if (text_shape[1] != 1 || text_shape[2] != kModelDim || text_mask_shape[0] != 1 || text_mask_shape[1] != text_shape[0]) {
        throw std::runtime_error("invalid text or text mask shape");
    }

    const int64_t h = image_shape_nchw[2];
    const int64_t w = image_shape_nchw[3];
    const int32_t image_seq_len = static_cast<int32_t>(h * w);
    const int32_t text_seq_len = static_cast<int32_t>(text_shape[0]);

    const std::vector<float> image_seq = flatten_nchw_to_seq(image_nchw, 1, kModelDim, h, w);
    const std::vector<float> pos_seq = flatten_nchw_to_seq(pos_nchw, 1, kModelDim, h, w);

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

    ggml_tensor * image = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, image_seq_len);
    ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, image_seq_len);
    ggml_backend_sched_set_tensor_backend(sched, image, backend);
    ggml_backend_sched_set_tensor_backend(sched, pos, backend);

    ggml_tensor * cur = ggml_view_2d(ctx, require_tensor(model_, "geometry_encoder.cls_embed.weight"), kModelDim, 1, require_tensor(model_, "geometry_encoder.cls_embed.weight")->nb[1], 0);
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

    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = layer_prefix(layer);
        ggml_tensor * norm1 = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".norm1.weight"),
            require_tensor(model_, prefix + ".norm1.bias")
        );
        ggml_tensor * self_out = mha(
            ctx,
            linear(ctx, require_tensor(model_, prefix + ".self_attn.query_proj.weight"), require_tensor(model_, prefix + ".self_attn.query_proj.bias"), norm1),
            linear(ctx, require_tensor(model_, prefix + ".self_attn.key_proj.weight"), require_tensor(model_, prefix + ".self_attn.key_proj.bias"), norm1),
            linear(ctx, require_tensor(model_, prefix + ".self_attn.value_proj.weight"), require_tensor(model_, prefix + ".self_attn.value_proj.bias"), norm1),
            1,
            1,
            nullptr
        );
        self_out = linear(ctx, require_tensor(model_, prefix + ".self_attn.out_proj.weight"), require_tensor(model_, prefix + ".self_attn.out_proj.bias"), self_out);
        cur = ggml_add(ctx, cur, self_out);

        ggml_tensor * norm2 = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".norm2.weight"),
            require_tensor(model_, prefix + ".norm2.bias")
        );
        ggml_tensor * cross_out = mha(
            ctx,
            linear(ctx, require_tensor(model_, prefix + ".cross_attn_image.query_proj.weight"), require_tensor(model_, prefix + ".cross_attn_image.query_proj.bias"), norm2),
            linear(ctx, require_tensor(model_, prefix + ".cross_attn_image.key_proj.weight"), require_tensor(model_, prefix + ".cross_attn_image.key_proj.bias"), ggml_add(ctx, image, pos)),
            linear(ctx, require_tensor(model_, prefix + ".cross_attn_image.value_proj.weight"), require_tensor(model_, prefix + ".cross_attn_image.value_proj.bias"), image),
            1,
            image_seq_len,
            nullptr
        );
        cross_out = linear(ctx, require_tensor(model_, prefix + ".cross_attn_image.out_proj.weight"), require_tensor(model_, prefix + ".cross_attn_image.out_proj.bias"), cross_out);
        cur = ggml_add(ctx, cur, cross_out);

        ggml_tensor * norm3 = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".norm3.weight"),
            require_tensor(model_, prefix + ".norm3.bias")
        );
        ggml_tensor * ffn = linear(ctx, require_tensor(model_, prefix + ".linear1.weight"), require_tensor(model_, prefix + ".linear1.bias"), norm3);
        ffn = relu(ctx, ffn);
        ffn = linear(ctx, require_tensor(model_, prefix + ".linear2.weight"), require_tensor(model_, prefix + ".linear2.bias"), ffn);
        cur = ggml_add(ctx, cur, ffn);
    }

    ggml_tensor * geo_token = layer_norm(
        ctx,
        cur,
        require_tensor(model_, "geometry_encoder.encode_norm.weight"),
        require_tensor(model_, "geometry_encoder.encode_norm.bias")
    );
    ggml_tensor * geo_capture = ggml_cont(ctx, geo_token);
    ggml_build_forward_expand(gf, geo_capture);

    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(image, image_seq.data(), 0, image_seq.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_seq.data(), 0, pos_seq.size() * sizeof(float));

    const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        throw std::runtime_error("dummy prompt graph compute failed");
    }

    DummyPromptOutput out;
    out.text_seq_len = text_seq_len;
    out.prompt_seq_len = text_seq_len + 1;
    out.hidden_dim = kModelDim;
    out.geo_token.resize(static_cast<size_t>(kModelDim));
    ggml_backend_tensor_get(geo_capture, out.geo_token.data(), 0, out.geo_token.size() * sizeof(float));

    out.prompt.resize(static_cast<size_t>(out.prompt_seq_len * kModelDim));
    for (int32_t s = 0; s < text_seq_len; ++s) {
        const size_t src_off = static_cast<size_t>(s * kModelDim);
        const size_t dst_off = src_off;
        for (int32_t c = 0; c < kModelDim; ++c) {
            out.prompt[dst_off + static_cast<size_t>(c)] = text[src_off + static_cast<size_t>(c)];
        }
    }
    for (int32_t c = 0; c < kModelDim; ++c) {
        out.prompt[static_cast<size_t>(text_seq_len * kModelDim + c)] = out.geo_token[static_cast<size_t>(c)];
    }

    out.prompt_mask.resize(static_cast<size_t>(out.prompt_seq_len));
    for (int32_t i = 0; i < text_seq_len; ++i) {
        out.prompt_mask[static_cast<size_t>(i)] = text_mask[static_cast<size_t>(i)];
    }
    out.prompt_mask.back() = 0.0f;

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return out;
}

}  // namespace sam3
