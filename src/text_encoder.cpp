#include "sam3/text_encoder.h"

#include "ggml-alloc.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kTextWidth = 1024;
constexpr int32_t kTextHeads = 16;
constexpr int32_t kTextLayers = 24;
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

ggml_tensor * self_attention(
    ggml_context * ctx,
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    int32_t seq_len
) {
    const int32_t head_dim = kTextWidth / kTextHeads;

    ggml_tensor * qh = ggml_permute(ctx, ggml_cont_3d(ctx, q, head_dim, kTextHeads, seq_len), 0, 2, 1, 3);
    ggml_tensor * kh = ggml_permute(ctx, ggml_cont_3d(ctx, k, head_dim, kTextHeads, seq_len), 0, 2, 1, 3);
    ggml_tensor * vh = ggml_cont_3d(
        ctx,
        ggml_permute(ctx, ggml_cont_3d(ctx, v, head_dim, kTextHeads, seq_len), 1, 2, 0, 3),
        seq_len,
        head_dim,
        kTextHeads
    );

    ggml_tensor * kq = ensure_f32(ctx, ggml_mul_mat(ctx, kh, qh));
    ggml_tensor * scaled = ggml_scale(ctx, kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    ggml_tensor * masked = ggml_diag_mask_inf(ctx, scaled, 0);
    ggml_tensor * probs = ggml_soft_max(ctx, masked);
    ggml_tensor * kqv = ensure_f32(ctx, ggml_mul_mat(ctx, vh, probs));
    ggml_tensor * merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, merged, kTextWidth, seq_len);
}

std::string block_prefix(int layer) {
    return "backbone.language_backbone.encoder.transformer.resblocks." + std::to_string(layer);
}

}  // namespace

TextEncoder::TextEncoder(const GgufModel & model) : model_(model) {}

TextEncodeOutput TextEncoder::encode(const std::vector<int32_t> & tokens) const {
    if (tokens.empty()) {
        throw std::runtime_error("no tokens provided");
    }

    const int32_t seq_len = static_cast<int32_t>(tokens.size());
    std::vector<int32_t> positions(tokens.size());
    for (int32_t i = 0; i < seq_len; ++i) {
        positions[i] = i;
    }

    const size_t graph_size = 32768;
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
    const bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_t cpu_backend = is_cpu ? backend : model_.cpu_backend();
    ggml_backend_t backends[2] = { backend, cpu_backend };
    const int n_backends = (is_cpu || cpu_backend == nullptr) ? 1 : 2;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
    if (sched == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to create backend scheduler");
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(token_ids, "token_ids");
    ggml_backend_sched_set_tensor_backend(sched, token_ids, cpu_backend);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos_ids, "position_ids");
    ggml_backend_sched_set_tensor_backend(sched, pos_ids, cpu_backend);

    ggml_tensor * tok = ggml_get_rows(ctx, require_tensor(model_, "backbone.language_backbone.encoder.token_embedding.weight"), token_ids);
    ggml_tensor * pos = ggml_get_rows(ctx, require_tensor(model_, "backbone.language_backbone.encoder.positional_embedding"), pos_ids);
    ggml_tensor * cur = ggml_add(ctx, tok, pos);
    const bool mixed_backend = n_backends > 1;

    for (int layer = 0; layer < kTextLayers; ++layer) {
        const std::string prefix = block_prefix(layer);

        ggml_tensor * ln1 = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".ln_1.weight"),
            require_tensor(model_, prefix + ".ln_1.bias")
        );
        if (mixed_backend) {
            ggml_backend_sched_set_tensor_backend(sched, ln1, cpu_backend);
        }

        ggml_tensor * q = linear(
            ctx,
            require_tensor(model_, prefix + ".attn.query_proj.weight"),
            require_tensor(model_, prefix + ".attn.query_proj.bias"),
            ln1
        );
        ggml_tensor * k = linear(
            ctx,
            require_tensor(model_, prefix + ".attn.key_proj.weight"),
            require_tensor(model_, prefix + ".attn.key_proj.bias"),
            ln1
        );
        ggml_tensor * v = linear(
            ctx,
            require_tensor(model_, prefix + ".attn.value_proj.weight"),
            require_tensor(model_, prefix + ".attn.value_proj.bias"),
            ln1
        );
        ggml_tensor * attn = self_attention(ctx, q, k, v, seq_len);
        attn = linear(
            ctx,
            require_tensor(model_, prefix + ".attn.out_proj.weight"),
            require_tensor(model_, prefix + ".attn.out_proj.bias"),
            attn
        );
        cur = ggml_add(ctx, cur, attn);

        ggml_tensor * ln2 = layer_norm(
            ctx,
            cur,
            require_tensor(model_, prefix + ".ln_2.weight"),
            require_tensor(model_, prefix + ".ln_2.bias")
        );
        if (mixed_backend) {
            ggml_backend_sched_set_tensor_backend(sched, ln2, cpu_backend);
        }
        ggml_tensor * mlp = linear(
            ctx,
            require_tensor(model_, prefix + ".mlp.c_fc.weight"),
            require_tensor(model_, prefix + ".mlp.c_fc.bias"),
            ln2
        );
        mlp = ggml_gelu_erf(ctx, mlp);
        if (mixed_backend) {
            ggml_backend_sched_set_tensor_backend(sched, mlp, cpu_backend);
        }
        mlp = linear(
            ctx,
            require_tensor(model_, prefix + ".mlp.c_proj.weight"),
            require_tensor(model_, prefix + ".mlp.c_proj.bias"),
            mlp
        );
        cur = ggml_add(ctx, cur, mlp);
    }

    ggml_tensor * ln_final = layer_norm(
        ctx,
        cur,
        require_tensor(model_, "backbone.language_backbone.encoder.ln_final.weight"),
        require_tensor(model_, "backbone.language_backbone.encoder.ln_final.bias")
    );
    if (mixed_backend) {
        ggml_backend_sched_set_tensor_backend(sched, ln_final, cpu_backend);
    }
    ggml_tensor * resized = linear(
        ctx,
        require_tensor(model_, "backbone.language_backbone.resizer.weight"),
        require_tensor(model_, "backbone.language_backbone.resizer.bias"),
        ln_final
    );
    ggml_build_forward_expand(gf, resized);

    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(token_ids, tokens.data(), 0, tokens.size() * sizeof(int32_t));
    ggml_backend_tensor_set(pos_ids, positions.data(), 0, positions.size() * sizeof(int32_t));

    const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        throw std::runtime_error("graph compute failed");
    }

    std::vector<float> raw(static_cast<size_t>(ggml_nelements(resized)));
    ggml_backend_tensor_get(resized, raw.data(), 0, ggml_nbytes(resized));

    TextEncodeOutput output;
    output.seq_len = seq_len;
    output.hidden_dim = 256;
    output.memory.resize(static_cast<size_t>(seq_len) * output.hidden_dim);

    for (int32_t s = 0; s < seq_len; ++s) {
        for (int32_t d = 0; d < output.hidden_dim; ++d) {
            output.memory[static_cast<size_t>(s) * output.hidden_dim + d] =
                raw[static_cast<size_t>(s) * output.hidden_dim + d];
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return output;
}

}  // namespace sam3
