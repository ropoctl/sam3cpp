#include "sam3/grounding_head.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kModelDim = 256;
constexpr int32_t kPromptMlpHidden = 2048;
constexpr float kLayerNormEps = 1e-5f;
constexpr float kInvSigmoidEps = 1e-3f;
constexpr float kClampLogit = 12.0f;

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * tensor = model.find_weight(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return tensor;
}

std::vector<float> load_tensor_f32(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = require_tensor(model, name);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> linear_rows(
    const std::vector<float> & x,
    int64_t rows,
    int64_t in_dim,
    const std::vector<float> & weight,
    int64_t out_dim,
    const std::vector<float> & bias
) {
    std::vector<float> out(static_cast<size_t>(rows * out_dim));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t o = 0; o < out_dim; ++o) {
            float sum = bias[static_cast<size_t>(o)];
            for (int64_t i = 0; i < in_dim; ++i) {
                sum += x[static_cast<size_t>(r * in_dim + i)] * weight[static_cast<size_t>(o * in_dim + i)];
            }
            out[static_cast<size_t>(r * out_dim + o)] = sum;
        }
    }
    return out;
}

void relu_inplace(std::vector<float> & x) {
    for (float & v : x) {
        v = std::max(0.0f, v);
    }
}

void layer_norm_rows_inplace(
    std::vector<float> & x,
    int64_t rows,
    int64_t dim,
    const std::vector<float> & weight,
    const std::vector<float> & bias
) {
    for (int64_t r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            mean += x[static_cast<size_t>(r * dim + i)];
        }
        mean /= static_cast<float>(dim);
        float var = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            const float d = x[static_cast<size_t>(r * dim + i)] - mean;
            var += d * d;
        }
        var /= static_cast<float>(dim);
        const float inv = 1.0f / std::sqrt(var + kLayerNormEps);
        for (int64_t i = 0; i < dim; ++i) {
            const float norm = (x[static_cast<size_t>(r * dim + i)] - mean) * inv;
            x[static_cast<size_t>(r * dim + i)] = norm * weight[static_cast<size_t>(i)] + bias[static_cast<size_t>(i)];
        }
    }
}

std::vector<float> inverse_sigmoid_vec(const std::vector<float> & x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float clipped = std::min(1.0f - kInvSigmoidEps, std::max(kInvSigmoidEps, x[i]));
        out[i] = std::log(clipped / (1.0f - clipped));
    }
    return out;
}

std::vector<float> sigmoid_vec(const std::vector<float> & x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    return out;
}

}  // namespace

GroundingHead::GroundingHead(const GgufModel & model) : model_(model) {}

GroundingHeadOutput GroundingHead::run(
    const std::vector<std::vector<float>> & hs_layers,
    const std::vector<std::vector<float>> & ref_layers,
    const std::vector<float> & prompt,
    const std::vector<int64_t> & prompt_shape,
    const std::vector<float> & prompt_mask,
    const std::vector<int64_t> & prompt_mask_shape
) const {
    if (hs_layers.size() != ref_layers.size()) {
        throw std::runtime_error("hs_layers and ref_layers size mismatch");
    }
    if (prompt_shape.size() != 3 || prompt_mask_shape.size() != 2 || prompt_shape[1] != 1 || prompt_mask_shape[0] != 1 || prompt_shape[2] != kModelDim) {
        throw std::runtime_error("invalid prompt shapes");
    }

    const int64_t seq = prompt_shape[0];
    const std::vector<float> prompt_mlp_w0 = load_tensor_f32(model_, "dot_prod_scoring.prompt_mlp.layers.0.weight");
    const std::vector<float> prompt_mlp_b0 = load_tensor_f32(model_, "dot_prod_scoring.prompt_mlp.layers.0.bias");
    const std::vector<float> prompt_mlp_w1 = load_tensor_f32(model_, "dot_prod_scoring.prompt_mlp.layers.1.weight");
    const std::vector<float> prompt_mlp_b1 = load_tensor_f32(model_, "dot_prod_scoring.prompt_mlp.layers.1.bias");
    const std::vector<float> prompt_mlp_ln_w = load_tensor_f32(model_, "dot_prod_scoring.prompt_mlp.out_norm.weight");
    const std::vector<float> prompt_mlp_ln_b = load_tensor_f32(model_, "dot_prod_scoring.prompt_mlp.out_norm.bias");
    const std::vector<float> prompt_proj_w = load_tensor_f32(model_, "dot_prod_scoring.prompt_proj.weight");
    const std::vector<float> prompt_proj_b = load_tensor_f32(model_, "dot_prod_scoring.prompt_proj.bias");
    const std::vector<float> hs_proj_w = load_tensor_f32(model_, "dot_prod_scoring.hs_proj.weight");
    const std::vector<float> hs_proj_b = load_tensor_f32(model_, "dot_prod_scoring.hs_proj.bias");
    const std::vector<float> bbox_w0 = load_tensor_f32(model_, "transformer.decoder.bbox_embed.layers.0.weight");
    const std::vector<float> bbox_b0 = load_tensor_f32(model_, "transformer.decoder.bbox_embed.layers.0.bias");
    const std::vector<float> bbox_w1 = load_tensor_f32(model_, "transformer.decoder.bbox_embed.layers.1.weight");
    const std::vector<float> bbox_b1 = load_tensor_f32(model_, "transformer.decoder.bbox_embed.layers.1.bias");
    const std::vector<float> bbox_w2 = load_tensor_f32(model_, "transformer.decoder.bbox_embed.layers.2.weight");
    const std::vector<float> bbox_b2 = load_tensor_f32(model_, "transformer.decoder.bbox_embed.layers.2.bias");

    std::vector<float> prompt_proj_tokens = linear_rows(prompt, seq, kModelDim, prompt_mlp_w0, kPromptMlpHidden, prompt_mlp_b0);
    relu_inplace(prompt_proj_tokens);
    prompt_proj_tokens = linear_rows(prompt_proj_tokens, seq, kPromptMlpHidden, prompt_mlp_w1, kModelDim, prompt_mlp_b1);
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t d = 0; d < kModelDim; ++d) {
            prompt_proj_tokens[static_cast<size_t>(s * kModelDim + d)] += prompt[static_cast<size_t>(s * kModelDim + d)];
        }
    }
    layer_norm_rows_inplace(prompt_proj_tokens, seq, kModelDim, prompt_mlp_ln_w, prompt_mlp_ln_b);

    std::vector<float> pooled(kModelDim, 0.0f);
    float valid = 0.0f;
    for (int64_t s = 0; s < seq; ++s) {
        const float is_valid = prompt_mask[static_cast<size_t>(s)] > 0.5f ? 0.0f : 1.0f;
        valid += is_valid;
        for (int64_t d = 0; d < kModelDim; ++d) {
            pooled[static_cast<size_t>(d)] += prompt_proj_tokens[static_cast<size_t>(s * kModelDim + d)] * is_valid;
        }
    }
    valid = std::max(1.0f, valid);
    for (float & v : pooled) {
        v /= valid;
    }

    std::vector<float> pooled_proj = linear_rows(pooled, 1, kModelDim, prompt_proj_w, kModelDim, prompt_proj_b);

    GroundingHeadOutput out;
    out.num_layers = static_cast<int64_t>(hs_layers.size());
    out.num_queries = hs_layers.empty() ? 0 : static_cast<int64_t>(hs_layers[0].size() / kModelDim);
    out.pred_logits.resize(hs_layers.size());
    out.pred_boxes.resize(hs_layers.size());

    for (size_t layer = 0; layer < hs_layers.size(); ++layer) {
        const int64_t num_queries = static_cast<int64_t>(hs_layers[layer].size() / kModelDim);
        std::vector<float> hs_proj = linear_rows(hs_layers[layer], num_queries, kModelDim, hs_proj_w, kModelDim, hs_proj_b);
        out.pred_logits[layer].resize(static_cast<size_t>(num_queries));
        for (int64_t q = 0; q < num_queries; ++q) {
            float dot = 0.0f;
            for (int64_t d = 0; d < kModelDim; ++d) {
                dot += hs_proj[static_cast<size_t>(q * kModelDim + d)] * pooled_proj[static_cast<size_t>(d)];
            }
            dot *= 1.0f / std::sqrt(static_cast<float>(kModelDim));
            out.pred_logits[layer][static_cast<size_t>(q)] = std::max(-kClampLogit, std::min(kClampLogit, dot));
        }

        std::vector<float> delta = linear_rows(hs_layers[layer], num_queries, kModelDim, bbox_w0, kModelDim, bbox_b0);
        relu_inplace(delta);
        delta = linear_rows(delta, num_queries, kModelDim, bbox_w1, kModelDim, bbox_b1);
        relu_inplace(delta);
        delta = linear_rows(delta, num_queries, kModelDim, bbox_w2, 4, bbox_b2);
        const std::vector<float> ref_inv = inverse_sigmoid_vec(ref_layers[layer]);
        for (size_t i = 0; i < delta.size(); ++i) {
            delta[i] += ref_inv[i];
        }
        out.pred_boxes[layer] = sigmoid_vec(delta);
    }

    return out;
}

}  // namespace sam3
