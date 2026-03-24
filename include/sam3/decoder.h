#pragma once

#include "sam3/gguf_model.h"

#include <vector>

namespace sam3 {

struct DecoderOutput {
    int64_t num_layers = 0;
    int64_t num_queries = 0;
    int64_t hidden_dim = 0;
    std::vector<std::vector<float>> hs;
    std::vector<std::vector<float>> reference_boxes;
    std::vector<std::vector<float>> presence_logits;
};

class Decoder {
public:
    explicit Decoder(const GgufModel & model);

    DecoderOutput run(
        const std::vector<float> & memory,
        const std::vector<int64_t> & memory_shape,
        const std::vector<float> & pos_embed,
        const std::vector<int64_t> & pos_shape,
        const std::vector<float> & prompt,
        const std::vector<int64_t> & prompt_shape,
        const std::vector<float> & prompt_mask,
        const std::vector<int64_t> & prompt_mask_shape
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
