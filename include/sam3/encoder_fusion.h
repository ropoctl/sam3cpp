#pragma once

#include "sam3/gguf_model.h"

#include <vector>

namespace sam3 {

struct EncoderFusionOutput {
    int64_t image_seq_len = 0;
    int64_t prompt_seq_len = 0;
    int64_t hidden_dim = 0;
    std::vector<float> memory;
    std::vector<float> pos_embed;
};

class EncoderFusion {
public:
    explicit EncoderFusion(const GgufModel & model);

    EncoderFusionOutput run(
        const std::vector<float> & image_nchw,
        const std::vector<int64_t> & image_shape_nchw,
        const std::vector<float> & pos_nchw,
        const std::vector<int64_t> & pos_shape_nchw,
        const std::vector<float> & prompt,
        const std::vector<int64_t> & prompt_shape,
        const std::vector<float> & prompt_mask,
        const std::vector<int64_t> & prompt_mask_shape
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
