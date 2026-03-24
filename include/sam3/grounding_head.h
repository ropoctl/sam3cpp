#pragma once

#include "sam3/gguf_model.h"

#include <vector>

namespace sam3 {

struct GroundingHeadOutput {
    int64_t num_layers = 0;
    int64_t num_queries = 0;
    std::vector<std::vector<float>> pred_logits;
    std::vector<std::vector<float>> pred_boxes;
};

class GroundingHead {
public:
    explicit GroundingHead(const GgufModel & model);

    GroundingHeadOutput run(
        const std::vector<std::vector<float>> & hs_layers,
        const std::vector<std::vector<float>> & ref_layers,
        const std::vector<float> & prompt,
        const std::vector<int64_t> & prompt_shape,
        const std::vector<float> & prompt_mask,
        const std::vector<int64_t> & prompt_mask_shape
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
