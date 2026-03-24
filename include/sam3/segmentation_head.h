#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

struct SegmentationHeadOutput {
    int64_t batch = 0;
    int64_t num_queries = 0;
    int64_t hidden_dim = 0;
    int64_t height = 0;
    int64_t width = 0;
    std::vector<float> pred_masks;
    std::vector<float> semantic_seg;
};

class SegmentationHead {
public:
    explicit SegmentationHead(const GgufModel & model);

    SegmentationHeadOutput run(
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
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
