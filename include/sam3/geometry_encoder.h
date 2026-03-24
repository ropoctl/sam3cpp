#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

struct GeometryEncoderOutput {
    int64_t geo_seq_len = 0;
    int64_t hidden_dim = 0;
    std::vector<float> geo_tokens;
    std::vector<float> geo_mask;
};

class GeometryEncoder {
public:
    explicit GeometryEncoder(const GgufModel & model);

    /// Encode point and/or box prompts into geometry tokens.
    /// points_xy: [count * 2] array of (x, y) pairs, normalized [0, 1].
    /// labels:    [count] array, 1 = positive, 0 = negative.
    /// boxes:     [box_count * 4] array of (cx, cy, w, h), normalized [0, 1].
    /// box_labels: [box_count] array, 1 = positive, 0 = negative.
    GeometryEncoderOutput run(
        const float * points_xy, const int32_t * labels, int32_t point_count,
        const float * boxes_cxcywh, const int32_t * box_labels, int32_t box_count,
        const std::vector<float> & image_nchw,
        const std::vector<int64_t> & image_shape_nchw,
        const std::vector<float> & pos_nchw,
        const std::vector<int64_t> & pos_shape_nchw
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
