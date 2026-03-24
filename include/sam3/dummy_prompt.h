#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

struct DummyPromptOutput {
    int64_t text_seq_len = 0;
    int64_t prompt_seq_len = 0;
    int64_t hidden_dim = 0;
    std::vector<float> geo_token;
    std::vector<float> prompt;
    std::vector<float> prompt_mask;
};

class DummyPromptEncoder {
public:
    explicit DummyPromptEncoder(const GgufModel & model);

    DummyPromptOutput run(
        const std::vector<float> & image_nchw,
        const std::vector<int64_t> & image_shape_nchw,
        const std::vector<float> & pos_nchw,
        const std::vector<int64_t> & pos_shape_nchw,
        const std::vector<float> & text,
        const std::vector<int64_t> & text_shape,
        const std::vector<float> & text_mask,
        const std::vector<int64_t> & text_mask_shape
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
