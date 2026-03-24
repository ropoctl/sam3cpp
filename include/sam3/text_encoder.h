#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

struct TextEncodeOutput {
    int64_t seq_len = 0;
    int64_t hidden_dim = 0;
    std::vector<float> memory;
};

class TextEncoder {
public:
    explicit TextEncoder(const GgufModel & model);

    TextEncodeOutput encode(const std::vector<int32_t> & tokens) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
