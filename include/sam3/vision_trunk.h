#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

struct VisionTrunkOutput {
    std::vector<float> trunk_nchw;
    std::vector<int64_t> shape_nchw;
};

class VisionTrunk {
public:
    explicit VisionTrunk(const GgufModel & model);

    VisionTrunkOutput run(
        const std::vector<float> & image_nchw,
        const std::vector<int64_t> & shape_nchw
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
