#pragma once

#include "sam3/gguf_model.h"

#include <optional>
#include <vector>

namespace sam3 {

struct VisionNeckOutput {
    std::vector<std::vector<float>> levels;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<float>> positions;
    std::vector<std::vector<int64_t>> position_shapes;
};

class VisionNeck {
public:
    explicit VisionNeck(const GgufModel & model);

    VisionNeckOutput run(
        const std::vector<float> & trunk_nchw,
        const std::vector<int64_t> & shape_nchw,
        std::optional<int> only_level = std::nullopt
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
