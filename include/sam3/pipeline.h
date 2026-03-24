#pragma once

#include "sam3/dummy_prompt.h"
#include "sam3/geometry_encoder.h"
#include "sam3/encoder_fusion.h"
#include "sam3/decoder.h"
#include "sam3/gguf_model.h"
#include "sam3/grounding_head.h"
#include "sam3/segmentation_head.h"
#include "sam3/text_encoder.h"
#include "sam3/tokenizer.h"
#include "sam3/vision_neck.h"
#include "sam3/vision_trunk.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sam3 {

/// Control ggml log verbosity. 0 = silent (default), 1+ = ggml default logging.
void set_log_level(int level);

struct Sam3Prediction {
    int32_t width = 0;
    int32_t height = 0;
    int32_t count = 0;
    std::vector<float> scores;
    std::vector<float> boxes_xyxy;
    std::vector<float> masks;
};

class Sam3ImagePipeline {
public:
    Sam3ImagePipeline(const std::string & gguf_path, bool prefer_gpu = true, const std::string & bpe_path = {});

    Sam3Prediction predict(const std::string & image_path, const std::string & prompt) const;
    Sam3Prediction predict_tokens(const std::string & image_path, const std::vector<int32_t> & tokens) const;

    /// Point-prompted segmentation. points_xy is [count*2] normalized [0,1], labels is [count] (1=pos, 0=neg).
    Sam3Prediction predict_points(const std::string & image_path,
        const float * points_xy, const int32_t * labels, int32_t count) const;

    /// Box-prompted segmentation. Coordinates in pixels (xyxy format).
    Sam3Prediction predict_box(const std::string & image_path,
        float x1, float y1, float x2, float y2) const;

private:
    GgufModel model_;
    SimpleTokenizer tokenizer_;
};

}  // namespace sam3
