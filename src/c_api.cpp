#include "sam3/c_api.h"
#include "sam3/pipeline.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct sam3_handle {
    explicit sam3_handle(std::unique_ptr<sam3::Sam3ImagePipeline> && pipeline_) : pipeline(std::move(pipeline_)) {}
    std::unique_ptr<sam3::Sam3ImagePipeline> pipeline;
};

namespace {

thread_local std::string g_last_error;

void set_error(const std::string & error) {
    g_last_error = error;
}

int fill_result(const sam3::Sam3Prediction & pred, sam3_result * out_result) {
    if (out_result == nullptr) {
        set_error("out_result is null");
        return -1;
    }

    std::memset(out_result, 0, sizeof(*out_result));
    out_result->width = pred.width;
    out_result->height = pred.height;
    out_result->count = pred.count;

    const size_t scores_bytes = pred.scores.size() * sizeof(float);
    const size_t boxes_bytes = pred.boxes_xyxy.size() * sizeof(float);
    const size_t masks_bytes = pred.masks.size() * sizeof(float);

    out_result->scores = static_cast<float *>(std::malloc(scores_bytes > 0 ? scores_bytes : 1));
    out_result->boxes_xyxy = static_cast<float *>(std::malloc(boxes_bytes > 0 ? boxes_bytes : 1));
    out_result->masks = static_cast<float *>(std::malloc(masks_bytes > 0 ? masks_bytes : 1));
    if (out_result->scores == nullptr || out_result->boxes_xyxy == nullptr || out_result->masks == nullptr) {
        sam3_result_free(out_result);
        set_error("allocation failed");
        return -1;
    }

    if (!pred.scores.empty()) {
        std::memcpy(out_result->scores, pred.scores.data(), scores_bytes);
    }
    if (!pred.boxes_xyxy.empty()) {
        std::memcpy(out_result->boxes_xyxy, pred.boxes_xyxy.data(), boxes_bytes);
    }
    if (!pred.masks.empty()) {
        std::memcpy(out_result->masks, pred.masks.data(), masks_bytes);
    }

    return 0;
}

}  // namespace

extern "C" {

void sam3_set_log_level(int level) {
    sam3::set_log_level(level);
}

sam3_handle * sam3_create(const char * gguf_path, const char * bpe_path, int prefer_gpu) {
    try {
        if (gguf_path == nullptr) {
            throw std::runtime_error("gguf_path is null");
        }
        std::unique_ptr<sam3::Sam3ImagePipeline> pipeline = std::make_unique<sam3::Sam3ImagePipeline>(
            gguf_path,
            prefer_gpu != 0,
            bpe_path != nullptr ? std::string(bpe_path) : std::string());
        return new sam3_handle(std::move(pipeline));
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    }
}

void sam3_destroy(sam3_handle * handle) {
    delete handle;
}

int sam3_predict(sam3_handle * handle, const char * image_path, const char * prompt, sam3_result * out_result) {
    try {
        if (handle == nullptr || handle->pipeline == nullptr) {
            throw std::runtime_error("handle is null");
        }
        if (image_path == nullptr || prompt == nullptr) {
            throw std::runtime_error("image_path or prompt is null");
        }
        const sam3::Sam3Prediction pred = handle->pipeline->predict(image_path, prompt);
        return fill_result(pred, out_result);
    } catch (const std::exception & e) {
        set_error(e.what());
        return -1;
    }
}

int sam3_predict_tokens(
    sam3_handle * handle,
    const char * image_path,
    const int32_t * tokens,
    int32_t token_count,
    sam3_result * out_result
) {
    try {
        if (handle == nullptr || handle->pipeline == nullptr) {
            throw std::runtime_error("handle is null");
        }
        if (image_path == nullptr || tokens == nullptr || token_count <= 0) {
            throw std::runtime_error("invalid token prediction arguments");
        }
        std::vector<int32_t> token_vec(tokens, tokens + token_count);
        const sam3::Sam3Prediction pred = handle->pipeline->predict_tokens(image_path, token_vec);
        return fill_result(pred, out_result);
    } catch (const std::exception & e) {
        set_error(e.what());
        return -1;
    }
}

int sam3_predict_points(
    sam3_handle * handle,
    const char * image_path,
    const float * points_xy,
    const int32_t * labels,
    int32_t point_count,
    sam3_result * out_result
) {
    try {
        if (handle == nullptr || handle->pipeline == nullptr) {
            throw std::runtime_error("handle is null");
        }
        if (image_path == nullptr || points_xy == nullptr || labels == nullptr || point_count <= 0) {
            throw std::runtime_error("invalid point prediction arguments");
        }
        const sam3::Sam3Prediction pred = handle->pipeline->predict_points(image_path, points_xy, labels, point_count);
        return fill_result(pred, out_result);
    } catch (const std::exception & e) {
        set_error(e.what());
        return -1;
    }
}

int sam3_predict_box(
    sam3_handle * handle,
    const char * image_path,
    float x1, float y1, float x2, float y2,
    sam3_result * out_result
) {
    try {
        if (handle == nullptr || handle->pipeline == nullptr) {
            throw std::runtime_error("handle is null");
        }
        if (image_path == nullptr) {
            throw std::runtime_error("image_path is null");
        }
        const sam3::Sam3Prediction pred = handle->pipeline->predict_box(image_path, x1, y1, x2, y2);
        return fill_result(pred, out_result);
    } catch (const std::exception & e) {
        set_error(e.what());
        return -1;
    }
}

void sam3_result_free(sam3_result * result) {
    if (result == nullptr) {
        return;
    }
    std::free(result->scores);
    std::free(result->boxes_xyxy);
    std::free(result->masks);
    std::memset(result, 0, sizeof(*result));
}

const char * sam3_get_last_error(void) {
    return g_last_error.c_str();
}

}  // extern "C"
