#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sam3_handle sam3_handle;

typedef struct sam3_result {
    int32_t width;
    int32_t height;
    int32_t count;
    float * scores;
    float * boxes_xyxy;
    float * masks;
} sam3_result;

/// Control ggml log verbosity. 0 = silent (default), 1+ = ggml default logging.
/// Call before sam3_create to see Metal init messages, or after to re-silence.
void sam3_set_log_level(int level);

sam3_handle * sam3_create(const char * gguf_path, const char * bpe_path, int prefer_gpu);
void sam3_destroy(sam3_handle * handle);

int sam3_predict(sam3_handle * handle, const char * image_path, const char * prompt, sam3_result * out_result);
int sam3_predict_tokens(
    sam3_handle * handle,
    const char * image_path,
    const int32_t * tokens,
    int32_t token_count,
    sam3_result * out_result);

int sam3_predict_points(
    sam3_handle * handle,
    const char * image_path,
    const float * points_xy,
    const int32_t * labels,
    int32_t point_count,
    sam3_result * out_result);

int sam3_predict_box(
    sam3_handle * handle,
    const char * image_path,
    float x1, float y1, float x2, float y2,
    sam3_result * out_result);

void sam3_result_free(sam3_result * result);
const char * sam3_get_last_error(void);

#ifdef __cplusplus
}
#endif
