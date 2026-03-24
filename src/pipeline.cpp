#include "sam3/pipeline.h"
#include "sam3/geometry_encoder.h"

#include "ggml.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

static void null_log_callback(enum ggml_log_level, const char *, void *) {}

void set_log_level(int level) {
    if (level <= 0) {
        ggml_log_set(null_log_callback, nullptr);
    } else {
        ggml_log_set(nullptr, nullptr);  // restore default
    }
}

namespace {

constexpr int kImageSize = 1008;
constexpr float kDetectionThreshold = 0.5f;

struct LoadedImage {
    int width = 0;
    int height = 0;
    std::vector<float> preprocessed_nchw;
};

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

LoadedImage load_image_rgb(const std::string & path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> image(
        stbi_load(path.c_str(), &width, &height, &channels, 3),
        stbi_image_free);
    if (!image) {
        throw std::runtime_error("failed to load image: " + path);
    }

    std::vector<float> out(static_cast<size_t>(3 * kImageSize * kImageSize));
    const float x_scale = static_cast<float>(width) / static_cast<float>(kImageSize);
    const float y_scale = static_cast<float>(height) / static_cast<float>(kImageSize);
    for (int dy = 0; dy < kImageSize; ++dy) {
        const float sy = (static_cast<float>(dy) + 0.5f) * y_scale - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, height - 1);
        const int y1 = std::clamp(y0 + 1, 0, height - 1);
        const float wy = sy - static_cast<float>(y0);
        for (int dx = 0; dx < kImageSize; ++dx) {
            const float sx = (static_cast<float>(dx) + 0.5f) * x_scale - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, width - 1);
            const int x1 = std::clamp(x0 + 1, 0, width - 1);
            const float wx = sx - static_cast<float>(x0);
            for (int c = 0; c < 3; ++c) {
                const float p00 = static_cast<float>(image.get()[(y0 * width + x0) * 3 + c]) / 255.0f;
                const float p01 = static_cast<float>(image.get()[(y0 * width + x1) * 3 + c]) / 255.0f;
                const float p10 = static_cast<float>(image.get()[(y1 * width + x0) * 3 + c]) / 255.0f;
                const float p11 = static_cast<float>(image.get()[(y1 * width + x1) * 3 + c]) / 255.0f;
                const float top = p00 + (p01 - p00) * wx;
                const float bottom = p10 + (p11 - p10) * wx;
                const float value = top + (bottom - top) * wy;
                out[static_cast<size_t>((c * kImageSize + dy) * kImageSize + dx)] = (value - 0.5f) / 0.5f;
            }
        }
    }

    return LoadedImage{width, height, std::move(out)};
}

std::vector<float> make_text_mask(const std::vector<int32_t> & tokens) {
    std::vector<float> mask(tokens.size(), 0.0f);
    for (size_t i = 0; i < tokens.size(); ++i) {
        mask[i] = tokens[i] == 0 ? 1.0f : 0.0f;
    }
    return mask;
}

std::vector<float> resize_logits_bilinear(
    const float * src,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w
) {
    std::vector<float> out(static_cast<size_t>(dst_h * dst_w));
    const float x_scale = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float y_scale = static_cast<float>(src_h) / static_cast<float>(dst_h);
    for (int dy = 0; dy < dst_h; ++dy) {
        const float sy = (static_cast<float>(dy) + 0.5f) * y_scale - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, src_h - 1);
        const int y1 = std::clamp(y0 + 1, 0, src_h - 1);
        const float wy = sy - static_cast<float>(y0);
        for (int dx = 0; dx < dst_w; ++dx) {
            const float sx = (static_cast<float>(dx) + 0.5f) * x_scale - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, src_w - 1);
            const int x1 = std::clamp(x0 + 1, 0, src_w - 1);
            const float wx = sx - static_cast<float>(x0);
            const float p00 = src[y0 * src_w + x0];
            const float p01 = src[y0 * src_w + x1];
            const float p10 = src[y1 * src_w + x0];
            const float p11 = src[y1 * src_w + x1];
            const float top = p00 + (p01 - p00) * wx;
            const float bottom = p10 + (p11 - p10) * wx;
            out[static_cast<size_t>(dy * dst_w + dx)] = top + (bottom - top) * wy;
        }
    }
    return out;
}

void cxcywh_to_xyxy_inplace(std::vector<float> & boxes, float width, float height) {
    for (size_t i = 0; i + 3 < boxes.size(); i += 4) {
        const float cx = boxes[i + 0];
        const float cy = boxes[i + 1];
        const float bw = boxes[i + 2];
        const float bh = boxes[i + 3];
        boxes[i + 0] = (cx - 0.5f * bw) * width;
        boxes[i + 1] = (cy - 0.5f * bh) * height;
        boxes[i + 2] = (cx + 0.5f * bw) * width;
        boxes[i + 3] = (cy + 0.5f * bh) * height;
    }
}

}  // namespace

Sam3ImagePipeline::Sam3ImagePipeline(const std::string & gguf_path, bool prefer_gpu, const std::string & bpe_path)
    : tokenizer_(bpe_path) {
    set_log_level(0);  // silence ggml logs by default
    if (!model_.load(gguf_path, prefer_gpu)) {
        throw std::runtime_error("failed to load model: " + gguf_path);
    }
}

Sam3Prediction Sam3ImagePipeline::predict(const std::string & image_path, const std::string & prompt) const {
    return predict_tokens(image_path, tokenizer_.tokenize(prompt, 32));
}

Sam3Prediction Sam3ImagePipeline::predict_tokens(const std::string & image_path, const std::vector<int32_t> & tokens) const {
    if (tokens.empty()) {
        throw std::runtime_error("no tokens provided");
    }

    const LoadedImage image = load_image_rgb(image_path);

    VisionTrunk trunk(model_);
    const VisionTrunkOutput trunk_out = trunk.run(image.preprocessed_nchw, {1, 3, kImageSize, kImageSize});

    VisionNeck neck(model_);
    const VisionNeckOutput neck_out = neck.run(trunk_out.trunk_nchw, trunk_out.shape_nchw);
    if (neck_out.levels.size() < 3 || neck_out.positions.size() < 3) {
        throw std::runtime_error("vision neck returned insufficient levels");
    }

    TextEncoder text(model_);
    const TextEncodeOutput text_out = text.encode(tokens);
    const std::vector<float> text_mask = make_text_mask(tokens);

    DummyPromptEncoder prompt_encoder(model_);
    const DummyPromptOutput prompt_out = prompt_encoder.run(
        neck_out.levels[2], neck_out.shapes[2],
        neck_out.positions[2], neck_out.position_shapes[2],
        text_out.memory, {text_out.seq_len, 1, text_out.hidden_dim},
        text_mask, {1, static_cast<int64_t>(text_mask.size())});

    EncoderFusion encoder(model_);
    const EncoderFusionOutput encoder_out = encoder.run(
        neck_out.levels[2], neck_out.shapes[2],
        neck_out.positions[2], neck_out.position_shapes[2],
        prompt_out.prompt, {prompt_out.prompt_seq_len, 1, prompt_out.hidden_dim},
        prompt_out.prompt_mask, {1, prompt_out.prompt_seq_len});

    Decoder decoder(model_);
    const DecoderOutput decoder_out = decoder.run(
        encoder_out.memory, {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        encoder_out.pos_embed, {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        prompt_out.prompt, {prompt_out.prompt_seq_len, 1, prompt_out.hidden_dim},
        prompt_out.prompt_mask, {1, prompt_out.prompt_seq_len});

    GroundingHead grounding(model_);
    const GroundingHeadOutput grounding_out = grounding.run(
        decoder_out.hs,
        decoder_out.reference_boxes,
        prompt_out.prompt, {prompt_out.prompt_seq_len, 1, prompt_out.hidden_dim},
        prompt_out.prompt_mask, {1, prompt_out.prompt_seq_len});

    SegmentationHead segmentation(model_);
    const SegmentationHeadOutput seg_out = segmentation.run(
        {neck_out.levels[0], neck_out.levels[1], neck_out.levels[2]},
        {neck_out.shapes[0], neck_out.shapes[1], neck_out.shapes[2]},
        encoder_out.memory,
        {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        prompt_out.prompt,
        {prompt_out.prompt_seq_len, 1, prompt_out.hidden_dim},
        prompt_out.prompt_mask,
        {1, prompt_out.prompt_seq_len},
        decoder_out.hs.back(),
        {decoder_out.num_queries, 1, decoder_out.hidden_dim});

    const std::vector<float> & logits = grounding_out.pred_logits.back();
    std::vector<float> boxes = grounding_out.pred_boxes.back();
    const float presence = sigmoid(decoder_out.presence_logits.back().empty() ? 0.0f : decoder_out.presence_logits.back()[0]);

    std::vector<int32_t> keep;
    std::vector<float> scores;
    keep.reserve(logits.size());
    scores.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        const float score = sigmoid(logits[i]) * presence;
        if (score > kDetectionThreshold) {
            keep.push_back(static_cast<int32_t>(i));
            scores.push_back(score);
        }
    }

    Sam3Prediction pred;
    pred.width = image.width;
    pred.height = image.height;
    pred.count = static_cast<int32_t>(keep.size());
    pred.scores = std::move(scores);
    pred.boxes_xyxy.resize(static_cast<size_t>(pred.count) * 4);
    pred.masks.resize(static_cast<size_t>(pred.count) * pred.height * pred.width);

    for (int32_t out_i = 0; out_i < pred.count; ++out_i) {
        const int32_t q = keep[static_cast<size_t>(out_i)];
        std::memcpy(
            pred.boxes_xyxy.data() + static_cast<size_t>(out_i) * 4,
            boxes.data() + static_cast<size_t>(q) * 4,
            4 * sizeof(float));
    }
    cxcywh_to_xyxy_inplace(pred.boxes_xyxy, static_cast<float>(pred.width), static_cast<float>(pred.height));

    const int src_h = static_cast<int>(seg_out.height);
    const int src_w = static_cast<int>(seg_out.width);
    const size_t mask_stride = static_cast<size_t>(src_h * src_w);
    for (int32_t out_i = 0; out_i < pred.count; ++out_i) {
        const int32_t q = keep[static_cast<size_t>(out_i)];
        const float * src_mask = seg_out.pred_masks.data() + static_cast<size_t>(q) * mask_stride;
        std::vector<float> resized = resize_logits_bilinear(src_mask, src_h, src_w, pred.height, pred.width);
        float * dst = pred.masks.data() + static_cast<size_t>(out_i) * pred.height * pred.width;
        for (size_t i = 0; i < resized.size(); ++i) {
            dst[i] = sigmoid(resized[i]);
        }
    }

    return pred;
}

Sam3Prediction Sam3ImagePipeline::predict_points(
    const std::string & image_path,
    const float * points_xy,
    const int32_t * labels,
    int32_t count
) const {
    if (count <= 0 || points_xy == nullptr || labels == nullptr) {
        throw std::runtime_error("invalid point prompt arguments");
    }

    const LoadedImage image = load_image_rgb(image_path);

    VisionTrunk trunk(model_);
    const VisionTrunkOutput trunk_out = trunk.run(image.preprocessed_nchw, {1, 3, kImageSize, kImageSize});

    VisionNeck neck(model_);
    const VisionNeckOutput neck_out = neck.run(trunk_out.trunk_nchw, trunk_out.shape_nchw);
    if (neck_out.levels.size() < 3 || neck_out.positions.size() < 3) {
        throw std::runtime_error("vision neck returned insufficient levels");
    }

    GeometryEncoder geo(model_);
    const GeometryEncoderOutput geo_out = geo.run(
        points_xy, labels, count,
        nullptr, nullptr, 0,
        neck_out.levels[2], neck_out.shapes[2],
        neck_out.positions[2], neck_out.position_shapes[2]);

    EncoderFusion encoder(model_);
    const EncoderFusionOutput encoder_out = encoder.run(
        neck_out.levels[2], neck_out.shapes[2],
        neck_out.positions[2], neck_out.position_shapes[2],
        geo_out.geo_tokens, {geo_out.geo_seq_len, 1, geo_out.hidden_dim},
        geo_out.geo_mask, {1, geo_out.geo_seq_len});

    Decoder decoder(model_);
    const DecoderOutput decoder_out = decoder.run(
        encoder_out.memory, {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        encoder_out.pos_embed, {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        geo_out.geo_tokens, {geo_out.geo_seq_len, 1, geo_out.hidden_dim},
        geo_out.geo_mask, {1, geo_out.geo_seq_len});

    GroundingHead grounding(model_);
    const GroundingHeadOutput grounding_out = grounding.run(
        decoder_out.hs,
        decoder_out.reference_boxes,
        geo_out.geo_tokens, {geo_out.geo_seq_len, 1, geo_out.hidden_dim},
        geo_out.geo_mask, {1, geo_out.geo_seq_len});

    SegmentationHead segmentation(model_);
    const SegmentationHeadOutput seg_out = segmentation.run(
        {neck_out.levels[0], neck_out.levels[1], neck_out.levels[2]},
        {neck_out.shapes[0], neck_out.shapes[1], neck_out.shapes[2]},
        encoder_out.memory,
        {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        geo_out.geo_tokens,
        {geo_out.geo_seq_len, 1, geo_out.hidden_dim},
        geo_out.geo_mask,
        {1, geo_out.geo_seq_len},
        decoder_out.hs.back(),
        {decoder_out.num_queries, 1, decoder_out.hidden_dim});

    // Lower threshold for geometry prompts (direct_project only produces moderate scores)
    constexpr float kGeoThreshold = 0.01f;

    const std::vector<float> & logits = grounding_out.pred_logits.back();
    std::vector<float> boxes = grounding_out.pred_boxes.back();
    const float presence = sigmoid(decoder_out.presence_logits.back().empty() ? 0.0f : decoder_out.presence_logits.back()[0]);

    std::vector<int32_t> keep;
    std::vector<float> scores;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float score = sigmoid(logits[i]) * presence;
        if (score > kGeoThreshold) {
            keep.push_back(static_cast<int32_t>(i));
            scores.push_back(score);
        }
    }

    Sam3Prediction pred;
    pred.width = image.width;
    pred.height = image.height;
    pred.count = static_cast<int32_t>(keep.size());
    pred.scores = std::move(scores);
    pred.boxes_xyxy.resize(static_cast<size_t>(pred.count) * 4);
    pred.masks.resize(static_cast<size_t>(pred.count) * pred.height * pred.width);

    for (int32_t out_i = 0; out_i < pred.count; ++out_i) {
        const int32_t q = keep[static_cast<size_t>(out_i)];
        std::memcpy(
            pred.boxes_xyxy.data() + static_cast<size_t>(out_i) * 4,
            boxes.data() + static_cast<size_t>(q) * 4,
            4 * sizeof(float));
    }
    cxcywh_to_xyxy_inplace(pred.boxes_xyxy, static_cast<float>(pred.width), static_cast<float>(pred.height));

    const int src_h = static_cast<int>(seg_out.height);
    const int src_w = static_cast<int>(seg_out.width);
    const size_t mask_stride = static_cast<size_t>(src_h * src_w);
    for (int32_t out_i = 0; out_i < pred.count; ++out_i) {
        const int32_t q = keep[static_cast<size_t>(out_i)];
        const float * src_mask = seg_out.pred_masks.data() + static_cast<size_t>(q) * mask_stride;
        std::vector<float> resized = resize_logits_bilinear(src_mask, src_h, src_w, pred.height, pred.width);
        float * dst = pred.masks.data() + static_cast<size_t>(out_i) * pred.height * pred.width;
        for (size_t i = 0; i < resized.size(); ++i) {
            dst[i] = sigmoid(resized[i]);
        }
    }

    return pred;
}

Sam3Prediction Sam3ImagePipeline::predict_box(
    const std::string & image_path,
    float x1, float y1, float x2, float y2
) const {
    // Load image to get dimensions for normalization
    int w = 0, h = 0, ch = 0;
    if (!stbi_info(image_path.c_str(), &w, &h, &ch)) {
        throw std::runtime_error("failed to read image info: " + image_path);
    }

    // Convert pixel xyxy to normalized cxcywh
    const float cx = ((x1 + x2) / 2.0f) / static_cast<float>(w);
    const float cy = ((y1 + y2) / 2.0f) / static_cast<float>(h);
    const float bw = (x2 - x1) / static_cast<float>(w);
    const float bh = (y2 - y1) / static_cast<float>(h);
    const float box[4] = {cx, cy, bw, bh};
    const int32_t label = 1;

    // Use predict_points infrastructure but with box
    const LoadedImage image = load_image_rgb(image_path);

    VisionTrunk trunk(model_);
    const VisionTrunkOutput trunk_out = trunk.run(image.preprocessed_nchw, {1, 3, kImageSize, kImageSize});

    VisionNeck neck(model_);
    const VisionNeckOutput neck_out = neck.run(trunk_out.trunk_nchw, trunk_out.shape_nchw);
    if (neck_out.levels.size() < 3 || neck_out.positions.size() < 3) {
        throw std::runtime_error("vision neck returned insufficient levels");
    }

    GeometryEncoder geo(model_);
    const GeometryEncoderOutput geo_out = geo.run(
        nullptr, nullptr, 0,
        box, &label, 1,
        neck_out.levels[2], neck_out.shapes[2],
        neck_out.positions[2], neck_out.position_shapes[2]);

    EncoderFusion encoder(model_);
    const EncoderFusionOutput encoder_out = encoder.run(
        neck_out.levels[2], neck_out.shapes[2],
        neck_out.positions[2], neck_out.position_shapes[2],
        geo_out.geo_tokens, {geo_out.geo_seq_len, 1, geo_out.hidden_dim},
        geo_out.geo_mask, {1, geo_out.geo_seq_len});

    Decoder decoder(model_);
    const DecoderOutput decoder_out = decoder.run(
        encoder_out.memory, {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        encoder_out.pos_embed, {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        geo_out.geo_tokens, {geo_out.geo_seq_len, 1, geo_out.hidden_dim},
        geo_out.geo_mask, {1, geo_out.geo_seq_len});

    GroundingHead grounding(model_);
    const GroundingHeadOutput grounding_out = grounding.run(
        decoder_out.hs,
        decoder_out.reference_boxes,
        geo_out.geo_tokens, {geo_out.geo_seq_len, 1, geo_out.hidden_dim},
        geo_out.geo_mask, {1, geo_out.geo_seq_len});

    SegmentationHead segmentation(model_);
    const SegmentationHeadOutput seg_out = segmentation.run(
        {neck_out.levels[0], neck_out.levels[1], neck_out.levels[2]},
        {neck_out.shapes[0], neck_out.shapes[1], neck_out.shapes[2]},
        encoder_out.memory,
        {encoder_out.image_seq_len, 1, encoder_out.hidden_dim},
        geo_out.geo_tokens,
        {geo_out.geo_seq_len, 1, geo_out.hidden_dim},
        geo_out.geo_mask,
        {1, geo_out.geo_seq_len},
        decoder_out.hs.back(),
        {decoder_out.num_queries, 1, decoder_out.hidden_dim});

    // Return all queries (box prompt should produce exactly the box region)
    const std::vector<float> & logits = grounding_out.pred_logits.back();
    std::vector<float> pred_boxes = grounding_out.pred_boxes.back();
    const float presence = sigmoid(decoder_out.presence_logits.back().empty() ? 0.0f : decoder_out.presence_logits.back()[0]);

    std::vector<int32_t> keep;
    std::vector<float> scores;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float score = sigmoid(logits[i]) * presence;
        if (score > 0.3f) {
            keep.push_back(static_cast<int32_t>(i));
            scores.push_back(score);
        }
    }

    Sam3Prediction pred;
    pred.width = image.width;
    pred.height = image.height;
    pred.count = static_cast<int32_t>(keep.size());
    pred.scores = std::move(scores);
    pred.boxes_xyxy.resize(static_cast<size_t>(pred.count) * 4);
    pred.masks.resize(static_cast<size_t>(pred.count) * pred.height * pred.width);

    for (int32_t out_i = 0; out_i < pred.count; ++out_i) {
        const int32_t q = keep[static_cast<size_t>(out_i)];
        std::memcpy(
            pred.boxes_xyxy.data() + static_cast<size_t>(out_i) * 4,
            pred_boxes.data() + static_cast<size_t>(q) * 4,
            4 * sizeof(float));
    }
    cxcywh_to_xyxy_inplace(pred.boxes_xyxy, static_cast<float>(pred.width), static_cast<float>(pred.height));

    const int src_h_m = static_cast<int>(seg_out.height);
    const int src_w_m = static_cast<int>(seg_out.width);
    const size_t mask_stride = static_cast<size_t>(src_h_m * src_w_m);
    for (int32_t out_i = 0; out_i < pred.count; ++out_i) {
        const int32_t q = keep[static_cast<size_t>(out_i)];
        const float * src_mask = seg_out.pred_masks.data() + static_cast<size_t>(q) * mask_stride;
        std::vector<float> resized = resize_logits_bilinear(src_mask, src_h_m, src_w_m, pred.height, pred.width);
        float * dst = pred.masks.data() + static_cast<size_t>(out_i) * pred.height * pred.width;
        for (size_t i = 0; i < resized.size(); ++i) {
            dst[i] = sigmoid(resized[i]);
        }
    }

    return pred;
}

}  // namespace sam3
