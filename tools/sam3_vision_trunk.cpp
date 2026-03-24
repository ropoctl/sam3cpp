#include "sam3/gguf_model.h"
#include "sam3/npy.h"
#include "sam3/vision_trunk.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kImageSize = 1008;

bool has_npy_suffix(const std::string & path) {
    if (path.size() < 4) {
        return false;
    }
    std::string suffix = path.substr(path.size() - 4);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return suffix == ".npy";
}

std::vector<float> bilinear_resize_rgb(
    const uint8_t * src,
    int src_w,
    int src_h
) {
    std::vector<float> out(static_cast<size_t>(3 * kImageSize * kImageSize));
    const float x_scale = static_cast<float>(src_w) / static_cast<float>(kImageSize);
    const float y_scale = static_cast<float>(src_h) / static_cast<float>(kImageSize);

    for (int dy = 0; dy < kImageSize; ++dy) {
        const float sy = (static_cast<float>(dy) + 0.5f) * y_scale - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, src_h - 1);
        const int y1 = std::clamp(y0 + 1, 0, src_h - 1);
        const float wy = sy - static_cast<float>(y0);
        for (int dx = 0; dx < kImageSize; ++dx) {
            const float sx = (static_cast<float>(dx) + 0.5f) * x_scale - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, src_w - 1);
            const int x1 = std::clamp(x0 + 1, 0, src_w - 1);
            const float wx = sx - static_cast<float>(x0);

            for (int c = 0; c < 3; ++c) {
                const float p00 = static_cast<float>(src[(y0 * src_w + x0) * 3 + c]) / 255.0f;
                const float p01 = static_cast<float>(src[(y0 * src_w + x1) * 3 + c]) / 255.0f;
                const float p10 = static_cast<float>(src[(y1 * src_w + x0) * 3 + c]) / 255.0f;
                const float p11 = static_cast<float>(src[(y1 * src_w + x1) * 3 + c]) / 255.0f;
                const float top = p00 + (p01 - p00) * wx;
                const float bot = p10 + (p11 - p10) * wx;
                const float value = top + (bot - top) * wy;
                out[static_cast<size_t>((c * kImageSize + dy) * kImageSize + dx)] = (value - 0.5f) / 0.5f;
            }
        }
    }

    return out;
}

std::vector<float> load_image_or_tensor(const std::string & path) {
    if (has_npy_suffix(path)) {
        const sam3::NpyArrayF32 image = sam3::read_npy_f32(path);
        if (image.shape != std::vector<int64_t>({1, 3, kImageSize, kImageSize})) {
            throw std::runtime_error("preprocessed input tensor must have shape [1, 3, 1008, 1008]");
        }
        return image.data;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> image(
        stbi_load(path.c_str(), &width, &height, &channels, 3),
        stbi_image_free);
    if (!image) {
        throw std::runtime_error("failed to load image: " + path);
    }

    return bilinear_resize_rgb(image.get(), width, height);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: sam3-vision-trunk <model.gguf> <image-or-image.npy> <output-prefix> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string out_prefix = argv[3];
    bool prefer_gpu = true;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cpu") {
            prefer_gpu = false;
            continue;
        }
        std::cerr << "unknown argument: " << arg << "\n";
        return 1;
    }

    sam3::GgufModel model;
    if (!model.load(model_path, prefer_gpu)) {
        std::cerr << "failed to load model\n";
        return 2;
    }

    try {
        const std::vector<float> image_nchw = load_image_or_tensor(image_path);
        sam3::VisionTrunk trunk(model);
        const sam3::VisionTrunkOutput out = trunk.run(image_nchw, {1, 3, kImageSize, kImageSize});
        sam3::write_npy_f32(out_prefix + ".trunk.npy", out.trunk_nchw, out.shape_nchw);
        std::cout << "trunk: "
                  << out.shape_nchw[0] << "x"
                  << out.shape_nchw[1] << "x"
                  << out.shape_nchw[2] << "x"
                  << out.shape_nchw[3] << "\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
