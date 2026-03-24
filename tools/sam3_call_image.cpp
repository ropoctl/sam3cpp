#include "sam3/gguf_model.h"
#include "sam3/npy.h"
#include "sam3/vision_neck.h"

#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: sam3-call-image <model.gguf> <trunk.npy> <output-prefix> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string trunk_path = argv[2];
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
        const sam3::NpyArrayF32 trunk = sam3::read_npy_f32(trunk_path);
        sam3::VisionNeck neck(model);
        const sam3::VisionNeckOutput out = neck.run(trunk.data, trunk.shape);

        // Match the public MLX backbone.call_image() contract, which returns
        // only the first 3 FPN levels because the upstream backbone is built
        // with scalp=1.
        constexpr int kReturnedLevels = 3;
        if (static_cast<int>(out.levels.size()) < kReturnedLevels) {
            std::cerr << "vision neck returned fewer levels than expected\n";
            return 3;
        }

        for (int i = 0; i < kReturnedLevels; ++i) {
            sam3::write_npy_f32(out_prefix + ".fpn_" + std::to_string(i) + ".npy", out.levels[static_cast<size_t>(i)], out.shapes[static_cast<size_t>(i)]);
            sam3::write_npy_f32(out_prefix + ".pos_" + std::to_string(i) + ".npy", out.positions[static_cast<size_t>(i)], out.position_shapes[static_cast<size_t>(i)]);
            std::cout << "fpn_" << i << ": "
                      << out.shapes[static_cast<size_t>(i)][0] << "x"
                      << out.shapes[static_cast<size_t>(i)][1] << "x"
                      << out.shapes[static_cast<size_t>(i)][2] << "x"
                      << out.shapes[static_cast<size_t>(i)][3] << "\n";
        }

        const size_t last = static_cast<size_t>(kReturnedLevels - 1);
        sam3::write_npy_f32(out_prefix + ".vision_features.npy", out.levels[last], out.shapes[last]);
        std::cout << "vision_features: "
                  << out.shapes[last][0] << "x"
                  << out.shapes[last][1] << "x"
                  << out.shapes[last][2] << "x"
                  << out.shapes[last][3] << "\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
