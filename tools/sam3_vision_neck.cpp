#include "sam3/gguf_model.h"
#include "sam3/npy.h"
#include "sam3/vision_neck.h"

#include <iostream>
#include <optional>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: sam3-vision-neck <model.gguf> <trunk.npy> <output-prefix> [--cpu] [--level N]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string trunk_path = argv[2];
    const std::string out_prefix = argv[3];
    bool prefer_gpu = true;
    std::optional<int> only_level;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cpu") {
            prefer_gpu = false;
            continue;
        }
        if (arg == "--level" && i + 1 < argc) {
            only_level = std::stoi(argv[++i]);
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
        const sam3::VisionNeckOutput out = neck.run(trunk.data, trunk.shape, only_level);
        const int level_offset = only_level.value_or(0);
        for (size_t i = 0; i < out.levels.size(); ++i) {
            const int level = level_offset + static_cast<int>(i);
            sam3::write_npy_f32(out_prefix + ".level_" + std::to_string(level) + ".npy", out.levels[i], out.shapes[i]);
            sam3::write_npy_f32(out_prefix + ".pos_" + std::to_string(level) + ".npy", out.positions[i], out.position_shapes[i]);
            std::cout << "level_" << level << ": "
                      << out.shapes[i][0] << "x" << out.shapes[i][1] << "x" << out.shapes[i][2] << "x" << out.shapes[i][3]
                      << "\n";
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
