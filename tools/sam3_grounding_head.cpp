#include "sam3/gguf_model.h"
#include "sam3/grounding_head.h"
#include "sam3/npy.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::cerr << "usage: sam3-grounding-head <model.gguf> <decoder-prefix> <prompt.npy> <prompt_mask.npy> <output-prefix> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string decoder_prefix = argv[2];
    const std::string prompt_path = argv[3];
    const std::string prompt_mask_path = argv[4];
    const std::string out_prefix = argv[5];
    bool prefer_gpu = true;
    for (int i = 6; i < argc; ++i) {
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
        std::vector<std::vector<float>> hs_layers;
        std::vector<std::vector<float>> ref_layers;
        for (int i = 0; i < 6; ++i) {
            const std::string hs_path = decoder_prefix + ".hs_" + (i < 10 ? "0" : "") + std::to_string(i) + ".npy";
            const sam3::NpyArrayF32 hs = sam3::read_npy_f32(hs_path);
            hs_layers.push_back(hs.data);
            const std::string ref_path = decoder_prefix + ".ref_" + (i < 10 ? "0" : "") + std::to_string(i) + ".npy";
            const sam3::NpyArrayF32 ref = sam3::read_npy_f32(ref_path);
            ref_layers.push_back(ref.data);
        }
        const sam3::NpyArrayF32 prompt = sam3::read_npy_f32(prompt_path);
        const sam3::NpyArrayF32 prompt_mask = sam3::read_npy_f32(prompt_mask_path);

        sam3::GroundingHead head(model);
        const sam3::GroundingHeadOutput out = head.run(hs_layers, ref_layers, prompt.data, prompt.shape, prompt_mask.data, prompt_mask.shape);

        for (int i = 0; i < out.num_layers; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), ".pred_logits_%02d.npy", i);
            sam3::write_npy_f32(out_prefix + buf, out.pred_logits[static_cast<size_t>(i)], {1, out.num_queries, 1});
            std::snprintf(buf, sizeof(buf), ".pred_boxes_%02d.npy", i);
            sam3::write_npy_f32(out_prefix + buf, out.pred_boxes[static_cast<size_t>(i)], {1, out.num_queries, 4});
        }

        std::cout << "layers: " << out.num_layers << "\n";
        std::cout << "queries: " << out.num_queries << "\n";
        std::cout << "output_prefix: " << out_prefix << "\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
