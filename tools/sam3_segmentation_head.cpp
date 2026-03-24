#include "sam3/gguf_model.h"
#include "sam3/npy.h"
#include "sam3/segmentation_head.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 10) {
        std::cerr << "usage: sam3-segmentation-head <model.gguf> <fpn0.npy> <fpn1.npy> <fpn2.npy> <memory.npy> <prompt.npy> <prompt_mask.npy> <hs.npy> <output-prefix> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string fpn0_path = argv[2];
    const std::string fpn1_path = argv[3];
    const std::string fpn2_path = argv[4];
    const std::string memory_path = argv[5];
    const std::string prompt_path = argv[6];
    const std::string prompt_mask_path = argv[7];
    const std::string hs_path = argv[8];
    const std::string out_prefix = argv[9];

    bool prefer_gpu = true;
    for (int i = 10; i < argc; ++i) {
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
        const std::filesystem::path out_path(out_prefix);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        const sam3::NpyArrayF32 fpn0 = sam3::read_npy_f32(fpn0_path);
        const sam3::NpyArrayF32 fpn1 = sam3::read_npy_f32(fpn1_path);
        const sam3::NpyArrayF32 fpn2 = sam3::read_npy_f32(fpn2_path);
        const sam3::NpyArrayF32 memory = sam3::read_npy_f32(memory_path);
        const sam3::NpyArrayF32 prompt = sam3::read_npy_f32(prompt_path);
        const sam3::NpyArrayF32 prompt_mask = sam3::read_npy_f32(prompt_mask_path);
        const sam3::NpyArrayF32 hs = sam3::read_npy_f32(hs_path);

        sam3::SegmentationHead head(model);
        const sam3::SegmentationHeadOutput out = head.run(
            {fpn0.data, fpn1.data, fpn2.data},
            {fpn0.shape, fpn1.shape, fpn2.shape},
            memory.data,
            memory.shape,
            prompt.data,
            prompt.shape,
            prompt_mask.data,
            prompt_mask.shape,
            hs.data,
            hs.shape
        );

        sam3::write_npy_f32(out_prefix + ".pred_masks.npy", out.pred_masks, {out.batch, out.num_queries, out.height, out.width});
        sam3::write_npy_f32(out_prefix + ".semantic_seg.npy", out.semantic_seg, {out.batch, 1, out.height, out.width});

        std::cout << "queries: " << out.num_queries << "\n";
        std::cout << "height: " << out.height << "\n";
        std::cout << "width: " << out.width << "\n";
        std::cout << "output_prefix: " << out_prefix << "\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
