#include "sam3/decoder.h"
#include "sam3/gguf_model.h"
#include "sam3/npy.h"

#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::cerr << "usage: sam3-decoder <model.gguf> <memory.npy> <pos_embed.npy> <prompt.npy> <prompt_mask.npy> <output-prefix> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string memory_path = argv[2];
    const std::string pos_path = argv[3];
    const std::string prompt_path = argv[4];
    const std::string prompt_mask_path = argv[5];
    const std::string out_prefix = argv[6];
    bool prefer_gpu = true;
    for (int i = 7; i < argc; ++i) {
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
        const sam3::NpyArrayF32 memory = sam3::read_npy_f32(memory_path);
        const sam3::NpyArrayF32 pos = sam3::read_npy_f32(pos_path);
        const sam3::NpyArrayF32 prompt = sam3::read_npy_f32(prompt_path);
        const sam3::NpyArrayF32 prompt_mask = sam3::read_npy_f32(prompt_mask_path);

        sam3::Decoder decoder(model);
        const sam3::DecoderOutput out = decoder.run(
            memory.data, memory.shape,
            pos.data, pos.shape,
            prompt.data, prompt.shape,
            prompt_mask.data, prompt_mask.shape
        );

        for (int i = 0; i < out.num_layers; ++i) {
            char layer_buf[64];
            std::snprintf(layer_buf, sizeof(layer_buf), ".hs_%02d.npy", i);
            sam3::write_npy_f32(out_prefix + layer_buf, out.hs[static_cast<size_t>(i)], {out.num_queries, 1, out.hidden_dim});
            std::snprintf(layer_buf, sizeof(layer_buf), ".ref_%02d.npy", i);
            sam3::write_npy_f32(out_prefix + layer_buf, out.reference_boxes[static_cast<size_t>(i)], {out.num_queries, 1, 4});
            std::snprintf(layer_buf, sizeof(layer_buf), ".presence_%02d.npy", i);
            sam3::write_npy_f32(out_prefix + layer_buf, out.presence_logits[static_cast<size_t>(i)], {1, 1, 1});
        }

        std::cout << "layers: " << out.num_layers << "\n";
        std::cout << "queries: " << out.num_queries << "\n";
        std::cout << "hidden_dim: " << out.hidden_dim << "\n";
        std::cout << "output_prefix: " << out_prefix << "\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
