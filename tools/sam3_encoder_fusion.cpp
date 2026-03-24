#include "sam3/encoder_fusion.h"
#include "sam3/gguf_model.h"
#include "sam3/npy.h"

#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::cerr << "usage: sam3-encoder-fusion <model.gguf> <image.npy> <pos.npy> <prompt.npy> <prompt-mask.npy> <output-prefix> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
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
        const sam3::NpyArrayF32 image = sam3::read_npy_f32(image_path);
        const sam3::NpyArrayF32 pos = sam3::read_npy_f32(pos_path);
        const sam3::NpyArrayF32 prompt = sam3::read_npy_f32(prompt_path);
        const sam3::NpyArrayF32 prompt_mask = sam3::read_npy_f32(prompt_mask_path);

        sam3::EncoderFusion encoder(model);
        const sam3::EncoderFusionOutput out = encoder.run(
            image.data,
            image.shape,
            pos.data,
            pos.shape,
            prompt.data,
            prompt.shape,
            prompt_mask.data,
            prompt_mask.shape
        );

        sam3::write_npy_f32(out_prefix + ".memory.npy", out.memory, {out.image_seq_len, 1, out.hidden_dim});
        sam3::write_npy_f32(out_prefix + ".pos_embed.npy", out.pos_embed, {out.image_seq_len, 1, out.hidden_dim});

        std::cout << "image_seq_len: " << out.image_seq_len << "\n";
        std::cout << "prompt_seq_len: " << out.prompt_seq_len << "\n";
        std::cout << "hidden_dim: " << out.hidden_dim << "\n";
        std::cout << "memory: " << out_prefix << ".memory.npy\n";
        std::cout << "pos_embed: " << out_prefix << ".pos_embed.npy\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
