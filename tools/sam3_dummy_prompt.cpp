#include "sam3/dummy_prompt.h"
#include "sam3/gguf_model.h"
#include "sam3/npy.h"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::cerr << "usage: sam3-dummy-prompt <model.gguf> <image.npy> <pos.npy> <text.npy> <text_mask.npy> <output-prefix> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string pos_path = argv[3];
    const std::string text_path = argv[4];
    const std::string text_mask_path = argv[5];
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
        const std::filesystem::path out_path(out_prefix);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        const sam3::NpyArrayF32 image = sam3::read_npy_f32(image_path);
        const sam3::NpyArrayF32 pos = sam3::read_npy_f32(pos_path);
        const sam3::NpyArrayF32 text = sam3::read_npy_f32(text_path);
        const sam3::NpyArrayF32 text_mask = sam3::read_npy_f32(text_mask_path);

        sam3::DummyPromptEncoder encoder(model);
        const sam3::DummyPromptOutput out = encoder.run(
            image.data, image.shape,
            pos.data, pos.shape,
            text.data, text.shape,
            text_mask.data, text_mask.shape
        );

        sam3::write_npy_f32(out_prefix + ".geo_token.npy", out.geo_token, {1, 1, out.hidden_dim});
        sam3::write_npy_f32(out_prefix + ".prompt.npy", out.prompt, {out.prompt_seq_len, 1, out.hidden_dim});
        sam3::write_npy_f32(out_prefix + ".prompt_mask.npy", out.prompt_mask, {1, out.prompt_seq_len});

        std::cout << "text_seq_len: " << out.text_seq_len << "\n";
        std::cout << "prompt_seq_len: " << out.prompt_seq_len << "\n";
        std::cout << "hidden_dim: " << out.hidden_dim << "\n";
        std::cout << "output_prefix: " << out_prefix << "\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
