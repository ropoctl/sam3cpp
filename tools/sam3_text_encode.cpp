#include "sam3/gguf_model.h"
#include "sam3/npy.h"
#include "sam3/text_encoder.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> read_tokens(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open tokens file: " + path);
    }
    std::vector<int32_t> tokens;
    int32_t value = 0;
    while (in >> value) {
        tokens.push_back(value);
    }
    return tokens;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: sam3-text-encode <model.gguf> <tokens.txt> <output-prefix> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string tokens_path = argv[2];
    const std::string out_prefix = argv[3];
    bool prefer_gpu = true;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cpu") {
            prefer_gpu = false;
        }
    }

    sam3::GgufModel model;
    if (!model.load(model_path, prefer_gpu)) {
        std::cerr << "failed to load model: " << model_path << "\n";
        return 2;
    }

    try {
        const std::vector<int32_t> tokens = read_tokens(tokens_path);
        sam3::TextEncoder encoder(model);
        const sam3::TextEncodeOutput output = encoder.encode(tokens);

        sam3::write_i32_lines(out_prefix + ".tokens.txt", tokens);
        sam3::write_npy_f32(
            out_prefix + ".memory.npy",
            output.memory,
            {output.seq_len, 1, output.hidden_dim}
        );

        std::cout << "seq_len: " << output.seq_len << "\n";
        std::cout << "hidden_dim: " << output.hidden_dim << "\n";
        std::cout << "output: " << out_prefix << ".memory.npy\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 3;
    }

    return 0;
}
