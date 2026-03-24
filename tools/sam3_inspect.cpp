#include "sam3/gguf_model.h"

#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: sam3-inspect <model.gguf>\n";
        return 1;
    }

    sam3::GgufModel model;
    if (!model.load(argv[1])) {
        std::cerr << "failed to load model: " << argv[1] << "\n";
        return 2;
    }

    const auto & meta = model.metadata();
    std::cout << "architecture: " << meta.architecture << "\n";
    std::cout << "source_repo: " << meta.source_repo << "\n";
    std::cout << "source_impl: " << meta.source_impl << "\n";
    std::cout << "image_size: " << meta.image_size << "\n";
    std::cout << "patch_size: " << meta.patch_size << "\n";
    std::cout << "vision_layers: " << meta.vision_layers << "\n";
    std::cout << "text_layers: " << meta.text_layers << "\n";
    std::cout << "tensor_count: " << model.tensors().size() << "\n";

    for (size_t i = 0; i < model.tensors().size() && i < 20; ++i) {
        const auto & tensor = model.tensors()[i];
        std::cout << tensor.name << " [";
        for (size_t j = 0; j < tensor.shape.size(); ++j) {
            if (j) {
                std::cout << ", ";
            }
            std::cout << tensor.shape[j];
        }
        std::cout << "] " << tensor.type << " " << tensor.nbytes << " bytes\n";
    }

    return 0;
}
