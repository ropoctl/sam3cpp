#include "sam3/pipeline.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: sam3-profile <model.gguf> <image> <prompt> [--cpu]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string prompt = argv[3];
    bool prefer_gpu = true;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu") {
            prefer_gpu = false;
        }
    }

    std::fprintf(stderr, "--- sam3 pipeline profile ---\n");
    sam3::Sam3ImagePipeline pipeline(model_path, prefer_gpu);
    auto pred = pipeline.predict(image_path, prompt);
    std::fprintf(stderr, "--- %d detections ---\n", pred.count);
    return 0;
}
