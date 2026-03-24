#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

inline void write_npy_f32(
    const std::string & path,
    const std::vector<float> & data,
    const std::vector<int64_t> & shape
) {
    std::ostringstream dict;
    dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            dict << ", ";
        }
        dict << shape[i];
    }
    if (shape.size() == 1) {
        dict << ",";
    }
    dict << "), }";

    std::string header = dict.str();
    while ((10 + header.size() + 1) % 16 != 0) {
        header.push_back(' ');
    }
    header.push_back('\n');

    if (header.size() > 65535) {
        throw std::runtime_error("NPY header too large");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }

    const char magic[] = "\x93NUMPY";
    out.write(magic, sizeof(magic) - 1);
    const uint8_t major = 1;
    const uint8_t minor = 0;
    out.put(static_cast<char>(major));
    out.put(static_cast<char>(minor));

    const uint16_t header_len = static_cast<uint16_t>(header.size());
    out.put(static_cast<char>(header_len & 0xff));
    out.put(static_cast<char>((header_len >> 8) & 0xff));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

inline void write_i32_lines(const std::string & path, const std::vector<int32_t> & values) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    for (size_t i = 0; i < values.size(); ++i) {
        out << values[i] << '\n';
    }
}

struct NpyArrayF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

inline NpyArrayF32 read_npy_f32(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open npy file: " + path);
    }

    char magic[6];
    in.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy magic: " + path);
    }

    char version[2];
    in.read(version, 2);
    if (version[0] != 1 || version[1] != 0) {
        throw std::runtime_error("unsupported npy version");
    }

    uint8_t h0 = 0;
    uint8_t h1 = 0;
    in.read(reinterpret_cast<char *>(&h0), 1);
    in.read(reinterpret_cast<char *>(&h1), 1);
    const uint16_t header_len = static_cast<uint16_t>(h0 | (h1 << 8));

    std::string header(header_len, '\0');
    in.read(header.data(), header_len);

    if (header.find("'descr': '<f4'") == std::string::npos) {
        throw std::runtime_error("only little-endian float32 npy files are supported");
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error("fortran-order npy arrays are not supported");
    }

    const std::regex shape_re(R"(\(\s*([0-9,\s]+)\s*\))");
    std::smatch match;
    if (!std::regex_search(header, match, shape_re)) {
        throw std::runtime_error("failed to parse npy shape");
    }

    std::vector<int64_t> shape;
    std::stringstream ss(match[1].str());
    while (ss.good()) {
        int64_t dim = 0;
        ss >> dim;
        if (ss.fail()) {
            break;
        }
        shape.push_back(dim);
        if (ss.peek() == ',') {
            ss.ignore();
        }
        while (ss.peek() == ' ') {
            ss.ignore();
        }
    }

    int64_t numel = 1;
    for (int64_t dim : shape) {
        numel *= dim;
    }

    NpyArrayF32 arr;
    arr.shape = std::move(shape);
    arr.data.resize(static_cast<size_t>(numel));
    in.read(reinterpret_cast<char *>(arr.data.data()), static_cast<std::streamsize>(arr.data.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("failed to read npy payload");
    }
    return arr;
}

}  // namespace sam3
