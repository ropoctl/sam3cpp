#include "sam3/gguf_model.h"

#include <fstream>
#include <regex>
#include <cstdio>
#include <stdexcept>

namespace sam3 {

namespace {

std::string gguf_get_string_opt(const gguf_context * ctx, const char * key) {
    const int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        return {};
    }
    return gguf_get_val_str(ctx, idx);
}

int32_t gguf_get_i32_opt(const gguf_context * ctx, const char * key) {
    const int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        return 0;
    }
    return gguf_get_val_i32(ctx, idx);
}

void load_tensor_aliases(
    const std::string & path,
    std::unordered_map<std::string, std::string> & aliases
) {
    aliases.clear();

    std::ifstream in(path);
    if (!in) {
        return;
    }

    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    static const std::regex pair_re(R"JSON("([^"]+)"\s*:\s*"([^"]+)")JSON");
    for (std::sregex_iterator it(text.begin(), text.end(), pair_re), end; it != end; ++it) {
        const std::string short_name = (*it)[1].str();
        const std::string original_name = (*it)[2].str();
        aliases.emplace(short_name, short_name);
        aliases.emplace(original_name, short_name);
    }
}

}  // namespace

GgufModel::~GgufModel() {
    clear();
}

void GgufModel::clear() {
    tensors_.clear();
    tensor_index_.clear();
    tensor_aliases_.clear();
    metadata_ = Metadata{};

    if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (cpu_backend_ != nullptr) {
        ggml_backend_free(cpu_backend_);
        cpu_backend_ = nullptr;
    }
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    if (weights_ctx_ != nullptr) {
        ggml_free(weights_ctx_);
        weights_ctx_ = nullptr;
    }
    if (gguf_ctx_ != nullptr) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
    }
}

bool GgufModel::load(const std::string & path, bool prefer_gpu, const std::string & tensor_map_path) {
    clear();

    ggml_context * tmp_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx =*/ &tmp_ctx,
    };
    gguf_ctx_ = gguf_init_from_file(path.c_str(), params);
    if (gguf_ctx_ == nullptr || tmp_ctx == nullptr) {
        clear();
        return false;
    }

    weights_ctx_ = ggml_init({
        /*.mem_size =*/ static_cast<size_t>(ggml_tensor_overhead() * gguf_get_n_tensors(gguf_ctx_)),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    });
    if (weights_ctx_ == nullptr) {
        clear();
        return false;
    }

    backend_ = nullptr;
    if (prefer_gpu) {
        backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (backend_ == nullptr) {
            backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        }
    }
    if (backend_ == nullptr) {
        backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (backend_ == nullptr) {
        clear();
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx_);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx_, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        ggml_tensor * dst = ggml_dup_tensor(weights_ctx_, src);
        ggml_set_name(dst, name);

        TensorInfo info;
        info.name = name;
        info.type = ggml_type_name(src->type);
        info.nbytes = ggml_nbytes(src);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (src->ne[d] <= 0) {
                break;
            }
            info.shape.push_back(src->ne[d]);
        }
        tensor_index_.emplace(info.name, tensors_.size());
        tensors_.push_back(std::move(info));
    }

    buffer_ = ggml_backend_alloc_ctx_tensors(weights_ctx_, backend_);
    if (buffer_ == nullptr) {
        clear();
        ggml_free(tmp_ctx);
        return false;
    }

    for (ggml_tensor * cur = ggml_get_first_tensor(weights_ctx_); cur != nullptr; cur = ggml_get_next_tensor(weights_ctx_, cur)) {
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
    }

    metadata_.architecture = gguf_get_string_opt(gguf_ctx_, "general.architecture");
    metadata_.source_repo = gguf_get_string_opt(gguf_ctx_, "sam3.source_repo");
    metadata_.source_impl = gguf_get_string_opt(gguf_ctx_, "sam3.source_impl");
    metadata_.image_size = gguf_get_i32_opt(gguf_ctx_, "sam3.image_size");
    metadata_.patch_size = gguf_get_i32_opt(gguf_ctx_, "sam3.patch_size");
    metadata_.vision_layers = gguf_get_i32_opt(gguf_ctx_, "sam3.vision_layers");
    metadata_.text_layers = gguf_get_i32_opt(gguf_ctx_, "sam3.text_layers");

    ggml_free(tmp_ctx);

    const std::string resolved_map_path = tensor_map_path.empty() ? path + ".tensor_map.json" : tensor_map_path;
    load_tensor_aliases(resolved_map_path, tensor_aliases_);
    return true;
}

ggml_backend_t GgufModel::cpu_backend() const {
    if (cpu_backend_ == nullptr) {
        cpu_backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    return cpu_backend_;
}

bool GgufModel::has_tensor(const std::string & name) const {
    return tensor_index_.find(name) != tensor_index_.end();
}

const TensorInfo * GgufModel::find_tensor(const std::string & name) const {
    const std::string resolved_name = resolve_tensor_name(name);
    const auto it = tensor_index_.find(resolved_name);
    if (it == tensor_index_.end()) {
        return nullptr;
    }
    return &tensors_[it->second];
}

ggml_tensor * GgufModel::find_weight(const std::string & name) const {
    if (weights_ctx_ == nullptr) {
        return nullptr;
    }
    const std::string resolved_name = resolve_tensor_name(name);
    return ggml_get_tensor(weights_ctx_, resolved_name.c_str());
}

std::string GgufModel::resolve_tensor_name(const std::string & name) const {
    const auto alias_it = tensor_aliases_.find(name);
    if (alias_it != tensor_aliases_.end()) {
        return alias_it->second;
    }
    return name;
}

}  // namespace sam3
