#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam3 {

class SimpleTokenizer {
public:
    explicit SimpleTokenizer(const std::string & bpe_path = {});
    ~SimpleTokenizer();

    SimpleTokenizer(const SimpleTokenizer &) = delete;
    SimpleTokenizer & operator=(const SimpleTokenizer &) = delete;
    SimpleTokenizer(SimpleTokenizer && other) noexcept;
    SimpleTokenizer & operator=(SimpleTokenizer && other) noexcept;

    std::vector<int32_t> tokenize(const std::string & text, int context_length = 32) const;

    int32_t sot_token_id() const { return sot_token_id_; }
    int32_t eot_token_id() const { return eot_token_id_; }
    int32_t vocab_size() const { return static_cast<int32_t>(encoder_size_); }

private:
    int32_t sot_token_id_ = 0;
    int32_t eot_token_id_ = 0;
    size_t encoder_size_ = 0;

    struct Impl;
    Impl * impl_ = nullptr;
};

}  // namespace sam3
