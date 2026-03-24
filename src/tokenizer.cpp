#include "sam3/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <climits>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <zlib.h>

#ifndef SAM3CPP_DEFAULT_BPE_PATH
#define SAM3CPP_DEFAULT_BPE_PATH "assets/bpe_simple_vocab_16e6.txt.gz"
#endif

namespace sam3 {

namespace {

std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

std::vector<std::string> split_utf8_codepoints(const std::string & text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}

std::string collapse_ascii_whitespace_lower(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    bool last_space = true;
    for (unsigned char ch : text) {
        if (std::isspace(ch)) {
            if (!last_space) {
                out.push_back(' ');
                last_space = true;
            }
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(ch)));
        last_space = false;
    }
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

std::vector<std::string> split_tokens_ascii(const std::string & text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (std::isspace(ch)) {
            ++i;
            continue;
        }

        size_t j = i + 1;
        if (std::isalpha(ch)) {
            while (j < text.size() && std::isalpha(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
        } else if (std::isdigit(ch)) {
            while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
        } else {
            while (j < text.size()) {
                const unsigned char cur = static_cast<unsigned char>(text[j]);
                if (std::isspace(cur) || std::isalpha(cur) || std::isdigit(cur)) {
                    break;
                }
                ++j;
            }
        }
        out.push_back(text.substr(i, j - i));
        i = j;
    }
    return out;
}

std::vector<std::string> load_bpe_lines(const std::string & path) {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (gz == nullptr) {
        throw std::runtime_error("failed to open BPE gzip: " + path);
    }

    std::string text;
    char buf[4096];
    int nread = 0;
    while ((nread = gzread(gz, buf, sizeof(buf))) > 0) {
        text.append(buf, buf + nread);
    }
    gzclose(gz);

    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            lines.push_back(text.substr(start));
            break;
        }
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return lines;
}

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string> & v) const noexcept {
        return std::hash<std::string>{}(v.first) ^ (std::hash<std::string>{}(v.second) << 1);
    }
};

}  // namespace

struct SimpleTokenizer::Impl {
    std::unordered_map<uint8_t, std::string> byte_encoder;
    std::unordered_map<std::string, int32_t> encoder;
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> bpe_ranks;
    mutable std::unordered_map<std::string, std::string> cache;
};

SimpleTokenizer::SimpleTokenizer(const std::string & bpe_path) : impl_(new Impl()) {
    std::string path = bpe_path.empty() ? std::string(SAM3CPP_DEFAULT_BPE_PATH) : bpe_path;

    std::vector<int> bs;
    for (int c = static_cast<int>('!'); c <= static_cast<int>('~'); ++c) {
        bs.push_back(c);
    }
    for (int c = 0xA1; c <= 0xAC; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xAE; c <= 0xFF; ++c) {
        bs.push_back(c);
    }
    std::unordered_set<int> present(bs.begin(), bs.end());
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (present.find(b) == present.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        impl_->byte_encoder.emplace(static_cast<uint8_t>(bs[i]), utf8_encode(static_cast<uint32_t>(cs[i])));
    }

    const std::vector<std::string> lines = load_bpe_lines(path);
    const size_t end = std::min<size_t>(lines.size(), 49152 - 256 - 2 + 1);
    std::vector<std::pair<std::string, std::string>> merges;
    for (size_t i = 1; i < end; ++i) {
        const std::string & line = lines[i];
        const size_t split = line.find(' ');
        if (split == std::string::npos) {
            continue;
        }
        merges.emplace_back(line.substr(0, split), line.substr(split + 1));
    }

    std::vector<std::string> vocab;
    vocab.reserve(512 + merges.size());
    for (const auto & kv : impl_->byte_encoder) {
        vocab.push_back(kv.second);
    }
    std::sort(vocab.begin(), vocab.end());
    const size_t base = vocab.size();
    for (size_t i = 0; i < base; ++i) {
        vocab.push_back(vocab[i] + "</w>");
    }
    for (const auto & merge : merges) {
        vocab.push_back(merge.first + merge.second);
    }

    const std::vector<std::string> special_tokens = {"<start_of_text>", "<end_of_text>"};
    vocab.insert(vocab.end(), special_tokens.begin(), special_tokens.end());

    for (size_t i = 0; i < vocab.size(); ++i) {
        impl_->encoder.emplace(vocab[i], static_cast<int32_t>(i));
    }
    for (size_t i = 0; i < merges.size(); ++i) {
        impl_->bpe_ranks.emplace(merges[i], static_cast<int32_t>(i));
    }
    for (const std::string & special : special_tokens) {
        impl_->cache.emplace(special, special);
    }

    sot_token_id_ = impl_->encoder.at("<start_of_text>");
    eot_token_id_ = impl_->encoder.at("<end_of_text>");
    encoder_size_ = impl_->encoder.size();
}

SimpleTokenizer::~SimpleTokenizer() {
    delete impl_;
}

SimpleTokenizer::SimpleTokenizer(SimpleTokenizer && other) noexcept
    : sot_token_id_(other.sot_token_id_),
      eot_token_id_(other.eot_token_id_),
      encoder_size_(other.encoder_size_),
      impl_(other.impl_) {
    other.sot_token_id_ = 0;
    other.eot_token_id_ = 0;
    other.encoder_size_ = 0;
    other.impl_ = nullptr;
}

SimpleTokenizer & SimpleTokenizer::operator=(SimpleTokenizer && other) noexcept {
    if (this == &other) {
        return *this;
    }

    delete impl_;
    sot_token_id_ = other.sot_token_id_;
    eot_token_id_ = other.eot_token_id_;
    encoder_size_ = other.encoder_size_;
    impl_ = other.impl_;

    other.sot_token_id_ = 0;
    other.eot_token_id_ = 0;
    other.encoder_size_ = 0;
    other.impl_ = nullptr;
    return *this;
}

std::vector<int32_t> SimpleTokenizer::tokenize(const std::string & text, int context_length) const {
    if (impl_ == nullptr) {
        throw std::runtime_error("tokenizer not initialized");
    }
    std::vector<int32_t> out(static_cast<size_t>(context_length), 0);
    std::vector<int32_t> ids;
    ids.push_back(sot_token_id_);

    const std::string cleaned = collapse_ascii_whitespace_lower(text);
    const std::vector<std::string> pieces = split_tokens_ascii(cleaned);
    for (const std::string & piece : pieces) {
        std::string encoded_bytes;
        for (unsigned char ch : piece) {
            encoded_bytes += impl_->byte_encoder.at(ch);
        }

        std::string bpe_result;
        const auto cache_it = impl_->cache.find(encoded_bytes);
        if (cache_it != impl_->cache.end()) {
            bpe_result = cache_it->second;
        } else {
            std::vector<std::string> word = split_utf8_codepoints(encoded_bytes);
            if (!word.empty()) {
                word.back() += "</w>";
            }

            auto get_pairs = [](const std::vector<std::string> & symbols) {
                std::unordered_set<std::pair<std::string, std::string>, PairHash> pairs;
                for (size_t i = 1; i < symbols.size(); ++i) {
                    pairs.emplace(symbols[i - 1], symbols[i]);
                }
                return pairs;
            };

            auto pairs = get_pairs(word);
            while (!pairs.empty()) {
                int32_t best_rank = INT32_MAX;
                std::pair<std::string, std::string> best;
                bool found = false;
                for (const auto & pair : pairs) {
                    const auto it = impl_->bpe_ranks.find(pair);
                    if (it != impl_->bpe_ranks.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best = pair;
                        found = true;
                    }
                }
                if (!found) {
                    break;
                }

                std::vector<std::string> new_word;
                for (size_t i = 0; i < word.size();) {
                    if (i + 1 < word.size() && word[i] == best.first && word[i + 1] == best.second) {
                        new_word.push_back(best.first + best.second);
                        i += 2;
                    } else {
                        new_word.push_back(word[i]);
                        ++i;
                    }
                }
                word.swap(new_word);
                if (word.size() == 1) {
                    break;
                }
                pairs = get_pairs(word);
            }

            for (size_t i = 0; i < word.size(); ++i) {
                if (i) {
                    bpe_result.push_back(' ');
                }
                bpe_result += word[i];
            }
            impl_->cache.emplace(encoded_bytes, bpe_result);
        }

        size_t start = 0;
        while (start < bpe_result.size()) {
            const size_t end = bpe_result.find(' ', start);
            const std::string token = end == std::string::npos ? bpe_result.substr(start) : bpe_result.substr(start, end - start);
            const auto enc_it = impl_->encoder.find(token);
            if (enc_it != impl_->encoder.end()) {
                ids.push_back(enc_it->second);
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
    }

    ids.push_back(eot_token_id_);
    if (static_cast<int>(ids.size()) > context_length) {
        ids.resize(static_cast<size_t>(context_length));
        ids.back() = eot_token_id_;
    }
    std::copy(ids.begin(), ids.end(), out.begin());
    return out;
}

}  // namespace sam3
