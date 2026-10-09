// Hugging Face tokenizer.json support for BPE models, covering both families
// of popular checkpoints:
//   * SentencePiece-style (Gemma, Llama 2, Mistral): Replace/Prepend
//     normalizers, Split/Metaspace pre-tokenizers, byte fallback, and the
//     Replace/ByteFallback/Fuse/Strip/Metaspace decoders;
//   * byte-level (Llama 3, Qwen, GPT-OSS, DeepSeek, GLM): NFC, regex Split
//     (an Oniguruma-compatible subset), the ByteLevel pre-tokenizer and
//     decoder, and added tokens matched after normalization.
// Anything else in the file fails at load time rather than tokenizing
// differently from the reference implementation.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "json.h"
#include "regex.h"

namespace lokahi {

class Tokenizer {
 public:
  static Tokenizer load(const std::string& tokenizer_json_path);
  static Tokenizer from_json(const Json& root);

  std::vector<int32_t> encode(std::string_view text, bool add_special_tokens) const;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const;

  // Number of leading ids whose decoded text can no longer change when more
  // ids are appended, so streaming callers can emit decode(ids[0, n)) safely.
  // Ids that may still join a later byte stay pending: byte-fallback runs
  // decode as one UTF-8 group (so "Z" + 0x91 decodes to two replacement
  // characters), and byte-level tokens can end inside a UTF-8 sequence.
  size_t stable_prefix(const std::vector<int32_t>& ids, bool skip_special_tokens) const;

  // The normalizer and pre-tokenizer applied to `text` without added-token
  // matching: the words BPE runs on, as the reference's
  // pre_tokenizer.pre_tokenize_str(normalizer.normalize_str(text)) reports.
  std::vector<std::string> words(std::string_view text) const;

  int32_t token_to_id(const std::string& token) const;  // -1 when absent
  const std::string& id_to_token(int32_t id) const;
  bool is_special(int32_t id) const { return special_ids_.count(id) != 0; }
  size_t size() const { return id_to_token_.size(); }

  // How Split keeps delimiters (tokenizers' SplitDelimiterBehavior).
  enum class SplitBehavior { Removed, Isolated, MergedWithPrevious, MergedWithNext, Contiguous };

 private:
  enum class PrependScheme { Always, First, Never };

  struct NormalizerStep {
    enum class Kind { Replace, Prepend, NFC } kind;
    std::string pattern;
    std::string content;
  };

  struct PreTokenizerStep {
    enum class Kind { Split, Metaspace, ByteLevel } kind;
    std::string literal;                 // Split on a String pattern
    std::shared_ptr<const Regex> regex;  // Split on a Regex pattern
    SplitBehavior behavior = SplitBehavior::Removed;
    std::string replacement;             // Metaspace
    PrependScheme prepend = PrependScheme::Always;
    bool split = true;
    bool add_prefix_space = false;       // ByteLevel
    bool use_regex = true;
  };

  // A pre-tokenized piece; at_start marks a piece at offset 0 of the input,
  // which Metaspace's "first" prepend scheme depends on.
  struct Word {
    std::string text;
    bool at_start;
  };

  struct DecoderStep {
    enum class Kind { Replace, ByteFallback, Fuse, Strip, Metaspace, ByteLevel } kind;
    std::string pattern;
    std::string content;
    int start = 0;
    int stop = 0;
    bool metaspace_prepend = true;
  };

  struct AddedToken {
    std::string content;
    int32_t id;
    bool special;
  };

  // Byte trie for leftmost-longest added-token matching.
  struct Trie {
    struct Node {
      std::unordered_map<unsigned char, int32_t> next;
      int32_t value = -1;
    };
    std::vector<Node> nodes = std::vector<Node>(1);
    bool empty() const { return nodes.size() == 1; }
    void insert(std::string_view key, int32_t value);
    // Value of the longest key starting at text[begin], or -1.
    int32_t longest_match(std::string_view text, size_t begin, size_t* end) const;
  };

  void load_normalizer(const Json* node);
  void load_pre_tokenizer(const Json* node);
  void load_model(const Json& model);
  void load_added_tokens(const Json* node);
  void load_post_processor(const Json* node);
  void load_decoder(const Json* node);

  std::string normalize(std::string_view text) const;
  std::vector<Word> pre_tokenize(std::vector<Word> words) const;
  void bpe(const std::string& word, std::vector<int32_t>& out) const;

  std::vector<NormalizerStep> normalizer_;
  std::vector<PreTokenizerStep> pre_tokenizer_;

  std::unordered_map<std::string, int32_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;  // (a, b) -> (rank, merged id)
  std::array<int32_t, 256> byte_ids_{};
  int32_t unk_id_ = -1;
  bool byte_fallback_ = false;
  bool fuse_unk_ = false;
  bool ignore_merges_ = false;

  std::vector<AddedToken> added_;
  Trie raw_trie_;         // added tokens matched on the input text
  Trie normalized_trie_;  // added tokens matched after normalization
  std::unordered_set<int32_t> special_ids_;

  std::vector<int32_t> prefix_ids_;
  std::vector<int32_t> suffix_ids_;

  std::vector<DecoderStep> decoder_;
};

}  // namespace lokahi
