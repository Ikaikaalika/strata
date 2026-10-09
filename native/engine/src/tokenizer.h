// Hugging Face tokenizer.json support for SentencePiece-style BPE models
// (Gemma, Llama 2, Mistral): Replace/Prepend normalizers, Split/Metaspace
// pre-tokenizers, BPE with byte fallback, TemplateProcessing, and the
// Replace/ByteFallback/Fuse/Strip/Metaspace decoders. Anything else in the
// file fails at load time rather than tokenizing differently from the
// reference implementation.
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

namespace lokahi {

class Tokenizer {
 public:
  static Tokenizer load(const std::string& tokenizer_json_path);
  static Tokenizer from_json(const Json& root);

  std::vector<int32_t> encode(std::string_view text, bool add_special_tokens) const;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const;

  // Number of leading ids whose decoded text can no longer change when more
  // ids are appended, so streaming callers can emit decode(ids[0, n)) safely.
  // Byte-fallback runs at the end stay pending: the reference decodes a run
  // of byte tokens as one UTF-8 group, so a later byte can rewrite earlier
  // output (for example "Z" + 0x91 decodes to two replacement characters).
  size_t stable_prefix(const std::vector<int32_t>& ids, bool skip_special_tokens) const;

  int32_t token_to_id(const std::string& token) const;  // -1 when absent
  const std::string& id_to_token(int32_t id) const;
  bool is_special(int32_t id) const { return special_ids_.count(id) != 0; }
  size_t size() const { return id_to_token_.size(); }

 private:
  enum class SplitBehavior { Removed, Isolated, MergedWithPrevious, MergedWithNext, Contiguous };
  enum class PrependScheme { Always, First, Never };

  struct NormalizerStep {
    bool prepend = false;  // Prepend(content) when true, else Replace(pattern -> content)
    std::string pattern;
    std::string content;
  };

  struct DecoderStep {
    enum class Kind { Replace, ByteFallback, Fuse, Strip, Metaspace } kind;
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

  struct TrieNode {
    std::unordered_map<unsigned char, int32_t> next;
    int32_t added = -1;  // index into added_ when a token ends here
  };

  void load_normalizer(const Json* node);
  void load_pre_tokenizer(const Json* node);
  void load_model(const Json& model);
  void load_added_tokens(const Json* node);
  void load_post_processor(const Json* node);
  void load_decoder(const Json* node);

  std::string normalize(std::string_view text) const;
  std::vector<std::string> pre_tokenize(const std::string& normalized, bool at_start) const;
  void bpe(const std::string& word, std::vector<int32_t>& out) const;

  std::vector<NormalizerStep> normalizer_;

  enum class PreKind { None, Split, Metaspace } pre_kind_ = PreKind::None;
  std::string split_pattern_;
  SplitBehavior split_behavior_ = SplitBehavior::Removed;
  std::string metaspace_replacement_ = "\xE2\x96\x81";  // U+2581
  PrependScheme metaspace_prepend_ = PrependScheme::Always;
  bool metaspace_split_ = true;

  std::unordered_map<std::string, int32_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;  // (a, b) -> (rank, merged id)
  std::array<int32_t, 256> byte_ids_{};
  int32_t unk_id_ = -1;
  bool byte_fallback_ = false;
  bool fuse_unk_ = false;
  bool ignore_merges_ = false;

  std::vector<AddedToken> added_;
  std::vector<TrieNode> trie_;
  std::unordered_set<int32_t> special_ids_;

  std::vector<int32_t> prefix_ids_;
  std::vector<int32_t> suffix_ids_;

  std::vector<DecoderStep> decoder_;
};

}  // namespace lokahi
