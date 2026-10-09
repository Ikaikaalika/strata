#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <queue>

#include "common.h"
#include "safetensors.h"
#include "unicode.h"

namespace lokahi {
namespace {

// Byte length of the UTF-8 sequence starting at text[i]; throws on malformed input.
size_t utf8_char_len(std::string_view text, size_t i) {
  const size_t len = unicode::utf8_sequence_length(text, i);
  LK_CHECK(len != 0, "tokenizer: input is not valid UTF-8");
  return len;
}

void validate_utf8(std::string_view text) {
  for (size_t i = 0; i < text.size(); i += utf8_char_len(text, i)) {
  }
}

bool is_valid_utf8(std::string_view text) {
  for (size_t i = 0; i < text.size();) {
    const size_t len = unicode::utf8_sequence_length(text, i);
    if (len == 0) return false;
    i += len;
  }
  return true;
}

std::string replace_all(std::string_view text, const std::string& pattern, const std::string& content) {
  if (pattern.empty()) return std::string(text);
  std::string out;
  out.reserve(text.size());
  size_t pos = 0;
  while (true) {
    size_t found = text.find(pattern, pos);
    if (found == std::string_view::npos) break;
    out.append(text.substr(pos, found - pos));
    out.append(content);
    pos = found + pattern.size();
  }
  out.append(text.substr(pos));
  return out;
}

bool starts_with(const std::string& text, const std::string& prefix) {
  return text.compare(0, prefix.size(), prefix) == 0;
}

std::string string_pattern(const Json& node, const char* what) {
  const Json& pattern = node.at("pattern");
  LK_CHECK(pattern.contains("String"),
           std::string("tokenizer: only String patterns are supported for ") + what);
  return pattern.at("String").as_string();
}

int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

uint64_t pair_key(int32_t a, int32_t b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

struct Symbol {
  int32_t id;
  int32_t prev;
  int32_t next;
  size_t len;  // bytes covered; 0 marks a symbol merged into its left neighbour
};

struct Candidate {
  int32_t rank;
  int32_t pos;
  int32_t new_id;
  // std::priority_queue is a max-heap: invert so the lowest rank, then the
  // leftmost position, comes first (Hugging Face's BPE merge order).
  bool operator<(const Candidate& other) const {
    if (rank != other.rank) return rank > other.rank;
    return pos > other.pos;
  }
};

}  // namespace

Tokenizer Tokenizer::load(const std::string& tokenizer_json_path) {
  return from_json(Json::parse(read_text_file(tokenizer_json_path)));
}

Tokenizer Tokenizer::from_json(const Json& root) {
  Tokenizer t;
  t.byte_ids_.fill(-1);
  t.load_model(root.at("model"));
  t.load_added_tokens(root.find("added_tokens"));
  t.load_normalizer(root.find("normalizer"));
  t.load_pre_tokenizer(root.find("pre_tokenizer"));
  t.load_post_processor(root.find("post_processor"));
  t.load_decoder(root.find("decoder"));
  return t;
}

void Tokenizer::load_normalizer(const Json* node) {
  if (node == nullptr || node->is_null()) return;
  const std::string type = node->at("type").as_string();
  if (type == "Sequence") {
    for (const Json& step : node->at("normalizers").as_array()) load_normalizer(&step);
  } else if (type == "Replace") {
    normalizer_.push_back({false, string_pattern(*node, "Replace"), node->at("content").as_string()});
  } else if (type == "Prepend") {
    normalizer_.push_back({true, "", node->at("prepend").as_string()});
  } else {
    fail("tokenizer: unsupported normalizer '" + type + "'");
  }
}

void Tokenizer::load_pre_tokenizer(const Json* node) {
  if (node == nullptr || node->is_null()) return;
  const std::string type = node->at("type").as_string();
  if (type == "Split") {
    LK_CHECK(!(node->find("invert") && node->at("invert").as_bool()), "tokenizer: inverted Split is not supported");
    pre_kind_ = PreKind::Split;
    split_pattern_ = string_pattern(*node, "Split");
    LK_CHECK(!split_pattern_.empty(), "tokenizer: empty Split pattern");
    const std::string behavior = node->at("behavior").as_string();
    if (behavior == "Removed") split_behavior_ = SplitBehavior::Removed;
    else if (behavior == "Isolated") split_behavior_ = SplitBehavior::Isolated;
    else if (behavior == "MergedWithPrevious") split_behavior_ = SplitBehavior::MergedWithPrevious;
    else if (behavior == "MergedWithNext") split_behavior_ = SplitBehavior::MergedWithNext;
    else if (behavior == "Contiguous") split_behavior_ = SplitBehavior::Contiguous;
    else fail("tokenizer: unsupported Split behavior '" + behavior + "'");
  } else if (type == "Metaspace") {
    pre_kind_ = PreKind::Metaspace;
    metaspace_replacement_ = node->string_or("replacement", metaspace_replacement_);
    const std::string scheme = node->string_or("prepend_scheme", "always");
    metaspace_prepend_ = scheme == "first" ? PrependScheme::First
                         : scheme == "never" ? PrependScheme::Never
                                             : PrependScheme::Always;
    if (const Json* split = node->find("split")) metaspace_split_ = split->as_bool();
  } else {
    fail("tokenizer: unsupported pre_tokenizer '" + type + "'");
  }
}

void Tokenizer::load_model(const Json& model) {
  const std::string type = model.string_or("type", "BPE");
  LK_CHECK(type == "BPE", "tokenizer: unsupported model '" + type + "'");
  for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix", "dropout"}) {
    const Json* value = model.find(key);
    LK_CHECK(value == nullptr || value->is_null(), std::string("tokenizer: BPE ") + key + " is not supported");
  }
  byte_fallback_ = model.find("byte_fallback") && model.at("byte_fallback").as_bool();
  fuse_unk_ = model.find("fuse_unk") && model.at("fuse_unk").as_bool();
  ignore_merges_ = model.find("ignore_merges") && model.at("ignore_merges").as_bool();

  int32_t max_id = -1;
  const auto& vocab = model.at("vocab").as_object();
  vocab_.reserve(vocab.size());
  for (const auto& [token, id] : vocab) {
    const auto value = static_cast<int32_t>(id.as_int());
    vocab_.emplace(token, value);
    max_id = std::max(max_id, value);
  }
  id_to_token_.assign(static_cast<size_t>(max_id + 1), std::string());
  for (const auto& [token, id] : vocab_) id_to_token_[static_cast<size_t>(id)] = token;

  if (const Json* unk = model.find("unk_token"); unk != nullptr && unk->is_string()) {
    auto it = vocab_.find(unk->as_string());
    LK_CHECK(it != vocab_.end(), "tokenizer: unk_token is not in the vocabulary");
    unk_id_ = it->second;
  }
  for (int b = 0; b < 256; ++b) {
    char name[8];
    std::snprintf(name, sizeof(name), "<0x%02X>", b);
    auto it = vocab_.find(name);
    if (it != vocab_.end()) byte_ids_[static_cast<size_t>(b)] = it->second;
  }

  const auto& merges = model.at("merges").as_array();
  merges_.reserve(merges.size());
  for (size_t rank = 0; rank < merges.size(); ++rank) {
    std::string a, b;
    if (merges[rank].is_array()) {
      const auto& pair = merges[rank].as_array();
      LK_CHECK(pair.size() == 2, "tokenizer: malformed merge");
      a = pair[0].as_string();
      b = pair[1].as_string();
    } else {
      const std::string& text = merges[rank].as_string();
      const size_t space = text.find(' ');
      LK_CHECK(space != std::string::npos, "tokenizer: malformed merge '" + text + "'");
      a = text.substr(0, space);
      b = text.substr(space + 1);
    }
    auto ia = vocab_.find(a), ib = vocab_.find(b), im = vocab_.find(a + b);
    LK_CHECK(ia != vocab_.end() && ib != vocab_.end() && im != vocab_.end(),
             "tokenizer: merge '" + a + " " + b + "' refers to tokens outside the vocabulary");
    // A repeated pair keeps its last rank, as tokenizers' merge map does.
    merges_[pair_key(ia->second, ib->second)] = std::make_pair(static_cast<int32_t>(rank), im->second);
  }
}

void Tokenizer::load_added_tokens(const Json* node) {
  trie_.assign(1, TrieNode{});
  if (node == nullptr || node->is_null()) return;
  // tokenizers ignores the serialized ids and reassigns them: a token already
  // in the vocabulary keeps that id, otherwise it takes the next id after the
  // vocabulary and earlier added tokens. Reject files where the two disagree
  // instead of tokenizing differently from the reference.
  const auto vocab_size = static_cast<int64_t>(vocab_.size());
  int64_t max_added = -1;
  std::unordered_map<std::string, int32_t> seen;
  for (const Json& entry : node->as_array()) {
    AddedToken token{entry.at("content").as_string(), static_cast<int32_t>(entry.at("id").as_int()),
                     entry.find("special") && entry.at("special").as_bool()};
    for (const char* flag : {"normalized", "lstrip", "rstrip", "single_word"}) {
      const Json* value = entry.find(flag);
      LK_CHECK(value == nullptr || !value->as_bool(),
               "tokenizer: added token '" + token.content + "' uses unsupported option " + flag);
    }
    LK_CHECK(!token.content.empty() && token.id >= 0, "tokenizer: malformed added token");
    LK_CHECK(seen.emplace(token.content, token.id).second, "tokenizer: duplicate added token '" + token.content + "'");
    auto in_vocab = vocab_.find(token.content);
    const int64_t expected = in_vocab != vocab_.end() ? in_vocab->second
                             : (max_added >= vocab_size || vocab_size == 0) ? max_added + 1
                                                                            : vocab_size;
    LK_CHECK(expected == token.id, "tokenizer: added token '" + token.content + "' has id " +
                                       std::to_string(token.id) + " but tokenizers would assign " +
                                       std::to_string(expected));
    max_added = std::max(max_added, expected);
    if (static_cast<size_t>(token.id) >= id_to_token_.size()) id_to_token_.resize(static_cast<size_t>(token.id) + 1);
    id_to_token_[static_cast<size_t>(token.id)] = token.content;
    if (token.special) special_ids_.insert(token.id);

    int32_t node_index = 0;
    for (unsigned char c : token.content) {
      auto it = trie_[static_cast<size_t>(node_index)].next.find(c);
      if (it == trie_[static_cast<size_t>(node_index)].next.end()) {
        trie_.push_back(TrieNode{});
        const auto created = static_cast<int32_t>(trie_.size() - 1);
        trie_[static_cast<size_t>(node_index)].next.emplace(c, created);
        node_index = created;
      } else {
        node_index = it->second;
      }
    }
    trie_[static_cast<size_t>(node_index)].added = static_cast<int32_t>(added_.size());
    added_.push_back(std::move(token));
  }
}

void Tokenizer::load_post_processor(const Json* node) {
  if (node == nullptr || node->is_null()) return;
  const std::string type = node->at("type").as_string();
  if (type == "Sequence") {
    for (const Json& step : node->at("processors").as_array()) load_post_processor(&step);
    return;
  }
  if (type == "ByteLevel") return;  // Adjusts offsets only; ids are unchanged.
  LK_CHECK(type == "TemplateProcessing", "tokenizer: unsupported post_processor '" + type + "'");
  const Json& specials = node->at("special_tokens");
  bool seen_sequence = false;
  for (const Json& item : node->at("single").as_array()) {
    if (item.contains("Sequence")) {
      LK_CHECK(item.at("Sequence").at("id").as_string() == "A", "tokenizer: unsupported template sequence");
      seen_sequence = true;
      continue;
    }
    const std::string name = item.at("SpecialToken").at("id").as_string();
    for (const Json& id : specials.at(name).at("ids").as_array()) {
      (seen_sequence ? suffix_ids_ : prefix_ids_).push_back(static_cast<int32_t>(id.as_int()));
    }
  }
  LK_CHECK(seen_sequence, "tokenizer: template has no $A sequence");
}

void Tokenizer::load_decoder(const Json* node) {
  if (node == nullptr || node->is_null()) return;
  const std::string type = node->at("type").as_string();
  DecoderStep step{};
  if (type == "Sequence") {
    for (const Json& child : node->at("decoders").as_array()) load_decoder(&child);
    return;
  } else if (type == "Replace") {
    step.kind = DecoderStep::Kind::Replace;
    step.pattern = string_pattern(*node, "Replace decoder");
    step.content = node->at("content").as_string();
  } else if (type == "ByteFallback") {
    step.kind = DecoderStep::Kind::ByteFallback;
  } else if (type == "Fuse") {
    step.kind = DecoderStep::Kind::Fuse;
  } else if (type == "Strip") {
    step.kind = DecoderStep::Kind::Strip;
    step.content = node->at("content").as_string();
    LK_CHECK(utf8_char_len(step.content, 0) == step.content.size(), "tokenizer: Strip content must be one character");
    step.start = static_cast<int>(node->at("start").as_int());
    step.stop = static_cast<int>(node->at("stop").as_int());
  } else if (type == "Metaspace") {
    step.kind = DecoderStep::Kind::Metaspace;
    step.content = node->string_or("replacement", "\xE2\x96\x81");
    step.metaspace_prepend = node->string_or("prepend_scheme", "always") != "never";
  } else {
    fail("tokenizer: unsupported decoder '" + type + "'");
  }
  decoder_.push_back(std::move(step));
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
  for (const AddedToken& added : added_) {
    if (added.content == token) return added.id;
  }
  auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

const std::string& Tokenizer::id_to_token(int32_t id) const {
  LK_CHECK(id >= 0 && static_cast<size_t>(id) < id_to_token_.size() && !id_to_token_[static_cast<size_t>(id)].empty(),
           "tokenizer: unknown token id " + std::to_string(id));
  return id_to_token_[static_cast<size_t>(id)];
}

std::string Tokenizer::normalize(std::string_view text) const {
  std::string out(text);
  for (const NormalizerStep& step : normalizer_) {
    if (step.prepend) {
      if (!out.empty()) out = step.content + out;
    } else {
      out = replace_all(out, step.pattern, step.content);
    }
  }
  return out;
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& normalized, bool at_start) const {
  std::vector<std::string> pieces;
  if (normalized.empty()) return pieces;
  if (pre_kind_ == PreKind::None) {
    pieces.push_back(normalized);
    return pieces;
  }

  std::string text = normalized;
  std::string pattern = split_pattern_;
  SplitBehavior behavior = split_behavior_;
  if (pre_kind_ == PreKind::Metaspace) {
    text = replace_all(text, " ", metaspace_replacement_);
    const bool prepend = metaspace_prepend_ == PrependScheme::Always ||
                         (metaspace_prepend_ == PrependScheme::First && at_start);
    if (prepend && !starts_with(text, metaspace_replacement_)) text = metaspace_replacement_ + text;
    if (!metaspace_split_) {
      pieces.push_back(text);
      return pieces;
    }
    pattern = metaspace_replacement_;
    behavior = SplitBehavior::MergedWithNext;
  }

  // Alternating (begin, end, is_delimiter) spans covering the text.
  struct Span {
    size_t begin, end;
    bool delim;
  };
  std::vector<Span> spans;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t found = text.find(pattern, pos);
    if (found == std::string::npos) {
      spans.push_back({pos, text.size(), false});
      break;
    }
    if (found > pos) spans.push_back({pos, found, false});
    spans.push_back({found, found + pattern.size(), true});
    pos = found + pattern.size();
  }

  std::vector<std::pair<size_t, size_t>> merged;
  switch (behavior) {
    case SplitBehavior::Removed:
      for (const Span& s : spans)
        if (!s.delim) merged.emplace_back(s.begin, s.end);
      break;
    case SplitBehavior::Isolated:
      for (const Span& s : spans) merged.emplace_back(s.begin, s.end);
      break;
    case SplitBehavior::MergedWithPrevious: {
      bool previous_delim = false;
      for (const Span& s : spans) {
        if (s.delim && !previous_delim && !merged.empty()) merged.back().second = s.end;
        else merged.emplace_back(s.begin, s.end);
        previous_delim = s.delim;
      }
      break;
    }
    case SplitBehavior::MergedWithNext: {
      bool previous_delim = false;
      for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
        if (it->delim && !previous_delim && !merged.empty()) merged.back().first = it->begin;
        else merged.emplace_back(it->begin, it->end);
        previous_delim = it->delim;
      }
      std::reverse(merged.begin(), merged.end());
      break;
    }
    case SplitBehavior::Contiguous: {
      bool previous_delim = false;
      for (const Span& s : spans) {
        if (s.delim && previous_delim && !merged.empty()) merged.back().second = s.end;
        else merged.emplace_back(s.begin, s.end);
        previous_delim = s.delim;
      }
      break;
    }
  }
  for (const auto& [begin, end] : merged) {
    if (end > begin) pieces.push_back(text.substr(begin, end - begin));
  }
  return pieces;
}

void Tokenizer::bpe(const std::string& word, std::vector<int32_t>& out) const {
  if (ignore_merges_) {
    auto it = vocab_.find(word);
    if (it != vocab_.end()) {
      out.push_back(it->second);
      return;
    }
  }
  std::vector<Symbol> symbols;
  symbols.reserve(word.size());
  auto add = [&](int32_t id, size_t len) {
    const auto index = static_cast<int32_t>(symbols.size());
    symbols.push_back({id, index - 1, -1, len});
    if (index > 0) symbols[static_cast<size_t>(index - 1)].next = index;
  };
  // Mirrors tokenizers' BPE::merge_word, including that a pending unknown
  // run is flushed only before an in-vocabulary character, not before
  // byte-fallback tokens.
  bool have_unk = false;
  size_t unk_len = 0;
  for (size_t i = 0; i < word.size();) {
    const size_t len = utf8_char_len(word, i);
    auto it = vocab_.find(word.substr(i, len));
    if (it != vocab_.end()) {
      if (have_unk) {
        add(unk_id_, unk_len);
        have_unk = false;
      }
      add(it->second, len);
    } else {
      bool handled = false;
      if (byte_fallback_) {
        handled = true;
        for (size_t k = 0; k < len; ++k) handled = handled && byte_ids_[static_cast<unsigned char>(word[i + k])] >= 0;
        if (handled) {
          for (size_t k = 0; k < len; ++k) add(byte_ids_[static_cast<unsigned char>(word[i + k])], 1);
        }
      }
      if (!handled) {
        LK_CHECK(unk_id_ >= 0, "tokenizer: character is not in the vocabulary and there is no unk_token");
        if (have_unk && fuse_unk_) {
          unk_len += len;
        } else {
          if (have_unk) add(unk_id_, unk_len);
          have_unk = true;
          unk_len = len;
        }
      }
    }
    i += len;
  }
  if (have_unk) add(unk_id_, unk_len);

  std::priority_queue<Candidate> queue;
  auto lookup = [&](int32_t a, int32_t b) -> const std::pair<int32_t, int32_t>* {
    auto it = merges_.find(pair_key(a, b));
    return it == merges_.end() ? nullptr : &it->second;
  };
  for (size_t i = 0; i + 1 < symbols.size(); ++i) {
    if (auto* m = lookup(symbols[i].id, symbols[i + 1].id)) queue.push({m->first, static_cast<int32_t>(i), m->second});
  }
  while (!queue.empty()) {
    const Candidate top = queue.top();
    queue.pop();
    Symbol& left = symbols[static_cast<size_t>(top.pos)];
    if (left.len == 0 || left.next < 0) continue;
    const int32_t right_index = left.next;
    const Symbol right = symbols[static_cast<size_t>(right_index)];
    const auto* current = lookup(left.id, right.id);
    if (current == nullptr || current->second != top.new_id) continue;
    left.id = top.new_id;
    left.len += right.len;
    left.next = right.next;
    symbols[static_cast<size_t>(right_index)].len = 0;
    if (right.next >= 0) symbols[static_cast<size_t>(right.next)].prev = top.pos;
    if (left.prev >= 0) {
      if (auto* m = lookup(symbols[static_cast<size_t>(left.prev)].id, left.id)) {
        queue.push({m->first, left.prev, m->second});
      }
    }
    if (left.next >= 0) {
      if (auto* m = lookup(left.id, symbols[static_cast<size_t>(left.next)].id)) {
        queue.push({m->first, top.pos, m->second});
      }
    }
  }
  for (const Symbol& s : symbols) {
    if (s.len != 0) out.push_back(s.id);
  }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_special_tokens) const {
  validate_utf8(text);
  std::vector<int32_t> out;
  if (add_special_tokens) out = prefix_ids_;

  auto encode_segment = [&](size_t begin, size_t end) {
    if (end <= begin) return;
    const std::string normalized = normalize(text.substr(begin, end - begin));
    for (const std::string& piece : pre_tokenize(normalized, begin == 0)) bpe(piece, out);
  };

  // Added tokens are matched on the raw text, leftmost-longest.
  size_t segment_begin = 0;
  size_t i = 0;
  while (i < text.size()) {
    int32_t node = 0;
    int32_t matched = -1;
    size_t matched_end = 0;
    for (size_t j = i; j < text.size(); ++j) {
      auto it = trie_[static_cast<size_t>(node)].next.find(static_cast<unsigned char>(text[j]));
      if (it == trie_[static_cast<size_t>(node)].next.end()) break;
      node = it->second;
      if (trie_[static_cast<size_t>(node)].added >= 0) {
        matched = trie_[static_cast<size_t>(node)].added;
        matched_end = j + 1;
      }
    }
    if (matched < 0) {
      ++i;
      continue;
    }
    encode_segment(segment_begin, i);
    out.push_back(added_[static_cast<size_t>(matched)].id);
    i = matched_end;
    segment_begin = i;
  }
  encode_segment(segment_begin, text.size());
  if (add_special_tokens) out.insert(out.end(), suffix_ids_.begin(), suffix_ids_.end());
  return out;
}

size_t Tokenizer::stable_prefix(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
  bool byte_fallback = false;
  for (const DecoderStep& step : decoder_) {
    // Stripping trailing characters after Fuse depends on what comes later.
    if (step.kind == DecoderStep::Kind::Strip && step.stop > 0) return 0;
    byte_fallback = byte_fallback || step.kind == DecoderStep::Kind::ByteFallback;
  }
  size_t n = ids.size();
  if (!byte_fallback) return n;
  auto is_byte = [&](int32_t id) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return false;
    const std::string& token = id_to_token_[static_cast<size_t>(id)];
    return token.size() == 6 && token.compare(0, 3, "<0x") == 0 && token[5] == '>' && hex_digit(token[3]) >= 0 &&
           hex_digit(token[4]) >= 0;
  };
  while (n > 0 && (is_byte(ids[n - 1]) || (skip_special_tokens && is_special(ids[n - 1])))) --n;
  return n;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
  std::vector<std::string> tokens;
  tokens.reserve(ids.size());
  for (int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size() || id_to_token_[static_cast<size_t>(id)].empty()) {
      continue;  // Unknown ids are dropped, as in tokenizers.
    }
    if (skip_special_tokens && is_special(id)) continue;
    tokens.push_back(id_to_token_[static_cast<size_t>(id)]);
  }

  for (const DecoderStep& step : decoder_) {
    switch (step.kind) {
      case DecoderStep::Kind::Replace:
        for (std::string& token : tokens) token = replace_all(token, step.pattern, step.content);
        break;
      case DecoderStep::Kind::ByteFallback: {
        std::vector<std::string> next;
        std::string pending;
        auto flush = [&] {
          if (pending.empty()) return;
          if (is_valid_utf8(pending)) {
            next.push_back(pending);
          } else {
            for (size_t k = 0; k < pending.size(); ++k) next.push_back("\xEF\xBF\xBD");  // U+FFFD
          }
          pending.clear();
        };
        for (const std::string& token : tokens) {
          const int hi = token.size() == 6 ? hex_digit(token[3]) : -1;
          const int lo = token.size() == 6 ? hex_digit(token[4]) : -1;
          if (token.compare(0, 3, "<0x") == 0 && token.back() == '>' && hi >= 0 && lo >= 0) {
            pending.push_back(static_cast<char>(hi * 16 + lo));
          } else {
            flush();
            next.push_back(token);
          }
        }
        flush();
        tokens.swap(next);
        break;
      }
      case DecoderStep::Kind::Fuse: {
        std::string fused;
        for (const std::string& token : tokens) fused += token;
        tokens.assign(1, fused);
        break;
      }
      case DecoderStep::Kind::Strip:
        for (std::string& token : tokens) {
          for (int k = 0; k < step.start && starts_with(token, step.content); ++k) token.erase(0, step.content.size());
          for (int k = 0; k < step.stop && token.size() >= step.content.size() &&
                          token.compare(token.size() - step.content.size(), step.content.size(), step.content) == 0;
               ++k) {
            token.erase(token.size() - step.content.size());
          }
        }
        break;
      case DecoderStep::Kind::Metaspace:
        // As in tokenizers, every replacement character of the first token is
        // dropped (not just a leading one) when a prefix space was prepended.
        for (size_t k = 0; k < tokens.size(); ++k) {
          tokens[k] = replace_all(tokens[k], step.content, k == 0 && step.metaspace_prepend ? "" : " ");
        }
        break;
    }
  }
  std::string text;
  for (const std::string& token : tokens) text += token;
  return text;
}

}  // namespace lokahi
