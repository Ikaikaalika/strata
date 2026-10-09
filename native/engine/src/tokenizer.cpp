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

// Rust's String::from_utf8_lossy: each maximal ill-formed subpart becomes one
// U+FFFD.
std::string utf8_lossy(std::string_view bytes) {
  std::string out;
  out.reserve(bytes.size());
  for (size_t i = 0; i < bytes.size();) {
    if (size_t len = unicode::utf8_sequence_length(bytes, i)) {
      out.append(bytes.substr(i, len));
      i += len;
      continue;
    }
    size_t needed;
    i += std::max<size_t>(1, unicode::utf8_valid_prefix(bytes, i, &needed));
    out.append("\xEF\xBF\xBD");
  }
  return out;
}

// Whether the bytes end with an incomplete but so far well-formed sequence,
// which later bytes could still complete.
bool ends_inside_sequence(std::string_view bytes) {
  for (size_t back = 1; back <= 3 && back <= bytes.size(); ++back) {
    if ((static_cast<unsigned char>(bytes[bytes.size() - back]) & 0xC0) == 0x80) continue;
    size_t needed;
    return unicode::utf8_valid_prefix(bytes, bytes.size() - back, &needed) == back && needed > back;
  }
  return false;
}

// GPT-2's reversible byte-to-character mapping: printable Latin-1 bytes map
// to themselves and the rest to U+0100 onwards, so every byte has a visible
// character in the vocabulary.
struct ByteLevelAlphabet {
  std::array<std::string, 256> encode;  // byte -> UTF-8 character
  std::array<int16_t, 324> decode;      // code point -> byte, or -1
  ByteLevelAlphabet() {
    decode.fill(-1);
    char32_t next = 256;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
      const char32_t cp = printable ? static_cast<char32_t>(b) : next++;
      unicode::append_utf8(encode[static_cast<size_t>(b)], cp);
      decode[cp] = static_cast<int16_t>(b);
    }
  }
};

const ByteLevelAlphabet& byte_level() {
  static const ByteLevelAlphabet alphabet;
  return alphabet;
}

// The ByteLevel pre-tokenizer's built-in GPT-2 split pattern.
const Regex& gpt2_regex() {
  static const Regex regex =
      Regex::compile(R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)");
  return regex;
}

// Bytes a byte-level token stands for; tokens with characters outside the
// alphabet (such as added tokens) stand for their own UTF-8, as in tokenizers.
void append_byte_level_bytes(const std::string& token, std::string& out) {
  const ByteLevelAlphabet& alphabet = byte_level();
  const size_t start = out.size();
  for (size_t i = 0, len = 0; i < token.size(); i += len) {
    const char32_t cp = unicode::decode_utf8_at(token, i, &len);
    if (cp >= alphabet.decode.size() || alphabet.decode[cp] < 0) {
      out.resize(start);
      out.append(token);
      return;
    }
    out.push_back(static_cast<char>(alphabet.decode[cp]));
  }
}

bool flag(const Json& node, const char* key, bool fallback) {
  const Json* value = node.find(key);
  return value == nullptr || value->is_null() ? fallback : value->as_bool();
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
  // Before the added tokens: normalized ones are matched in normalized form.
  t.load_normalizer(root.find("normalizer"));
  t.load_pre_tokenizer(root.find("pre_tokenizer"));
  t.load_added_tokens(root.find("added_tokens"));
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
    normalizer_.push_back({NormalizerStep::Kind::Replace, string_pattern(*node, "Replace"),
                           node->at("content").as_string()});
  } else if (type == "Prepend") {
    normalizer_.push_back({NormalizerStep::Kind::Prepend, "", node->at("prepend").as_string()});
  } else if (type == "NFC") {
    normalizer_.push_back({NormalizerStep::Kind::NFC, "", ""});
  } else {
    fail("tokenizer: unsupported normalizer '" + type + "'");
  }
}

void Tokenizer::load_pre_tokenizer(const Json* node) {
  if (node == nullptr || node->is_null()) return;
  const std::string type = node->at("type").as_string();
  PreTokenizerStep step{};
  if (type == "Sequence") {
    for (const Json& child : node->at("pretokenizers").as_array()) load_pre_tokenizer(&child);
    return;
  } else if (type == "Split") {
    LK_CHECK(!flag(*node, "invert", false), "tokenizer: inverted Split is not supported");
    step.kind = PreTokenizerStep::Kind::Split;
    const Json& pattern = node->at("pattern");
    if (pattern.contains("Regex")) {
      const std::string& source = pattern.at("Regex").as_string();
      step.regex = std::make_shared<const Regex>(Regex::compile(source));
    } else {
      step.literal = string_pattern(*node, "Split");
      LK_CHECK(!step.literal.empty(), "tokenizer: empty Split pattern");
    }
    const std::string behavior = node->at("behavior").as_string();
    if (behavior == "Removed") step.behavior = SplitBehavior::Removed;
    else if (behavior == "Isolated") step.behavior = SplitBehavior::Isolated;
    else if (behavior == "MergedWithPrevious") step.behavior = SplitBehavior::MergedWithPrevious;
    else if (behavior == "MergedWithNext") step.behavior = SplitBehavior::MergedWithNext;
    else if (behavior == "Contiguous") step.behavior = SplitBehavior::Contiguous;
    else fail("tokenizer: unsupported Split behavior '" + behavior + "'");
  } else if (type == "Metaspace") {
    step.kind = PreTokenizerStep::Kind::Metaspace;
    step.replacement = node->string_or("replacement", "\xE2\x96\x81");  // U+2581
    const std::string scheme = node->string_or("prepend_scheme", "always");
    step.prepend = scheme == "first" ? PrependScheme::First
                   : scheme == "never" ? PrependScheme::Never
                                       : PrependScheme::Always;
    step.split = flag(*node, "split", true);
  } else if (type == "ByteLevel") {
    step.kind = PreTokenizerStep::Kind::ByteLevel;
    step.add_prefix_space = flag(*node, "add_prefix_space", true);
    step.use_regex = flag(*node, "use_regex", true);
  } else {
    fail("tokenizer: unsupported pre_tokenizer '" + type + "'");
  }
  pre_tokenizer_.push_back(std::move(step));
}

void Tokenizer::load_model(const Json& model) {
  const std::string type = model.string_or("type", "BPE");
  LK_CHECK(type == "BPE", "tokenizer: unsupported model '" + type + "'");
  // An empty prefix or suffix (Qwen writes "") is the same as none.
  for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const Json* value = model.find(key);
    LK_CHECK(value == nullptr || value->is_null() || value->as_string().empty(),
             std::string("tokenizer: BPE ") + key + " is not supported");
  }
  const Json* dropout = model.find("dropout");
  LK_CHECK(dropout == nullptr || dropout->is_null(), "tokenizer: BPE dropout is not supported");
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
                     flag(entry, "special", false)};
    for (const char* option : {"lstrip", "rstrip", "single_word"}) {
      LK_CHECK(!flag(entry, option, false),
               "tokenizer: added token '" + token.content + "' uses unsupported option " + option);
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

    // A normalized token is matched in normalized text, against its own
    // content run through the normalizer (Prepend applies to it too).
    const auto index = static_cast<int32_t>(added_.size());
    if (flag(entry, "normalized", !token.special)) {
      const std::string key = normalize(token.content);
      LK_CHECK(!key.empty(), "tokenizer: added token '" + token.content + "' normalizes to nothing");
      normalized_trie_.insert(key, index);
    } else {
      raw_trie_.insert(token.content, index);
    }
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
  } else if (type == "ByteLevel") {
    step.kind = DecoderStep::Kind::ByteLevel;  // Its options only affect offsets.
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

void Tokenizer::Trie::insert(std::string_view key, int32_t value) {
  int32_t node = 0;
  for (unsigned char c : key) {
    auto it = nodes[static_cast<size_t>(node)].next.find(c);
    if (it == nodes[static_cast<size_t>(node)].next.end()) {
      nodes.emplace_back();
      const auto created = static_cast<int32_t>(nodes.size() - 1);
      nodes[static_cast<size_t>(node)].next.emplace(c, created);
      node = created;
    } else {
      node = it->second;
    }
  }
  nodes[static_cast<size_t>(node)].value = value;
}

int32_t Tokenizer::Trie::longest_match(std::string_view text, size_t begin, size_t* end) const {
  int32_t node = 0;
  int32_t matched = -1;
  for (size_t j = begin; j < text.size(); ++j) {
    const auto& next = nodes[static_cast<size_t>(node)].next;
    auto it = next.find(static_cast<unsigned char>(text[j]));
    if (it == next.end()) break;
    node = it->second;
    if (nodes[static_cast<size_t>(node)].value >= 0) {
      matched = nodes[static_cast<size_t>(node)].value;
      *end = j + 1;
    }
  }
  return matched;
}

std::string Tokenizer::normalize(std::string_view text) const {
  std::string out(text);
  for (const NormalizerStep& step : normalizer_) {
    switch (step.kind) {
      case NormalizerStep::Kind::Replace:
        out = replace_all(out, step.pattern, step.content);
        break;
      case NormalizerStep::Kind::Prepend:
        if (!out.empty()) out = step.content + out;
        break;
      case NormalizerStep::Kind::NFC:
        out = unicode::nfc(out);
        break;
    }
  }
  return out;
}

namespace {

struct Span {
  size_t begin, end;
  bool match;
};

// Alternating (begin, end, match) byte ranges covering `text`, following
// tokenizers' find_matches. Regex matches use the onig crate's find_iter: an
// empty match right where the previous match ended is skipped by advancing
// one character.
std::vector<Span> find_matches(std::string_view text, const std::string& literal, const Regex* regex) {
  std::vector<Span> spans;
  size_t previous = 0;
  auto add_match = [&](size_t begin, size_t end) {
    if (begin != previous) spans.push_back({previous, begin, false});
    spans.push_back({begin, end, true});
    previous = end;
  };
  if (regex == nullptr) {
    for (size_t found = text.find(literal); found != std::string_view::npos;
         found = text.find(literal, found + literal.size())) {
      add_match(found, found + literal.size());
    }
  } else {
    std::u32string chars;
    std::vector<size_t> offsets;  // byte offset of each character, plus the end
    chars.reserve(text.size());
    offsets.reserve(text.size() + 1);
    for (size_t i = 0, len = 0; i < text.size(); i += len) {
      offsets.push_back(i);
      chars.push_back(unicode::decode_utf8_at(text, i, &len));
    }
    offsets.push_back(text.size());
    size_t last_end = 0;
    bool have_match = false;
    size_t last_match_end = 0;
    while (last_end <= chars.size()) {
      size_t begin, end;
      if (!regex->search(chars, last_end, &begin, &end)) break;
      if (begin == end && have_match && last_match_end == end) {
        ++last_end;
        continue;
      }
      last_end = end;
      have_match = true;
      last_match_end = end;
      add_match(offsets[begin], offsets[end]);
    }
  }
  if (previous != text.size()) spans.push_back({previous, text.size(), false});
  return spans;
}

// Groups spans into pieces by delimiter behavior (tokenizers'
// NormalizedString::split); empty pieces are dropped later.
std::vector<std::pair<size_t, size_t>> apply_behavior(const std::vector<Span>& spans, Tokenizer::SplitBehavior behavior) {
  using B = Tokenizer::SplitBehavior;
  std::vector<std::pair<size_t, size_t>> merged;
  switch (behavior) {
    case B::Removed:
      for (const Span& s : spans)
        if (!s.match) merged.emplace_back(s.begin, s.end);
      break;
    case B::Isolated:
      for (const Span& s : spans) merged.emplace_back(s.begin, s.end);
      break;
    case B::MergedWithPrevious: {
      bool previous_match = false;
      for (const Span& s : spans) {
        if (s.match && !previous_match && !merged.empty()) merged.back().second = s.end;
        else merged.emplace_back(s.begin, s.end);
        previous_match = s.match;
      }
      break;
    }
    case B::MergedWithNext: {
      bool previous_match = false;
      for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
        if (it->match && !previous_match && !merged.empty()) merged.back().first = it->begin;
        else merged.emplace_back(it->begin, it->end);
        previous_match = it->match;
      }
      std::reverse(merged.begin(), merged.end());
      break;
    }
    case B::Contiguous: {
      bool previous_match = false;
      for (const Span& s : spans) {
        if (s.match && previous_match && !merged.empty()) merged.back().second = s.end;
        else merged.emplace_back(s.begin, s.end);
        previous_match = s.match;
      }
      break;
    }
  }
  return merged;
}

}  // namespace

std::vector<Tokenizer::Word> Tokenizer::pre_tokenize(std::vector<Word> words) const {
  std::vector<Word> next;
  // Splits every word and keeps the non-empty pieces; only a piece at the
  // start of a word that starts the input still starts the input.
  auto split = [&](const Word& word, const std::string& literal, const Regex* regex, SplitBehavior behavior) {
    for (const auto& [begin, end] : apply_behavior(find_matches(word.text, literal, regex), behavior)) {
      if (end > begin) next.push_back({word.text.substr(begin, end - begin), word.at_start && begin == 0});
    }
  };
  for (const PreTokenizerStep& step : pre_tokenizer_) {
    next.clear();
    for (Word& word : words) {
      switch (step.kind) {
        case PreTokenizerStep::Kind::Split:
          split(word, step.literal, step.regex.get(), step.behavior);
          break;
        case PreTokenizerStep::Kind::Metaspace: {
          word.text = replace_all(word.text, " ", step.replacement);
          const bool prepend = step.prepend == PrependScheme::Always ||
                               (step.prepend == PrependScheme::First && word.at_start);
          if (prepend && !starts_with(word.text, step.replacement)) word.text = step.replacement + word.text;
          if (step.split) split(word, step.replacement, nullptr, SplitBehavior::MergedWithNext);
          else next.push_back(std::move(word));
          break;
        }
        case PreTokenizerStep::Kind::ByteLevel: {
          if (step.add_prefix_space && !starts_with(word.text, " ")) word.text.insert(0, " ");
          const size_t first = next.size();
          if (step.use_regex) split(word, "", &gpt2_regex(), SplitBehavior::Isolated);
          else next.push_back(std::move(word));
          const ByteLevelAlphabet& alphabet = byte_level();
          for (size_t k = first; k < next.size(); ++k) {
            std::string mapped;
            mapped.reserve(next[k].text.size() * 2);
            for (unsigned char c : next[k].text) mapped += alphabet.encode[c];
            next[k].text = std::move(mapped);
          }
          break;
        }
      }
    }
    words.swap(next);
  }
  return words;
}

std::vector<std::string> Tokenizer::words(std::string_view text) const {
  validate_utf8(text);
  std::vector<std::string> out;
  const std::string normalized = normalize(text);
  if (normalized.empty()) return out;
  for (Word& word : pre_tokenize({{normalized, true}})) out.push_back(std::move(word.text));
  return out;
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

  // Leftmost-longest added-token matches split `input`; the text between
  // them goes to `on_text` as byte ranges.
  auto split_added = [&](const Trie& trie, std::string_view input, auto&& on_text) {
    size_t segment = 0;
    for (size_t i = 0; !trie.empty() && i < input.size();) {
      size_t end = 0;
      const int32_t matched = trie.longest_match(input, i, &end);
      if (matched < 0) {
        ++i;
        continue;
      }
      if (i > segment) on_text(segment, i);
      out.push_back(added_[static_cast<size_t>(matched)].id);
      i = segment = end;
    }
    if (input.size() > segment) on_text(segment, input.size());
  };
  // As in tokenizers: added tokens that are not normalized are matched in
  // the input, each remaining piece is normalized on its own, normalized
  // added tokens are matched in that, and the rest is pre-tokenized.
  split_added(raw_trie_, text, [&](size_t begin, size_t end) {
    const std::string normalized = normalize(text.substr(begin, end - begin));
    split_added(normalized_trie_, normalized, [&](size_t b, size_t e) {
      for (const Word& word : pre_tokenize({{normalized.substr(b, e - b), begin == 0 && b == 0}})) bpe(word.text, out);
    });
  });
  if (add_special_tokens) out.insert(out.end(), suffix_ids_.begin(), suffix_ids_.end());
  return out;
}

size_t Tokenizer::stable_prefix(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
  bool byte_fallback = false, byte_level = false;
  for (const DecoderStep& step : decoder_) {
    // Stripping trailing characters after Fuse depends on what comes later.
    if (step.kind == DecoderStep::Kind::Strip && step.stop > 0) return 0;
    byte_fallback = byte_fallback || step.kind == DecoderStep::Kind::ByteFallback;
    byte_level = byte_level || step.kind == DecoderStep::Kind::ByteLevel;
  }
  auto decoded = [&](int32_t id) -> const std::string* {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size() || id_to_token_[static_cast<size_t>(id)].empty()) {
      return nullptr;  // dropped by decode
    }
    if (skip_special_tokens && is_special(id)) return nullptr;
    return &id_to_token_[static_cast<size_t>(id)];
  };
  size_t n = ids.size();
  if (byte_fallback) {
    auto is_byte = [&](int32_t id) {
      const std::string* token = decoded(id);
      return token != nullptr && token->size() == 6 && token->compare(0, 3, "<0x") == 0 && (*token)[5] == '>' &&
             hex_digit((*token)[3]) >= 0 && hex_digit((*token)[4]) >= 0;
    };
    // Trailing byte tokens, and ids decode drops between them, stay pending.
    while (n > 0 && (is_byte(ids[n - 1]) || decoded(ids[n - 1]) == nullptr)) --n;
  }
  if (byte_level) {
    // The longest prefix whose bytes do not end inside a UTF-8 sequence:
    // lossy decoding of it cannot change when more bytes follow. Only the
    // last three bytes decide that.
    for (; n > 0; --n) {
      std::string tail;
      for (size_t k = n; k > 0 && tail.size() < 3; --k) {
        if (const std::string* token = decoded(ids[k - 1])) {
          std::string bytes;
          append_byte_level_bytes(*token, bytes);
          tail.insert(0, bytes);
        }
      }
      if (!ends_inside_sequence(tail)) break;
    }
  }
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
      case DecoderStep::Kind::ByteLevel: {
        std::string bytes;
        for (const std::string& token : tokens) append_byte_level_bytes(token, bytes);
        tokens.assign(1, utf8_lossy(bytes));
        break;
      }
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
