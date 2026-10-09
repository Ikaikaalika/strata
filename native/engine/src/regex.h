// A backtracking regex engine for the Oniguruma subset that Hugging Face
// tokenizer pre-tokenizers use: literals, Unicode property classes, \s/\S/\d,
// alternation, (?:), (?i:) over ASCII literals, lookahead, and greedy, lazy and
// possessive quantifiers. Matching follows Oniguruma's leftmost-first
// semantics. Unsupported syntax fails at compile time.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lokahi {

class Regex {
 public:
  static Regex compile(std::string_view utf8_pattern);

  Regex(Regex&&) noexcept;
  Regex& operator=(Regex&&) noexcept;
  ~Regex();

  // Leftmost match starting at or after `from` (code-point indices). Returns
  // false when there is none.
  bool search(std::u32string_view text, size_t from, size_t* begin, size_t* end) const;

  struct Node;

 private:
  Regex();
  std::vector<std::unique_ptr<Node>> nodes_;
  const Node* root_ = nullptr;
};

}  // namespace lokahi
