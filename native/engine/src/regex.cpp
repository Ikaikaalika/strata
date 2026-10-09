#include "regex.h"

#include <algorithm>
#include <cstdint>

#include "common.h"
#include "unicode.h"

namespace lokahi {

namespace {

struct ClassItem {
  enum Kind : uint8_t { Range, Categories, Space, NotSpace } kind;
  char32_t lo = 0, hi = 0;
  uint32_t mask = 0;
  bool negate = false;  // for Categories (\P{..})

  bool matches(char32_t c) const {
    switch (kind) {
      case Range: return c >= lo && c <= hi;
      case Categories: return unicode::in(c, mask) != negate;
      case Space: return unicode::is_space(c);
      case NotSpace: return !unicode::is_space(c);
    }
    return false;
  }
};

// Oniguruma folds case with Unicode case folding. Case-insensitive groups
// may only contain ASCII literals, whose fold set is ASCII plus these.
char32_t fold(char32_t c) {
  if (c >= 'A' && c <= 'Z') return c + 32;
  if (c == 0x017F) return 's';  // LATIN SMALL LETTER LONG S
  if (c == 0x212A) return 'k';  // KELVIN SIGN
  return c;
}

}  // namespace

struct Regex::Node {
  enum Kind { Char, Class, Any, Group, Alt, Repeat, Look, LineStart, LineEnd } kind;
  const Node* next = nullptr;
  // Char
  char32_t ch = 0;
  bool icase = false;
  // Class
  std::vector<ClassItem> items;
  bool negated = false;
  // Group, Repeat, Look
  const Node* child = nullptr;
  // Alt
  std::vector<const Node*> alts;
  // Repeat
  int min = 0, max = -1;  // max < 0: unbounded
  enum Mode { Greedy, Lazy, Possessive } mode = Greedy;
  // Look
  bool look_negate = false;

  bool test(char32_t c) const {
    switch (kind) {
      case Char: return icase ? fold(c) == fold(ch) : c == ch;
      case Any: return c != '\n';
      case Class: {
        bool hit = false;
        for (const ClassItem& item : items) {
          if (item.matches(c)) {
            hit = true;
            break;
          }
        }
        return hit != negated;
      }
      default: return false;
    }
  }
  bool single_char() const { return kind == Char || kind == Class || kind == Any; }
};

namespace {

using Node = Regex::Node;

class Parser {
 public:
  Parser(std::u32string pattern, std::vector<std::unique_ptr<Node>>& arena)
      : p_(std::move(pattern)), arena_(arena) {}

  const Node* parse() {
    const Node* root = parse_alternation(false);
    LK_CHECK(pos_ == p_.size(), "regex: unbalanced ')'");
    return root;
  }

 private:
  [[noreturn]] void error(const std::string& what) const {
    fail("regex: " + what + " at offset " + std::to_string(pos_) + " in /" + unicode::encode_utf8(p_) + "/");
  }
  bool at_end() const { return pos_ >= p_.size(); }
  char32_t peek(size_t ahead = 0) const { return pos_ + ahead < p_.size() ? p_[pos_ + ahead] : 0; }
  char32_t take() {
    if (at_end()) error("unexpected end of pattern");
    return p_[pos_++];
  }
  Node* make(Node::Kind kind) {
    arena_.push_back(std::make_unique<Node>());
    arena_.back()->kind = kind;
    return arena_.back().get();
  }

  const Node* parse_alternation(bool icase) {
    std::vector<const Node*> alts = {parse_sequence(icase)};
    while (peek() == '|') {
      ++pos_;
      alts.push_back(parse_sequence(icase));
    }
    if (alts.size() == 1) return alts[0];
    Node* alt = make(Node::Alt);
    alt->alts = std::move(alts);
    return alt;
  }

  const Node* parse_sequence(bool icase) {
    Node* head = nullptr;
    Node* tail = nullptr;
    while (!at_end() && peek() != '|' && peek() != ')') {
      Node* atom = parse_quantified(icase);
      if (tail) tail->next = atom;
      else head = atom;
      tail = atom;
    }
    return head;
  }

  bool parse_count(int* value) {
    size_t start = pos_;
    int v = 0;
    while (peek() >= '0' && peek() <= '9') {
      v = v * 10 + static_cast<int>(take() - '0');
      if (v > 100000) error("repeat count too large");
    }
    *value = v;
    return pos_ > start;
  }

  Node* parse_quantified(bool icase) {
    Node* atom = parse_atom(icase);
    int min = -1, max = -1;
    const char32_t q = peek();
    if (q == '*') { ++pos_; min = 0; max = -1; }
    else if (q == '+') { ++pos_; min = 1; max = -1; }
    else if (q == '?') { ++pos_; min = 0; max = 1; }
    else if (q == '{') {
      // A '{' that does not start a valid interval is a literal (Oniguruma).
      const size_t save = pos_;
      ++pos_;
      int lo = 0, hi = -1;
      bool ok = parse_count(&lo);
      if (peek() == ',') {
        ++pos_;
        const bool has_hi = parse_count(&hi);
        if (!has_hi) hi = -1;
        ok = ok || has_hi;  // {,n} abbreviates {0,n}; {,} is a literal.
      } else {
        hi = lo;
      }
      if (ok && peek() == '}') {
        ++pos_;
        if (hi >= 0 && hi < lo) error("invalid repeat range");
        min = lo;
        max = hi;
      } else {
        pos_ = save;
      }
    }
    if (min < 0) return atom;
    if (atom->kind == Node::Look || atom->kind == Node::LineStart || atom->kind == Node::LineEnd) {
      error("quantifier on an assertion");
    }
    Node* repeat = make(Node::Repeat);
    repeat->child = atom;
    repeat->min = min;
    repeat->max = max;
    if (peek() == '?') { ++pos_; repeat->mode = Node::Lazy; }
    else if (peek() == '+' && q != '{') { ++pos_; repeat->mode = Node::Possessive; }
    return repeat;
  }

  Node* parse_atom(bool icase) {
    const char32_t c = take();
    switch (c) {
      case '(': return parse_group(icase);
      case '[': return parse_class(icase);
      case '.': return make(Node::Any);
      case '^': return make(Node::LineStart);
      case '$': return make(Node::LineEnd);
      case '\\': return parse_escape(icase);
      case '*': case '+': case '?': error("nothing to repeat");
      default: return literal(c, icase);
    }
  }

  Node* literal(char32_t c, bool icase) {
    if (icase && c >= 0x80) error("non-ASCII literal in a case-insensitive group is not supported");
    Node* node = make(Node::Char);
    node->ch = c;
    node->icase = icase && ((c | 32) >= 'a' && (c | 32) <= 'z');
    return node;
  }

  Node* parse_group(bool icase) {
    Node* group;
    if (peek() == '?') {
      ++pos_;
      const char32_t kind = take();
      if (kind == ':') {
        group = make(Node::Group);
      } else if (kind == '=' || kind == '!') {
        group = make(Node::Look);
        group->look_negate = kind == '!';
      } else if (kind == 'i' && peek() == ':') {
        ++pos_;
        group = make(Node::Group);
        icase = true;
      } else {
        error("unsupported group syntax (?" + unicode::encode_utf8(std::u32string(1, kind)) + "...)");
      }
    } else {
      group = make(Node::Group);  // Capturing groups only group here.
    }
    group->child = parse_alternation(icase);
    if (take() != ')') error("expected ')'");
    return group;
  }

  // Parses \p{Name} / \P{Name} after the 'p' or 'P'.
  ClassItem parse_property(bool negate) {
    if (take() != '{') error("expected '{' after \\p");
    if (peek() == '^') {
      ++pos_;
      negate = !negate;
    }
    // Oniguruma matches property names ignoring case, spaces, '_' and '-'.
    std::string name, key;
    while (peek() != '}') {
      const char32_t c = take();
      if (c >= 0x80) error("unsupported property name");
      name.push_back(static_cast<char>(c));
      if (c != ' ' && c != '_' && c != '-') key.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c + 32 : c));
    }
    ++pos_;
    using namespace unicode;
    static const std::pair<const char*, uint32_t> kNames[] = {
        {"l", kLetter}, {"letter", kLetter}, {"lc", bit(Lu) | bit(Ll) | bit(Lt)},
        {"casedletter", bit(Lu) | bit(Ll) | bit(Lt)}, {"lu", bit(Lu)}, {"uppercaseletter", bit(Lu)},
        {"ll", bit(Ll)}, {"lowercaseletter", bit(Ll)}, {"lt", bit(Lt)}, {"titlecaseletter", bit(Lt)},
        {"lm", bit(Lm)}, {"modifierletter", bit(Lm)}, {"lo", bit(Lo)}, {"otherletter", bit(Lo)},
        {"m", kMark}, {"mark", kMark}, {"combiningmark", kMark}, {"mn", bit(Mn)},
        {"nonspacingmark", bit(Mn)}, {"mc", bit(Mc)}, {"spacingmark", bit(Mc)}, {"me", bit(Me)},
        {"enclosingmark", bit(Me)}, {"n", kNumber}, {"number", kNumber}, {"nd", bit(Nd)},
        {"decimalnumber", bit(Nd)}, {"nl", bit(Nl)}, {"letternumber", bit(Nl)}, {"no", bit(No)},
        {"othernumber", bit(No)}, {"p", kPunctuation}, {"punctuation", kPunctuation}, {"pc", bit(Pc)},
        {"connectorpunctuation", bit(Pc)}, {"pd", bit(Pd)}, {"dashpunctuation", bit(Pd)}, {"ps", bit(Ps)},
        {"openpunctuation", bit(Ps)}, {"pe", bit(Pe)}, {"closepunctuation", bit(Pe)}, {"pi", bit(Pi)},
        {"initialpunctuation", bit(Pi)}, {"pf", bit(Pf)}, {"finalpunctuation", bit(Pf)}, {"po", bit(Po)},
        {"otherpunctuation", bit(Po)}, {"s", kSymbol}, {"symbol", kSymbol}, {"sm", bit(Sm)},
        {"mathsymbol", bit(Sm)}, {"sc", bit(Sc)}, {"currencysymbol", bit(Sc)}, {"sk", bit(Sk)},
        {"modifiersymbol", bit(Sk)}, {"so", bit(So)}, {"othersymbol", bit(So)}, {"z", kSeparator},
        {"separator", kSeparator}, {"zs", bit(Zs)}, {"spaceseparator", bit(Zs)}, {"zl", bit(Zl)},
        {"lineseparator", bit(Zl)}, {"zp", bit(Zp)}, {"paragraphseparator", bit(Zp)}, {"c", kOther},
        {"other", kOther}, {"cc", bit(Cc)}, {"control", bit(Cc)}, {"cf", bit(Cf)}, {"format", bit(Cf)},
        {"cs", bit(Cs)}, {"surrogate", bit(Cs)}, {"co", bit(Co)}, {"privateuse", bit(Co)},
        {"cn", bit(Cn)}, {"unassigned", bit(Cn)},
    };
    for (const auto& [known, mask] : kNames) {
      if (key == known) {
        ClassItem item{ClassItem::Categories};
        item.mask = mask;
        item.negate = negate;
        return item;
      }
    }
    error("unsupported property \\p{" + name + "}");
  }

  uint32_t parse_hex(size_t digits) {
    uint32_t value = 0;
    for (size_t i = 0; i < digits; ++i) {
      const char32_t h = take();
      value <<= 4;
      if (h >= '0' && h <= '9') value |= h - '0';
      else if (h >= 'a' && h <= 'f') value |= h - 'a' + 10;
      else if (h >= 'A' && h <= 'F') value |= h - 'A' + 10;
      else error("invalid hex escape");
    }
    return value;
  }

  // Escapes that denote one character; returns false for class escapes.
  bool char_escape(char32_t e, char32_t* out) {
    switch (e) {
      case 'n': *out = '\n'; return true;
      case 'r': *out = '\r'; return true;
      case 't': *out = '\t'; return true;
      case 'f': *out = '\f'; return true;
      case 'v': *out = '\v'; return true;
      case 'a': *out = 0x07; return true;
      case 'e': *out = 0x1B; return true;
      case 'x':
        if (peek() == '{') {
          ++pos_;
          uint32_t value = 0;
          while (peek() != '}') value = (value << 4) | parse_hex(1);
          ++pos_;
          *out = value;
        } else {
          *out = parse_hex(2);
        }
        return true;
      case 'u': *out = parse_hex(4); return true;
      default:
        if ((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z') || (e >= '0' && e <= '9')) return false;
        *out = e;  // Escaped punctuation is literal.
        return true;
    }
  }

  // Class escapes (\s \S \d \D \p \P) as one class item; false otherwise.
  bool class_escape(char32_t e, ClassItem* item) {
    switch (e) {
      case 's': *item = ClassItem{ClassItem::Space}; return true;
      case 'S': *item = ClassItem{ClassItem::NotSpace}; return true;
      case 'd':
      case 'D':
        *item = ClassItem{ClassItem::Categories};
        item->mask = unicode::bit(unicode::Nd);
        item->negate = e == 'D';
        return true;
      case 'p': *item = parse_property(false); return true;
      case 'P': *item = parse_property(true); return true;
      default: return false;
    }
  }

  Node* parse_escape(bool icase) {
    const char32_t e = take();
    ClassItem item{ClassItem::Range};
    if (class_escape(e, &item)) {
      if (icase) error("class escape in a case-insensitive group is not supported");
      Node* node = make(Node::Class);
      node->items.push_back(item);
      return node;
    }
    char32_t c;
    if (!char_escape(e, &c)) error("unsupported escape \\" + unicode::encode_utf8(std::u32string(1, e)));
    return literal(c, icase);
  }

  Node* parse_class(bool icase) {
    if (icase) error("character class in a case-insensitive group is not supported");
    Node* node = make(Node::Class);
    if (peek() == '^') {
      ++pos_;
      node->negated = true;
    }
    bool first = true;
    while (true) {
      char32_t c = take();
      if (c == ']' && !first) break;
      first = false;
      if (c == '[') error("nested character classes are not supported");
      if (c == '&' && peek() == '&') error("class intersection is not supported");
      if (c == '\\') {
        const char32_t e = take();
        ClassItem item{ClassItem::Range};
        if (class_escape(e, &item)) {
          node->items.push_back(item);
          continue;
        }
        if (!char_escape(e, &c)) error("unsupported escape in class");
      }
      char32_t hi = c;
      if (peek() == '-' && peek(1) != ']' && peek(1) != 0) {
        ++pos_;
        hi = take();
        if (hi == '\\') {
          const char32_t e = take();
          if (!char_escape(e, &hi)) error("invalid range end");
        } else if (hi == '[') {
          error("nested character classes are not supported");
        }
        if (hi < c) error("invalid class range");
      }
      ClassItem item{ClassItem::Range};
      item.lo = c;
      item.hi = hi;
      node->items.push_back(item);
    }
    return node;
  }

  std::u32string p_;
  size_t pos_ = 0;
  std::vector<std::unique_ptr<Node>>& arena_;
};

// Continuation: what to match after the current node list ends. A repeat
// continuation re-enters its loop after one more iteration.
struct Cont {
  const Node* node;
  const Cont* next;
  const Node* repeat = nullptr;
  int count = 0;
  size_t iteration_start = 0;
};

// Each iteration of a repeated group costs a few stack frames, and tokenizers
// may run on threads with small stacks (512 KiB for Swift tasks). No
// tokenizer pattern repeats a group anywhere near this often.
constexpr int kMaxGroupIterations = 256;

class Matcher {
 public:
  explicit Matcher(std::u32string_view text) : text_(text) {}

  bool run(const Node* n, size_t pos, const Cont* k, size_t* end) const {
    while (true) {
      if (n == nullptr) {
        if (k == nullptr) {
          *end = pos;
          return true;
        }
        if (k->repeat) return loop(k->repeat, k->count, pos, k->iteration_start, k->next, end);
        n = k->node;
        k = k->next;
        continue;
      }
      switch (n->kind) {
        case Node::Char:
        case Node::Class:
        case Node::Any:
          if (pos < text_.size() && n->test(text_[pos])) {
            ++pos;
            n = n->next;
            continue;
          }
          return false;
        case Node::LineStart:
          if (pos != 0 && text_[pos - 1] != '\n') return false;
          n = n->next;
          continue;
        case Node::LineEnd:
          if (pos != text_.size() && text_[pos] != '\n') return false;
          n = n->next;
          continue;
        case Node::Group: {
          const Cont after{n->next, k};
          return run(n->child, pos, &after, end);
        }
        case Node::Alt: {
          const Cont after{n->next, k};
          for (const Node* alt : n->alts) {
            if (run(alt, pos, &after, end)) return true;
          }
          return false;
        }
        case Node::Look: {
          size_t ignored;
          if (run(n->child, pos, nullptr, &ignored) == n->look_negate) return false;
          n = n->next;
          continue;
        }
        case Node::Repeat:
          if (n->child->single_char() && n->child->next == nullptr) return repeat_char(n, pos, k, end);
          return loop(n, 0, pos, static_cast<size_t>(-1), k, end);
      }
    }
  }

 private:
  // Repeats of one character never need nested continuations.
  bool repeat_char(const Node* r, size_t pos, const Cont* k, size_t* end) const {
    const size_t room = text_.size() - pos;
    const size_t limit = r->max < 0 ? room : std::min(room, static_cast<size_t>(r->max));
    size_t count = 0;
    while (count < limit && r->child->test(text_[pos + count])) ++count;
    const auto min = static_cast<size_t>(r->min);
    if (count < min) return false;
    switch (r->mode) {
      case Node::Possessive:
        return run(r->next, pos + count, k, end);
      case Node::Greedy:
        for (size_t i = count + 1; i-- > min;) {
          if (run(r->next, pos + i, k, end)) return true;
        }
        return false;
      case Node::Lazy:
        for (size_t i = min; i <= count; ++i) {
          if (run(r->next, pos + i, k, end)) return true;
        }
        return false;
    }
    return false;
  }

  // General repeat after `count` completed iterations; the last one started
  // at `iteration_start`. An empty iteration ends the loop.
  bool loop(const Node* r, int count, size_t pos, size_t iteration_start, const Cont* k, size_t* end) const {
    LK_CHECK(count <= kMaxGroupIterations, "regex: group repeated too many times");
    const bool can_repeat = (r->max < 0 || count < r->max) && !(count > 0 && pos == iteration_start);
    if (r->mode == Node::Possessive) {
      while ((r->max < 0 || count < r->max)) {
        size_t next_pos;
        if (!run(r->child, pos, nullptr, &next_pos) || next_pos == pos) break;
        pos = next_pos;
        ++count;
      }
      return count >= r->min && run(r->next, pos, k, end);
    }
    const Cont again{nullptr, k, r, count + 1, pos};
    if (r->mode == Node::Greedy) {
      if (can_repeat && run(r->child, pos, &again, end)) return true;
      return count >= r->min && run(r->next, pos, k, end);
    }
    if (count >= r->min && run(r->next, pos, k, end)) return true;
    return can_repeat && run(r->child, pos, &again, end);
  }

  std::u32string_view text_;
};

}  // namespace

Regex::Regex() = default;
Regex::Regex(Regex&&) noexcept = default;
Regex& Regex::operator=(Regex&&) noexcept = default;
Regex::~Regex() = default;

Regex Regex::compile(std::string_view utf8_pattern) {
  Regex regex;
  Parser parser(unicode::decode_utf8(utf8_pattern), regex.nodes_);
  regex.root_ = parser.parse();
  return regex;
}

bool Regex::search(std::u32string_view text, size_t from, size_t* begin, size_t* end) const {
  const Matcher matcher(text);
  for (size_t start = from; start <= text.size(); ++start) {
    if (matcher.run(root_, start, nullptr, end)) {
      *begin = start;
      return true;
    }
  }
  return false;
}

}  // namespace lokahi
