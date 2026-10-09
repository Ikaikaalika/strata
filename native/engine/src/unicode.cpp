#include "unicode.h"

#include <algorithm>
#include <vector>

#include "common.h"

namespace lokahi::unicode {
namespace {

struct CategoryRange {
  uint32_t first, last, category;
};
struct CombiningRange {
  uint32_t first, last, ccc;
};
struct Decomposition {
  uint32_t cp, first, second;  // second == 0 for singleton decompositions
};
struct Composition {
  uint32_t first, second, composite;
};

#include "unicode_data.inc"

constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount,
                   kSCount = kLCount * kNCount;

uint8_t combining_class(char32_t cp) {
  if (cp < 0x300) return 0;
  const auto* end = kCombiningClasses + sizeof(kCombiningClasses) / sizeof(kCombiningClasses[0]);
  const auto* it = std::upper_bound(kCombiningClasses, end, cp,
                                    [](char32_t value, const CombiningRange& r) { return value < r.first; });
  if (it == kCombiningClasses) return 0;
  --it;
  return cp <= it->last ? static_cast<uint8_t>(it->ccc) : 0;
}

void decompose(char32_t cp, std::u32string& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {
    const char32_t index = cp - kSBase;
    out.push_back(kLBase + index / kNCount);
    out.push_back(kVBase + (index % kNCount) / kTCount);
    if (index % kTCount) out.push_back(kTBase + index % kTCount);
    return;
  }
  const auto* end = kDecompositions + sizeof(kDecompositions) / sizeof(kDecompositions[0]);
  const auto* it =
      std::lower_bound(kDecompositions, end, cp, [](const Decomposition& d, char32_t value) { return d.cp < value; });
  if (it == end || it->cp != cp) {
    out.push_back(cp);
    return;
  }
  decompose(it->first, out);
  if (it->second) decompose(it->second, out);
}

char32_t compose(char32_t first, char32_t second) {
  if (first >= kLBase && first < kLBase + kLCount && second >= kVBase && second < kVBase + kVCount) {
    return kSBase + ((first - kLBase) * kVCount + (second - kVBase)) * kTCount;
  }
  if (first >= kSBase && first < kSBase + kSCount && (first - kSBase) % kTCount == 0 && second > kTBase &&
      second < kTBase + kTCount) {
    return first + (second - kTBase);
  }
  const auto* end = kCompositions + sizeof(kCompositions) / sizeof(kCompositions[0]);
  const auto* it = std::lower_bound(kCompositions, end, std::make_pair(first, second),
                                    [](const Composition& c, const std::pair<char32_t, char32_t>& key) {
                                      return c.first != key.first ? c.first < key.first : c.second < key.second;
                                    });
  return it != end && it->first == first && it->second == second ? it->composite : 0;
}

}  // namespace

const char* reference() { return LOKAHI_UNICODE_REFERENCE; }

Category category(char32_t cp) {
  if (cp >= 0x110000) return Cn;
  const auto* end = kCategoryRanges + sizeof(kCategoryRanges) / sizeof(kCategoryRanges[0]);
  const auto* it = std::upper_bound(kCategoryRanges, end, cp,
                                    [](char32_t value, const CategoryRange& r) { return value < r.first; });
  return static_cast<Category>((it - 1)->category);
}

bool is_space(char32_t cp) {
  if (cp <= 0x20) return cp == 0x20 || (cp >= 0x09 && cp <= 0x0D);
  if (cp == 0x85) return true;
  if (cp < 0xA0) return false;
  return in(cp, kSeparator);
}

size_t utf8_valid_prefix(std::string_view text, size_t i, size_t* needed) {
  const auto byte = [&](size_t k) { return static_cast<unsigned char>(text[i + k]); };
  const unsigned char lead = byte(0);
  *needed = 1;
  if (lead < 0x80) return 1;
  // Well-formed sequences (Unicode Table 3-7): the second byte's range
  // depends on the lead byte, which excludes overlongs, surrogates and
  // values above U+10FFFF.
  unsigned char lo = 0x80, hi = 0xBF;
  if (lead >= 0xC2 && lead <= 0xDF) {
    *needed = 2;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    *needed = 3;
    if (lead == 0xE0) lo = 0xA0;
    if (lead == 0xED) hi = 0x9F;
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    *needed = 4;
    if (lead == 0xF0) lo = 0x90;
    if (lead == 0xF4) hi = 0x8F;
  } else {
    return 0;
  }
  if (i + 1 >= text.size() || byte(1) < lo || byte(1) > hi) return 1;
  size_t k = 2;
  while (k < *needed && i + k < text.size() && (byte(k) & 0xC0) == 0x80) ++k;
  return k;
}

size_t utf8_sequence_length(std::string_view text, size_t i) {
  size_t needed;
  const size_t valid = utf8_valid_prefix(text, i, &needed);
  return valid == needed ? valid : 0;
}

char32_t decode_utf8_at(std::string_view text, size_t i, size_t* length) {
  const size_t len = utf8_sequence_length(text, i);
  LK_CHECK(len != 0, "unicode: input is not valid UTF-8");
  const auto lead = static_cast<unsigned char>(text[i]);
  char32_t cp = len == 1 ? lead : len == 2 ? (lead & 0x1F) : len == 3 ? (lead & 0x0F) : (lead & 0x07);
  for (size_t k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
  *length = len;
  return cp;
}

std::u32string decode_utf8(std::string_view text) {
  std::u32string out;
  out.reserve(text.size());
  for (size_t i = 0, len = 0; i < text.size(); i += len) out.push_back(decode_utf8_at(text, i, &len));
  return out;
}

void append_utf8(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string encode_utf8(std::u32string_view text) {
  std::string out;
  out.reserve(text.size());
  for (char32_t cp : text) append_utf8(out, cp);
  return out;
}

std::string nfc(std::string_view text) {
  // Everything below U+0300 is already composed and has no combining marks.
  bool trivial = true;
  for (char c : text) {
    if (static_cast<unsigned char>(c) >= 0xCC) {  // lead bytes of U+0300 and above
      trivial = false;
      break;
    }
  }
  if (trivial) return std::string(text);

  std::u32string decomposed;
  for (char32_t cp : decode_utf8(text)) decompose(cp, decomposed);

  // Canonical ordering: stable sort each run of non-starters by class.
  for (size_t i = 0; i < decomposed.size();) {
    if (combining_class(decomposed[i]) == 0) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < decomposed.size() && combining_class(decomposed[j]) != 0) ++j;
    std::stable_sort(decomposed.begin() + static_cast<long>(i), decomposed.begin() + static_cast<long>(j),
                     [](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
    i = j;
  }

  // Canonical composition: a mark composes with the last starter unless a
  // character in between has class 0 or a class not lower than its own.
  std::u32string out;
  out.reserve(decomposed.size());
  long starter = -1;
  uint8_t last_class = 0;
  for (char32_t cp : decomposed) {
    const uint8_t cc = combining_class(cp);
    if (starter >= 0) {
      const bool adjacent = static_cast<long>(out.size()) - 1 == starter;
      if (adjacent || (last_class != 0 && last_class < cc)) {
        if (char32_t composite = compose(out[static_cast<size_t>(starter)], cp)) {
          out[static_cast<size_t>(starter)] = composite;
          continue;
        }
      }
    }
    if (cc == 0) starter = static_cast<long>(out.size());
    last_class = cc;
    out.push_back(cp);
  }
  return encode_utf8(out);
}

}  // namespace lokahi::unicode
