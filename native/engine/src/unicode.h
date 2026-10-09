// Unicode properties and normalization used by the tokenizer. The tables
// (src/unicode_data.inc) are extracted from the reference tokenizer by
// tools/gen_unicode_tables.py so that regex classes and NFC agree with it
// code point for code point.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace lokahi::unicode {

// General categories, in the generator's order.
enum Category : uint8_t {
  Lu, Ll, Lt, Lm, Lo, Mn, Mc, Me, Nd, Nl, No,
  Pc, Pd, Ps, Pe, Pi, Pf, Po, Sm, Sc, Sk, So,
  Zs, Zl, Zp, Cc, Cf, Cs, Co, Cn,
};

constexpr uint32_t bit(Category c) { return 1u << c; }
constexpr uint32_t kLetter = bit(Lu) | bit(Ll) | bit(Lt) | bit(Lm) | bit(Lo);
constexpr uint32_t kMark = bit(Mn) | bit(Mc) | bit(Me);
constexpr uint32_t kNumber = bit(Nd) | bit(Nl) | bit(No);
constexpr uint32_t kPunctuation = bit(Pc) | bit(Pd) | bit(Ps) | bit(Pe) | bit(Pi) | bit(Pf) | bit(Po);
constexpr uint32_t kSymbol = bit(Sm) | bit(Sc) | bit(Sk) | bit(So);
constexpr uint32_t kSeparator = bit(Zs) | bit(Zl) | bit(Zp);
constexpr uint32_t kOther = bit(Cc) | bit(Cf) | bit(Cs) | bit(Co) | bit(Cn);

// The reference the tables were extracted from, e.g. "tokenizers 0.23.3".
const char* reference();
Category category(char32_t cp);
inline bool in(char32_t cp, uint32_t mask) { return (mask >> category(cp)) & 1u; }

// Oniguruma's \s for Unicode encodings: U+0009..U+000D, U+0085, and the
// Space_Separator, Line_Separator and Paragraph_Separator categories.
bool is_space(char32_t cp);

// How many bytes from text[i] form a well-formed UTF-8 sequence or a prefix
// of one, with the length the lead byte calls for in *needed. Returns 0 when
// text[i] cannot start a sequence.
size_t utf8_valid_prefix(std::string_view text, size_t i, size_t* needed);
// Length of the well-formed UTF-8 sequence starting at text[i], or 0 when it
// is malformed (truncated, overlong, a surrogate, or above U+10FFFF).
size_t utf8_sequence_length(std::string_view text, size_t i);
// Decodes the sequence at text[i] and stores its length; throws when it is
// malformed.
char32_t decode_utf8_at(std::string_view text, size_t i, size_t* length);
// Decodes well-formed UTF-8 (throws on malformed input).
std::u32string decode_utf8(std::string_view text);
void append_utf8(std::string& out, char32_t cp);
std::string encode_utf8(std::u32string_view text);

// Canonical composition (NFC).
std::string nfc(std::string_view text);

}  // namespace lokahi::unicode
