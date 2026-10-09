#include "json.h"

#include <cmath>
#include <cstdlib>

#include "common.h"

namespace lokahi {

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}

  Json parse_document() {
    Json value = parse_value(0);
    skip_ws();
    LK_CHECK(pos_ == text_.size(), "json: trailing characters");
    return value;
  }

 private:
  static constexpr int kMaxDepth = 256;

  void skip_ws() {
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
      ++pos_;
    }
  }

  char peek() {
    LK_CHECK(pos_ < text_.size(), "json: unexpected end of input");
    return text_[pos_];
  }

  void expect(char c) {
    LK_CHECK(peek() == c, std::string("json: expected '") + c + "'");
    ++pos_;
  }

  bool consume_literal(std::string_view literal) {
    if (text_.substr(pos_, literal.size()) == literal) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  Json parse_value(int depth) {
    LK_CHECK(depth < kMaxDepth, "json: nesting too deep");
    skip_ws();
    Json out;
    char c = peek();
    if (c == '{') {
      ++pos_;
      out.type_ = Json::Type::Object;
      skip_ws();
      if (peek() == '}') {
        ++pos_;
        return out;
      }
      while (true) {
        skip_ws();
        std::string key = parse_string();
        skip_ws();
        expect(':');
        out.object_[key] = parse_value(depth + 1);
        skip_ws();
        if (peek() == ',') {
          ++pos_;
          continue;
        }
        expect('}');
        return out;
      }
    }
    if (c == '[') {
      ++pos_;
      out.type_ = Json::Type::Array;
      skip_ws();
      if (peek() == ']') {
        ++pos_;
        return out;
      }
      while (true) {
        out.array_.push_back(parse_value(depth + 1));
        skip_ws();
        if (peek() == ',') {
          ++pos_;
          continue;
        }
        expect(']');
        return out;
      }
    }
    if (c == '"') {
      out.type_ = Json::Type::String;
      out.string_ = parse_string();
      return out;
    }
    if (consume_literal("true")) {
      out.type_ = Json::Type::Bool;
      out.bool_ = true;
      return out;
    }
    if (consume_literal("false")) {
      out.type_ = Json::Type::Bool;
      return out;
    }
    if (consume_literal("null")) return out;
    if (consume_literal("NaN")) {  // Emitted by some Python writers.
      out.type_ = Json::Type::Number;
      out.number_ = std::nan("");
      return out;
    }
    out.type_ = Json::Type::Number;
    out.number_ = parse_number();
    return out;
  }

  double parse_number() {
    size_t start = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
    if (consume_literal("Infinity")) return text_[start] == '-' ? -INFINITY : INFINITY;
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
        ++pos_;
      } else {
        break;
      }
    }
    LK_CHECK(pos_ > start, "json: invalid value");
    std::string token(text_.substr(start, pos_ - start));
    char* end = nullptr;
    double value = std::strtod(token.c_str(), &end);
    LK_CHECK(end == token.c_str() + token.size(), "json: invalid number '" + token + "'");
    return value;
  }

  static void append_utf8(std::string& out, uint32_t cp) {
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

  uint32_t parse_hex4() {
    LK_CHECK(pos_ + 4 <= text_.size(), "json: truncated \\u escape");
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      char c = text_[pos_++];
      value <<= 4;
      if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
      else fail("json: invalid \\u escape");
    }
    return value;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      LK_CHECK(pos_ < text_.size(), "json: unterminated string");
      char c = text_[pos_++];
      if (c == '"') return out;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      LK_CHECK(pos_ < text_.size(), "json: unterminated escape");
      char e = text_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t cp = parse_hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF && text_.substr(pos_, 2) == "\\u") {
            pos_ += 2;
            uint32_t low = parse_hex4();
            LK_CHECK(low >= 0xDC00 && low <= 0xDFFF, "json: invalid surrogate pair");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default:
          fail("json: invalid escape");
      }
    }
  }

  std::string_view text_;
  size_t pos_ = 0;
};

Json Json::parse(std::string_view text) { return JsonParser(text).parse_document(); }

bool Json::as_bool() const {
  LK_CHECK(type_ == Type::Bool, "json: expected bool");
  return bool_;
}

double Json::as_number() const {
  LK_CHECK(type_ == Type::Number, "json: expected number");
  return number_;
}

int64_t Json::as_int() const {
  double value = as_number();
  LK_CHECK(std::floor(value) == value, "json: expected integer");
  return static_cast<int64_t>(value);
}

const std::string& Json::as_string() const {
  LK_CHECK(type_ == Type::String, "json: expected string");
  return string_;
}

const std::vector<Json>& Json::as_array() const {
  LK_CHECK(type_ == Type::Array, "json: expected array");
  return array_;
}

const std::map<std::string, Json>& Json::as_object() const {
  LK_CHECK(type_ == Type::Object, "json: expected object");
  return object_;
}

const Json* Json::find(const std::string& key) const {
  if (type_ != Type::Object) return nullptr;
  auto it = object_.find(key);
  return it == object_.end() ? nullptr : &it->second;
}

const Json& Json::at(const std::string& key) const {
  const Json* value = find(key);
  LK_CHECK(value != nullptr, "json: missing key '" + key + "'");
  return *value;
}

double Json::number_or(const std::string& key, double fallback) const {
  const Json* value = find(key);
  return value && value->is_number() ? value->number_ : fallback;
}

int64_t Json::int_or(const std::string& key, int64_t fallback) const {
  const Json* value = find(key);
  return value && value->is_number() ? value->as_int() : fallback;
}

std::string Json::string_or(const std::string& key, const std::string& fallback) const {
  const Json* value = find(key);
  return value && value->is_string() ? value->string_ : fallback;
}

}  // namespace lokahi
