// Minimal, strict JSON value and parser for model configs and file headers.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lokahi {

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Json() = default;
  static Json parse(std::string_view text);

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool() const;
  double as_number() const;
  int64_t as_int() const;
  const std::string& as_string() const;
  const std::vector<Json>& as_array() const;
  const std::map<std::string, Json>& as_object() const;

  // Object lookup; returns nullptr when absent or when this is not an object.
  const Json* find(const std::string& key) const;
  const Json& at(const std::string& key) const;
  bool contains(const std::string& key) const { return find(key) != nullptr; }

  double number_or(const std::string& key, double fallback) const;
  int64_t int_or(const std::string& key, int64_t fallback) const;
  std::string string_or(const std::string& key, const std::string& fallback) const;

 private:
  friend class JsonParser;
  Type type_ = Type::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Json> array_;
  std::map<std::string, Json> object_;
};

}  // namespace lokahi
