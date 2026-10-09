// Shared error handling and small utilities for the Lokahi engine.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace lokahi {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

[[noreturn]] inline void fail(const std::string& message) { throw Error(message); }

#define LK_CHECK(cond, msg)                                        \
  do {                                                             \
    if (!(cond)) ::lokahi::fail(std::string("lokahi: ") + (msg)); \
  } while (0)

inline double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace lokahi
