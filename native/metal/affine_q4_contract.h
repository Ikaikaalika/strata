#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

// Bounded operator contract, not a model loader. Compatible with MLX affine
// Q4: eight little-nibble-first codes per uint32, out_in row-major weights,
// one FP16 scale and additive bias per group of 64 input columns.
namespace strata::q4 {
using Half = _Float16;
constexpr uint32_t kGroupSize = 64;
constexpr size_t kMaxAllocationBytes = 64 * 1024 * 1024;

struct Shape {
  uint32_t rows;
  uint32_t input_width;
  uint32_t output_width;
};

struct Footprint {
  size_t input_elements;
  size_t packed_words;
  size_t group_elements;
  size_t output_elements;
  size_t total_bytes;
};

inline bool Validate(Shape shape, Footprint *footprint) {
  if (footprint == nullptr || shape.rows == 0 || shape.rows > 512 ||
      shape.input_width == 0 || shape.input_width > 8192 ||
      shape.input_width % kGroupSize != 0 || shape.output_width == 0 ||
      shape.output_width > 8192) {
    return false;
  }
  const size_t m = shape.rows, k = shape.input_width, n = shape.output_width;
  Footprint value{m * k, n * (k / 8), n * (k / kGroupSize), m * n, 0};
  value.total_bytes = sizeof(Half) * value.input_elements +
      sizeof(uint32_t) * value.packed_words +
      2 * sizeof(Half) * value.group_elements +
      sizeof(Half) * value.output_elements;
  if (value.total_bytes > kMaxAllocationBytes) return false;
  *footprint = value;
  return true;
}

inline uint32_t Code(uint32_t packed, uint32_t column) {
  return (packed >> (4 * (column % 8))) & 15u;
}

// FP64 accumulation is an independent arithmetic control for the FP32 GPU
// candidates. Callers own all storage. No tensor math occurs in the wrappers.
inline bool Reference(Shape shape, const Half *input, size_t input_count,
                      const uint32_t *packed, size_t packed_count,
                      const Half *scales, const Half *biases, size_t group_count,
                      float *output, size_t output_count) {
  Footprint sizes{};
  if (!Validate(shape, &sizes) || input == nullptr || packed == nullptr ||
      scales == nullptr || biases == nullptr || output == nullptr ||
      input_count != sizes.input_elements || packed_count != sizes.packed_words ||
      group_count != sizes.group_elements || output_count != sizes.output_elements) {
    return false;
  }
  for (size_t i = 0; i < input_count; ++i) {
    if (!std::isfinite(float(input[i]))) return false;
  }
  for (size_t i = 0; i < group_count; ++i) {
    if (!std::isfinite(float(scales[i])) || !std::isfinite(float(biases[i]))) return false;
  }
  for (uint32_t row = 0; row < shape.rows; ++row) {
    for (uint32_t out = 0; out < shape.output_width; ++out) {
      double sum = 0;
      for (uint32_t k = 0; k < shape.input_width; ++k) {
        const size_t group = size_t(out) * (shape.input_width / kGroupSize) + k / kGroupSize;
        const uint32_t code = Code(packed[size_t(out) * (shape.input_width / 8) + k / 8], k);
        const double weight = double(scales[group]) * code + double(biases[group]);
        sum += double(input[size_t(row) * shape.input_width + k]) * weight;
      }
      output[size_t(row) * shape.output_width + out] = float(sum);
      if (!std::isfinite(output[size_t(row) * shape.output_width + out])) return false;
    }
  }
  return true;
}
}  // namespace strata::q4
