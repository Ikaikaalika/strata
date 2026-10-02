#include "affine_q4_contract.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
  using namespace strata::q4;
  Footprint sizes{};
  assert(!Validate({0, 64, 1}, &sizes));
  assert(!Validate({1, 65, 1}, &sizes));
  assert(!Validate({513, 64, 1}, &sizes));
  assert(!Validate({1, UINT32_MAX, 1}, &sizes));
  assert(!Validate({1, 64, UINT32_MAX}, &sizes));
  assert(!Validate({1, 64, 1}, nullptr));
  // Explicit nibble-order oracle, independent of the shader implementation.
  for (uint32_t k = 0; k < 8; ++k) assert(Code(0x76543210u, k) == k);
  size_t cases = 0;
  for (uint32_t rows : {1u, 7u, 8u, 9u, 64u}) {
    for (uint32_t width : {64u, 128u, 1024u}) {
      for (uint32_t outputs : {1u, 15u, 16u, 17u, 128u}) {
        const Shape shape{rows, width, outputs};
        assert(Validate(shape, &sizes));
        std::vector<Half> input(sizes.input_elements, Half(1));
        std::vector<uint32_t> packed(sizes.packed_words, 0x76543210u);
        std::vector<Half> scales(sizes.group_elements, Half(0.125));
        std::vector<Half> biases(sizes.group_elements, Half(-0.25));
        std::vector<float> result(sizes.output_elements);
        assert(Reference(shape, input.data(), input.size(), packed.data(), packed.size(),
                         scales.data(), biases.data(), scales.size(), result.data(), result.size()));
        // Sum of codes 0..7 is 28: each word contributes 28/8 - 8/4 = 1.5.
        for (float value : result) assert(value == float(width / 8) * 1.5f);
        assert(!Reference(shape, input.data(), input.size() - 1, packed.data(), packed.size(),
                          scales.data(), biases.data(), scales.size(), result.data(), result.size()));
        input[0] = Half(std::numeric_limits<float>::infinity());
        assert(!Reference(shape, input.data(), input.size(), packed.data(), packed.size(),
                          scales.data(), biases.data(), scales.size(), result.data(), result.size()));
        ++cases;
      }
    }
  }
  std::printf("{\"evidence_kind\":\"correctness\",\"cases\":%zu,\"passed\":true,"
              "\"promotion_eligible\":false}\n", cases);
}
