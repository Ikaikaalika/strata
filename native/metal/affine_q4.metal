#include <metal_stdlib>
using namespace metal;

// The host validates K % 64 == 0 and every buffer size before dispatch.
// Packed weights stay packed in unified memory; no full FP16 shadow is made.
inline float q4_value(device const uint *packed, device const half *scales,
                      device const half *biases, uint out, uint k, uint width) {
  const uint word = packed[out * (width / 8) + k / 8];
  const uint code = (word >> (4 * (k % 8))) & 15u;
  const uint group = out * (width / 64) + k / 64;
  return float(scales[group]) * float(code) + float(biases[group]);
}

kernel void affine_q4_scalar(
    device const half *input [[buffer(0)]], device const uint *packed [[buffer(1)]],
    device const half *scales [[buffer(2)]], device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &width [[buffer(6)]], constant uint &outputs [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]) {
  if (pos.x >= outputs || pos.y >= rows) return;
  float sum = 0;
  for (uint k = 0; k < width; ++k)
    sum = fma(float(input[pos.y * width + k]),
              q4_value(packed, scales, biases, pos.x, k, width), sum);
  output[pos.y * outputs + pos.x] = half(sum);
}

// Decode candidate: one SIMD group cooperates on an output dot product.
// Each lane loads one packed word and consumes all eight codes before moving
// on. Scale/bias loads are reused for those eight multiplies.
kernel void affine_q4_simd(
    device const half *input [[buffer(0)]], device const uint *packed [[buffer(1)]],
    device const half *scales [[buffer(2)]], device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &width [[buffer(6)]], constant uint &outputs [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint lanes [[threads_per_simdgroup]]) {
  const uint out = group.x, row = group.y;
  if (out >= outputs || row >= rows) return;
  float sum = 0;
  for (uint word_index = lane; word_index < width / 8; word_index += lanes) {
    const uint word = packed[out * (width / 8) + word_index];
    const uint g = out * (width / 64) + word_index / 8;
    const float scale = float(scales[g]), bias = float(biases[g]);
    for (uint bit = 0; bit < 8; ++bit) {
      const float w = scale * float((word >> (bit * 4)) & 15u) + bias;
      sum = fma(float(input[row * width + word_index * 8 + bit]), w, sum);
    }
  }
  const float total = simd_sum(sum);
  if (lane == 0) output[row * outputs + out] = half(total);
}

// Prefill candidate: 8 token rows share decoded weights and 16 output columns
// share activations. Decode into bounded FP32 threadgroup scratch to preserve
// the affine formula; never requantize or expand a complete weight matrix.
kernel void affine_q4_tiled(
    device const half *input [[buffer(0)]], device const uint *packed [[buffer(1)]],
    device const half *scales [[buffer(2)]], device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &width [[buffer(6)]], constant uint &outputs [[buffer(7)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup half x[8][32];
  threadgroup float w[16][32];
  const uint out = group.x * 16 + local.x, row = group.y * 8 + local.y;
  const uint linear_thread = local.y * 16 + local.x;
  float sum = 0;
  for (uint base = 0; base < width; base += 32) {
    for (uint i = linear_thread; i < 8 * 32; i += 128) {
      const uint r = group.y * 8 + i / 32;
      x[i / 32][i % 32] = r < rows ? input[r * width + base + i % 32] : half(0);
    }
    for (uint i = linear_thread; i < 16 * 32; i += 128) {
      const uint o = group.x * 16 + i / 32;
      w[i / 32][i % 32] = o < outputs
          ? q4_value(packed, scales, biases, o, base + i % 32, width) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (out < outputs && row < rows)
      for (uint k = 0; k < 32; ++k) sum = fma(float(x[local.y][k]), w[local.x][k], sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (out < outputs && row < rows) output[row * outputs + out] = half(sum);
}
