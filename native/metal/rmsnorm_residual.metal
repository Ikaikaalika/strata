#include <metal_stdlib>

using namespace metal;

// Correctness-first fused RMSNorm + residual probe.
//
// One GPU thread owns one row.  This deliberately avoids a reduction tuned to
// one Apple GPU family: the first milestone is proving native Metal dispatch,
// synchronization, and numerical agreement through Strata's adapter.  A later
// kernel can replace the serial row reduction without changing the probe's JSON
// evidence contract.
kernel void rmsnorm_residual_f32(
    device const float *input [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &hidden_size [[buffer(4)]],
    constant float &epsilon [[buffer(5)]],
    constant uint &row_count [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= row_count) {
    return;
  }

  const uint offset = row * hidden_size;
  float sum_squares = 0.0f;
  for (uint column = 0; column < hidden_size; ++column) {
    const float value = input[offset + column];
    sum_squares += value * value;
  }

  const float inverse_rms = rsqrt(sum_squares / float(hidden_size) + epsilon);
  for (uint column = 0; column < hidden_size; ++column) {
    const uint index = offset + column;
    output[index] = input[index] * inverse_rms * weight[column] + residual[index];
  }
}
