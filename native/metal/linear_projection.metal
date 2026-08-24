#include <metal_stdlib>

using namespace metal;

// Correctness-first direct-Metal implementation of the projection already
// qualified through Strata's bounded ANE worker:
//
//   output[t, o] = sum_k input[t, k] * weight[o, k]
//
// Logical shapes are input [64, 256], out_in weight [256, 256], and output
// [64, 256]. One GPU thread owns one output element and accumulates in float32
// before the final float16 cast. This is intentionally the simplest native
// baseline. Tiled/threadgroup and simdgroup variants must beat this complete
// dispatch boundary while preserving the same contract.
kernel void linear_projection_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &token_count [[buffer(3)]],
    constant uint &input_width [[buffer(4)]],
    constant uint &output_width [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint output_column = position.x;
  const uint token = position.y;
  if (token >= token_count || output_column >= output_width) {
    return;
  }

  const uint input_offset = token * input_width;
  const uint weight_offset = output_column * input_width;
  float sum = 0.0f;
  for (uint column = 0; column < input_width; ++column) {
    sum = fma(float(input[input_offset + column]),
              float(weight[weight_offset + column]), sum);
  }
  output[token * output_width + output_column] = half(sum);
}

// A 16-output by 8-token tile cooperatively stages both activation and weight
// slices. Each activation value is reused by sixteen threads and each weight is
// reused by eight threads before the next K slice is loaded. This variant keeps
// the exact public contract above so the benchmark can reject it if barriers
// and staging cost more than the global-memory traffic they remove.
kernel void linear_projection_f16_tiled(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &token_count [[buffer(3)]],
    constant uint &input_width [[buffer(4)]],
    constant uint &output_width [[buffer(5)]],
    ushort2 thread_position [[thread_position_in_threadgroup]],
    ushort2 group_position [[threadgroup_position_in_grid]]) {
  constexpr uint tile_outputs = 16;
  constexpr uint tile_tokens = 8;
  constexpr uint tile_k = 32;
  threadgroup half staged_input[tile_tokens][tile_k];
  threadgroup half staged_weight[tile_outputs][tile_k];

  const uint local_output = thread_position.x;
  const uint local_token = thread_position.y;
  const uint output_column = group_position.x * tile_outputs + local_output;
  const uint token = group_position.y * tile_tokens + local_token;
  const uint linear_thread = local_token * tile_outputs + local_output;
  float sum = 0.0f;

  for (uint k_base = 0; k_base < input_width; k_base += tile_k) {
    for (uint index = linear_thread; index < tile_tokens * tile_k;
         index += tile_outputs * tile_tokens) {
      const uint staged_token = index / tile_k;
      const uint staged_k = index % tile_k;
      const uint global_token = group_position.y * tile_tokens + staged_token;
      staged_input[staged_token][staged_k] =
          global_token < token_count && k_base + staged_k < input_width
              ? input[global_token * input_width + k_base + staged_k]
              : half(0.0h);
    }
    for (uint index = linear_thread; index < tile_outputs * tile_k;
         index += tile_outputs * tile_tokens) {
      const uint staged_output = index / tile_k;
      const uint staged_k = index % tile_k;
      const uint global_output =
          group_position.x * tile_outputs + staged_output;
      staged_weight[staged_output][staged_k] =
          global_output < output_width && k_base + staged_k < input_width
              ? weight[global_output * input_width + k_base + staged_k]
              : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (token < token_count && output_column < output_width) {
      for (uint k = 0; k < tile_k; ++k) {
        sum = fma(float(staged_input[local_token][k]),
                  float(staged_weight[local_output][k]), sum);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (token < token_count && output_column < output_width) {
    output[token * output_width + output_column] = half(sum);
  }
}
