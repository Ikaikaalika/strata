#include <metal_stdlib>
using namespace metal;

// Experimental group64 affine Q4, no linear output bias. T is the unchanged
// model storage dtype. Packed weights stay resident; no concatenated copy or
// complete expanded shadow is allocated. Both dots accumulate in FP32.
namespace strata_swiglu {
template <typename T>
inline float weight(device const uint *q, device const T *s,
                    device const T *b, uint out, uint k, uint width) {
  const uint code = (q[out * (width / 8) + k / 8] >> (4 * (k % 8))) & 15u;
  const uint g = out * (width / 64) + k / 64;
  return float(s[g]) * float(code) + float(b[g]);
}

template <typename T>
inline T activate(float gate, float up) {
  // Preserve projection-output storage rounding; activation remains a new
  // candidate requiring MLX numerical and complete-model token/state gates.
  const float g = float(T(gate)), u = float(T(up));
  return T((g / (1.0f + exp(-g))) * u);
}

template <typename T>
inline void simd(device const T *x,
                 device const uint *gq, device const T *gs, device const T *gb,
                 device const uint *uq, device const T *us, device const T *ub,
                 device T *y, uint rows, uint width, uint outputs,
                 uint2 group, uint lane, uint lanes) {
  const uint out = group.x, row = group.y;
  if (out >= outputs || row >= rows) return;
  float gate = 0, up = 0;
  for (uint w = lane; w < width / 8; w += lanes) {
    const uint gi = out * (width / 64) + w / 8;
    const uint gate_word = gq[out * (width / 8) + w];
    const uint up_word = uq[out * (width / 8) + w];
    const float gate_scale = float(gs[gi]), gate_bias = float(gb[gi]);
    const float up_scale = float(us[gi]), up_bias = float(ub[gi]);
    for (uint bit = 0; bit < 8; ++bit) {
      const float a = float(x[row * width + w * 8 + bit]);
      gate = fma(a, gate_scale * float((gate_word >> (4 * bit)) & 15u) + gate_bias, gate);
      up = fma(a, up_scale * float((up_word >> (4 * bit)) & 15u) + up_bias, up);
    }
  }
  const float gate_sum = simd_sum(gate), up_sum = simd_sum(up);
  if (lane == 0) y[row * outputs + out] = activate<T>(gate_sum, up_sum);
}

template <typename T>
inline void tiled(device const T *x,
                  device const uint *gq, device const T *gs, device const T *gb,
                  device const uint *uq, device const T *us, device const T *ub,
                  device T *y, uint rows, uint width, uint outputs,
                  uint2 group, uint2 local, threadgroup float *activations,
                  threadgroup float *gate_weights, threadgroup float *up_weights) {
  const uint out = group.x * 16 + local.x, row = group.y * 8 + local.y;
  const uint tid = local.y * 16 + local.x;
  float gate = 0, up = 0;
  for (uint base = 0; base < width; base += 32) {
    for (uint i = tid; i < 8 * 32; i += 128) {
      const uint r = group.y * 8 + i / 32;
      activations[i] = r < rows ? float(x[r * width + base + i % 32]) : 0;
    }
    for (uint i = tid; i < 16 * 32; i += 128) {
      const uint o = group.x * 16 + i / 32, k = base + i % 32;
      gate_weights[i] = o < outputs ? weight<T>(gq, gs, gb, o, k, width) : 0;
      up_weights[i] = o < outputs ? weight<T>(uq, us, ub, o, k, width) : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (out < outputs && row < rows)
      for (uint k = 0; k < 32; ++k) {
        const float a = activations[local.y * 32 + k];
        gate = fma(a, gate_weights[local.x * 32 + k], gate);
        up = fma(a, up_weights[local.x * 32 + k], up);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (out < outputs && row < rows) y[row * outputs + out] = activate<T>(gate, up);
}
}  // namespace strata_swiglu
