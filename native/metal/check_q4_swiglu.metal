#include "affine_q4_swiglu.metal"

// Standalone FP16 instantiations for warnings-as-errors compilation without
// loading a Metal device. MLX's BF16 type needs its separate execution gate.
kernel void check_swiglu_simd(
    device const half *x [[buffer(0)]], device const uint *gq [[buffer(1)]],
    device const half *gs [[buffer(2)]], device const half *gb [[buffer(3)]],
    device const uint *uq [[buffer(4)]], device const half *us [[buffer(5)]],
    device const half *ub [[buffer(6)]], device half *y [[buffer(7)]],
    constant uint *dims [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint lanes [[threads_per_simdgroup]]) {
  strata_swiglu::simd<half>(x, gq, gs, gb, uq, us, ub, y,
                          dims[0], dims[1], dims[2], group, lane, lanes);
}

kernel void check_swiglu_tiled(
    device const half *x [[buffer(0)]], device const uint *gq [[buffer(1)]],
    device const half *gs [[buffer(2)]], device const half *gb [[buffer(3)]],
    device const uint *uq [[buffer(4)]], device const half *us [[buffer(5)]],
    device const half *ub [[buffer(6)]], device half *y [[buffer(7)]],
    constant uint *dims [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[1280];
  strata_swiglu::tiled<half>(x, gq, gs, gb, uq, us, ub, y,
                           dims[0], dims[1], dims[2], group, local,
                           scratch, scratch + 256, scratch + 768);
}
