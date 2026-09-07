#ifndef STRATA_ANE_TENSOR_LAYOUT_H
#define STRATA_ANE_TENSOR_LAYOUT_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(STRATA_USE_NEON_TRANSPOSE) && defined(__aarch64__)
#include <arm_neon.h>
// Register-only 8x8 transpose; no FP16 arithmetic or conversion. Unaligned
// tensors are copied into vector values to preserve the byte-buffer contract.
static inline void strata_transpose8_neon(const unsigned char *src, unsigned char *dst,
                                         size_t input_stride, size_t output_stride) {
    uint16x8_t r[8];
    for (size_t i=0; i<8; ++i) memcpy(&r[i], src+2*i*input_stride, sizeof(r[i]));
    uint16x8x2_t a=vtrnq_u16(r[0],r[1]), b=vtrnq_u16(r[2],r[3]);
    uint16x8x2_t c=vtrnq_u16(r[4],r[5]), d=vtrnq_u16(r[6],r[7]);
    uint32x4x2_t e=vtrnq_u32(vreinterpretq_u32_u16(a.val[0]),vreinterpretq_u32_u16(b.val[0]));
    uint32x4x2_t f=vtrnq_u32(vreinterpretq_u32_u16(a.val[1]),vreinterpretq_u32_u16(b.val[1]));
    uint32x4x2_t g=vtrnq_u32(vreinterpretq_u32_u16(c.val[0]),vreinterpretq_u32_u16(d.val[0]));
    uint32x4x2_t h=vtrnq_u32(vreinterpretq_u32_u16(c.val[1]),vreinterpretq_u32_u16(d.val[1]));
    uint64x2_t top[4]={vreinterpretq_u64_u32(e.val[0]),vreinterpretq_u64_u32(f.val[0]),
                       vreinterpretq_u64_u32(e.val[1]),vreinterpretq_u64_u32(f.val[1])};
    uint64x2_t bottom[4]={vreinterpretq_u64_u32(g.val[0]),vreinterpretq_u64_u32(h.val[0]),
                          vreinterpretq_u64_u32(g.val[1]),vreinterpretq_u64_u32(h.val[1])};
    for(size_t i=0;i<4;++i) {
        uint64x2_t lo=vcombine_u64(vget_low_u64(top[i]),vget_low_u64(bottom[i]));
        uint64x2_t hi=vcombine_u64(vget_high_u64(top[i]),vget_high_u64(bottom[i]));
        memcpy(dst+2*i*output_stride,&lo,sizeof(lo));
        memcpy(dst+2*(i+4)*output_stride,&hi,sizeof(hi));
    }
}
#endif

// Out-of-place bit-preserving FP16 transpose. Callers own distinct buffers.
// A zero tile selects the original scalar traversal for paired experiments.
static inline void strata_transpose_fp16_bits(const void *input, void *output,
                                             size_t rows, size_t columns, size_t tile) {
    const unsigned char *src = (const unsigned char *)input;
    unsigned char *dst = (unsigned char *)output;
    if (tile == 0) {
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < columns; ++col)
                memcpy(dst + 2 * (col * rows + row), src + 2 * (row * columns + col), 2);
        return;
    }
    for (size_t r = 0; r < rows; r += tile) {
        size_t r_end = rows - r < tile ? rows : r + tile;
        for (size_t c = 0; c < columns; c += tile) {
            size_t c_end = columns - c < tile ? columns : c + tile;
#if defined(STRATA_USE_NEON_TRANSPOSE) && defined(__aarch64__)
            if (tile==8 && r_end-r==8 && c_end-c==8) {
                strata_transpose8_neon(src+2*(r*columns+c),dst+2*(c*rows+r),columns,rows);
                continue;
            }
#endif
            for (size_t row = r; row < r_end; ++row)
                for (size_t col = c; col < c_end; ++col)
                    memcpy(dst + 2 * (col * rows + row), src + 2 * (row * columns + col), 2);
        }
    }
}
#endif
