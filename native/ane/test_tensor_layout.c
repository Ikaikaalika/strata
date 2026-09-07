#include "tensor_layout.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t dimensions[] = {1, 7, 8, 15, 16, 17, 64, 256, 512, 1024};
    const size_t tiles[] = {0, 8, 16};
    size_t cases = 0;
    for (size_t i=0; i<10; ++i) for (size_t j=0; j<10; ++j) {
        size_t rows=dimensions[i], cols=dimensions[j], count=rows*cols;
        uint16_t *src=malloc((count+2)*2), *dst=malloc((count+2)*2), *back=malloc((count+2)*2);
        if (!src || !dst || !back) return 1;
        for (size_t k=0; k<count; ++k) src[k+1]=(uint16_t)(k*730+17);
        for (size_t t=0; t<3; ++t) {
            src[0]=dst[0]=back[0]=0xbeef;
            src[count+1]=dst[count+1]=back[count+1]=0xdead;
            strata_transpose_fp16_bits(src+1,dst+1,rows,cols,tiles[t]);
            for (size_t r=0; r<rows; ++r) for (size_t c=0; c<cols; ++c)
                if (dst[1+c*rows+r]!=src[1+r*cols+c]) return 2;
            strata_transpose_fp16_bits(dst+1,back+1,cols,rows,tiles[t]);
            for (size_t k=0; k<count; ++k) if (src[k+1]!=back[k+1]) return 3;
            if (src[0]!=0xbeef || dst[0]!=0xbeef || back[0]!=0xbeef || src[count+1]!=0xdead || dst[count+1]!=0xdead || back[count+1]!=0xdead) return 4;
            ++cases;
        }
        free(src); free(dst); free(back);
    }
    // Byte buffers need not have uint16_t alignment. Include incomplete vector
    // tiles and empty dimensions; every bit pattern (including NaNs) is data.
    const size_t edges[]={0,1,7,8,9,15,16,17,31,64};
    for(size_t i=0;i<10;++i) for(size_t j=0;j<10;++j) {
        size_t rows=edges[i],cols=edges[j],bytes=2*rows*cols;
        unsigned char *src=malloc(bytes+2), *dst=malloc(bytes+2);
        if(!src || !dst) return 5;
        for(size_t k=0;k<bytes+2;++k) src[k]=(unsigned char)(k*37+11);
        memset(dst,0xa5,bytes+2);
        strata_transpose_fp16_bits(src+1,dst+1,rows,cols,8);
        for(size_t r=0;r<rows;++r) for(size_t c=0;c<cols;++c)
            if(memcmp(dst+1+2*(c*rows+r),src+1+2*(r*cols+c),2)) return 6;
        if(dst[0]!=0xa5 || dst[bytes+1]!=0xa5) return 7;
        free(src);free(dst);++cases;
    }
    printf("%zu transpose shape/tile/alignment cases passed\n",cases);
    return 0;
}
