// Bounded hot-buffer CPU diagnostic. Not ANE/GPU or full-request throughput.
#include "tensor_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ns(void) {
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC_RAW,&t)) exit(2);
    return (double)t.tv_sec*1e9+(double)t.tv_nsec;
}
static int ascending(const void *a,const void *b) {
    double x=*(const double *)a,y=*(const double *)b;
    return (x>y)-(x<y);
}
static void roundtrip(const void *src,void *tmp,void *dst,size_t rows,size_t tile) {
    strata_transpose_fp16_bits(src,tmp,rows,1024,tile);
    strata_transpose_fp16_bits(tmp,dst,1024,rows,tile);
}
int main(void) {
#if !defined(STRATA_USE_NEON_TRANSPOSE) || !defined(__aarch64__)
    fputs("requires ARM64 and STRATA_USE_NEON_TRANSPOSE\n",stderr);return 64;
#else
    const size_t shapes[]={64,256,512},tiles[]={16,8};
    puts("{\"schema_version\":1,\"benchmark\":\"cpu-hot-buffer-roundtrip-layout\",\"promotion_eligible\":false,\"samples\":101,\"inner_iterations\":32,\"warmups_per_variant\":100,\"records\":[");
    for(size_t s=0;s<3;++s) {
        size_t rows=shapes[s],bytes=2*rows*1024;
        unsigned char *src=malloc(bytes),*tmp=malloc(bytes),*dst=malloc(bytes);
        if(!src || !tmp || !dst) return 1;
        for(size_t i=0;i<bytes;++i) src[i]=(unsigned char)(i*37+11);
        for(size_t v=0;v<2;++v) {
            for(size_t w=0;w<100;++w) roundtrip(src,tmp,dst,rows,tiles[v]);
            if(memcmp(src,dst,bytes)) return 3;
        }
        double samples[2][101];
        for(size_t sample=0;sample<101;++sample) for(size_t j=0;j<2;++j) {
            size_t v=(j+sample)%2;
            double start=now_ns();
            for(size_t k=0;k<32;++k) roundtrip(src,tmp,dst,rows,tiles[v]);
            samples[v][sample]=(now_ns()-start)/32/1000;
            // Verification outside timer ensures the compiler cannot drop work.
            if(memcmp(src,dst,bytes)) return 4;
        }
        printf("%s{\"tokens\":%zu,\"width\":1024,\"correctness\":true,\"variants\":[",s?",":"",rows);
        double medians[2];
        for(size_t v=0;v<2;++v) {
            double sorted[101];memcpy(sorted,samples[v],sizeof(sorted));
            qsort(sorted,101,sizeof(double),ascending);medians[v]=sorted[50];
            printf("%s{\"tile\":%zu,\"implementation\":\"%s\",\"median_us\":%.6f,\"raw_us\":[",v?",":"",tiles[v],v?"arm64-neon-8x8":"scalar-blocked-16",medians[v]);
            for(size_t k=0;k<101;++k) printf("%s%.6f",k?",":"",samples[v][k]);
            printf("]}");
        }
        printf("],\"control_over_neon\":%.6f}",medians[0]/medians[1]);
        free(src);free(tmp);free(dst);
    }
    puts("]}");return 0;
#endif
}
