// Bounded real-weight SwiGLU experiment, never a general-purpose model executor.
#define main strata_original_probe_main
#include "strata_ane_probe.m"
#undef main
#include <mach/mach_time.h>
#include "tensor_layout.h"

static void Check(BOOL ok, NSString *reason) {
    if (!ok) @throw [NSException exceptionWithName:@"StrataSwiGLUFailure" reason:reason ?: @"unknown failure" userInfo:nil];
}
static double ClockMS(void) {
    static mach_timebase_info_data_t info;
    if (!info.denom) mach_timebase_info(&info);
    return (double)mach_absolute_time()*info.numer/info.denom/1e6;
}
static NSData *ReadTensor(NSString *dir, NSString *name, NSUInteger bytes) {
    NSString *error=nil;
    NSData *data=ReadBoundedRegularFile([dir stringByAppendingPathComponent:name],bytes,bytes,&error);
    Check(data!=nil,error); Check(FP16DataIsFinite(data),@"nonfinite input or weights");
    return data;
}
static NSData *Blob(NSData *data) {
    NSMutableData *blob=[NSMutableData dataWithLength:128+data.length];
    uint8_t *p=blob.mutableBytes;
    p[0]=1; p[4]=2; p[64]=0xef; p[65]=0xbe; p[66]=0xad; p[67]=0xde; p[68]=1;
    uint32_t size=(uint32_t)data.length, offset=128;
    memcpy(p+72,&size,4); memcpy(p+80,&offset,4); memcpy(p+128,data.bytes,data.length);
    return blob;
}
static NSString *Program(NSUInteger tokens, BOOL tanhVariant) {
    NSMutableString *s=[NSMutableString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n func main<ios16>(tensor<fp16, [1, 1024, 1, %lu]> x) {\n",(unsigned long)tokens];
    [s appendString:
        @" tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        @" tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @" tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @" tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @" tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"];
    for(NSString *name in @[@"gate",@"up",@"down"]) {
        BOOL down=[name isEqualToString:@"down"];
        NSUInteger out=down?1024:3072, in=down?3072:1024;
        [s appendFormat:@" tensor<fp16, [%lu, %lu, 1, 1]> %@W = const()[name = tensor<string, []>(\"%@W\"), val = tensor<fp16, [%lu, %lu, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/%@.bin\"), offset = tensor<uint64, []>(64)))];\n",out,in,name,name,out,in,name];
        if(!down) [s appendFormat:@" tensor<fp16, [1, 3072, 1, %lu]> %@ = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = %@W, x = x)[name = tensor<string, []>(\"%@\")];\n",tokens,name,name,name];
    }
    if(tanhVariant) {
        [s appendString:@" tensor<fp16, []> half = const()[name = tensor<string, []>(\"half\"), val = tensor<fp16, []>(0.5)];\n tensor<fp16, []> one = const()[name = tensor<string, []>(\"one\"), val = tensor<fp16, []>(1.0)];\n"];
        [s appendFormat:@" tensor<fp16, [1, 3072, 1, %lu]> scaled = mul(x = gate, y = half)[name = tensor<string, []>(\"scaled\")];\n",tokens];
        [s appendFormat:@" tensor<fp16, [1, 3072, 1, %lu]> th = tanh(x = scaled)[name = tensor<string, []>(\"th\")];\n",tokens];
        [s appendFormat:@" tensor<fp16, [1, 3072, 1, %lu]> shifted = add(x = th, y = one)[name = tensor<string, []>(\"shifted\")];\n",tokens];
        [s appendFormat:@" tensor<fp16, [1, 3072, 1, %lu]> sig = mul(x = shifted, y = half)[name = tensor<string, []>(\"sig\")];\n",tokens];
    } else {
    [s appendFormat:@" tensor<fp16, [1, 3072, 1, %lu]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sig\")];\n",tokens];
    }
    [s appendFormat:@" tensor<fp16, [1, 3072, 1, %lu]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"silu\")];\n",tokens];
    [s appendFormat:@" tensor<fp16, [1, 3072, 1, %lu]> gated = mul(x = silu, y = up)[name = tensor<string, []>(\"gated\")];\n",tokens];
    [s appendFormat:@" tensor<fp16, [1, 1024, 1, %lu]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = downW, x = gated)[name = tensor<string, []>(\"down\")];\n } -> (y);\n}\n",tokens];
    return s;
}
static NSDictionary *Run(NSUInteger tokens, NSString *dir, BOOL tanhVariant, NSUInteger tile) {
#if defined(STRATA_USE_NEON_TRANSPOSE) && defined(__aarch64__)
    NSString *layoutImplementation=tile==8 ? @"arm64-neon-8x8" : @"scalar-blocked";
#else
    NSString *layoutImplementation=@"scalar-blocked";
#endif
    const NSUInteger width=1024, elements=tokens*width, bytes=elements*2;
    NSMutableDictionary *r=[@{@"schema_version":@1,@"benchmark":@"direct-ane-qwen3-swiglu",@"success":@NO,
        @"promotion_eligible":@NO,@"layout_tile":@(tile),@"tokens":@(tokens),@"width":@1024,@"intermediate":@3072,
        @"layout_implementation":layoutImplementation,
        @"warmups":@5,@"iterations":@20,@"atol":@0.01,@"rtol":@0.02,
        @"claim_boundary":@"One real-weight FFN with captured activations, not full-model prefill or decode"} mutableCopy];
    IOSurfaceRef input=NULL,output=NULL;
    BOOL inLocked=NO,outLocked=NO,loaded=NO,ownsDir=NO;
    uint32_t outputLockFlags=0;
    id model=nil;
    NSString *compilerDir=nil;
    NSFileManager *fm=NSFileManager.defaultManager;
    NSMutableArray *dispatch=[NSMutableArray array],*wall=[NSMutableArray array];
    NSUInteger dispatches=0;
    @try {
        struct stat status;
        Check(dir.isAbsolutePath && lstat(dir.fileSystemRepresentation,&status)==0 && S_ISDIR(status.st_mode),@"fixture must be an absolute nonsymlink directory");
        NSString *outputPath=[dir stringByAppendingPathComponent:@"output.bin"];
        Check(lstat(outputPath.fileSystemRepresentation,&status)!=0 && errno==ENOENT,@"output already exists or invalid");
        NSData *x=ReadTensor(dir,@"input.bin",bytes), *reference=ReadTensor(dir,@"reference.bin",bytes);
        NSMutableData *readback=[NSMutableData dataWithLength:bytes];
        NSMutableDictionary *weights=[NSMutableDictionary dictionary];
        for(NSString *name in @[@"gate",@"up",@"down"]) {
            NSData *blob=Blob(ReadTensor(dir,[name stringByAppendingString:@".bin"],1024*3072*2));
            weights[[NSString stringWithFormat:@"@model_path/weights/%@.bin",name]]=@{@"offset":@0,@"data":blob};
        }
        NSDictionary *discovery=BuildReport(NO,0,1);
        r[@"platform"]=discovery[@"platform"];
        Check([discovery[@"required_surface"][@"execution_surface_satisfied"] boolValue],@"private ANE unavailable");
        r[@"activation_variant"]=tanhVariant?@"tanh":@"sigmoid";
        NSData *mil=[Program(tokens,tanhVariant) dataUsingEncoding:NSUTF8StringEncoding];
        id descriptor=((id(*)(Class,SEL,id,id,id))objc_msgSend)(objc_getClass("_ANEInMemoryModelDescriptor"),sel_registerName("modelWithMILText:weights:optionsPlist:"),mil,weights,nil);
        Check(descriptor!=nil,@"descriptor failed");
        model=((id(*)(Class,SEL,id))objc_msgSend)(objc_getClass("_ANEInMemoryModel"),sel_registerName("inMemoryModelWithDescriptor:"),descriptor);
        Check(model!=nil,@"model creation failed");
        NSString *identifier=((id(*)(id,SEL))objc_msgSend)(model,sel_registerName("hexStringIdentifier"));
        Check([identifier isKindOfClass:NSString.class] && identifier.length>0 && identifier.length<=256 && ![identifier isEqual:@"."] && ![identifier isEqual:@".."] && [identifier rangeOfCharacterFromSet:[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_."].invertedSet].location==NSNotFound,@"unsafe compiler identifier");
        compilerDir=[NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        Check(lstat(compilerDir.fileSystemRepresentation,&status)!=0 && errno==ENOENT,@"compiler path collision");
        Check(mkdir(compilerDir.fileSystemRepresentation,0700)==0,@"compiler directory create failed"); ownsDir=YES;
        NSError *error=nil;
        Check([fm createDirectoryAtPath:[compilerDir stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:NO attributes:nil error:&error],error.localizedDescription);
        Check([mil writeToFile:[compilerDir stringByAppendingPathComponent:@"model.mil"] atomically:YES],@"MIL write failed");
        for(NSString *name in @[@"gate",@"up",@"down"])
            Check([weights[[NSString stringWithFormat:@"@model_path/weights/%@.bin",name]][@"data"] writeToFile:[compilerDir stringByAppendingPathComponent:[NSString stringWithFormat:@"weights/%@.bin",name]] atomically:YES],@"weight write failed");
        double start=ClockMS();
        BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model,sel_registerName("compileWithQoS:options:error:"),kANEQoS,@{},&error);
        r[@"compile_ms"]=@(ClockMS()-start); Check(ok,error.localizedDescription ?: @"compile failed");
        start=ClockMS();
        loaded=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model,sel_registerName("loadWithQoS:options:error:"),kANEQoS,@{},&error);
        r[@"load_ms"]=@(ClockMS()-start); Check(loaded,error.localizedDescription ?: @"load failed");
        input=CreateTensorSurface(bytes); output=CreateTensorSurface(bytes); Check(input && output,@"surface allocation failed");
        Class sc=objc_getClass("_ANEIOSurfaceObject");
        id inWrap=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(sc,sel_registerName("objectWithIOSurface:"),input);
        id outWrap=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(sc,sel_registerName("objectWithIOSurface:"),output);
        Check(inWrap && outWrap,@"surface wrapper failed");
        id request=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(objc_getClass("_ANERequest"),sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),@[inWrap],@[@0],@[outWrap],@[@0],nil,nil,@0);
        Check(request!=nil,@"request creation failed");
        double maxError=0,maxScaled=0;
        for(NSUInteger iteration=0;iteration<25;iteration++) {
            // Poison outputs outside the timing boundary to detect missing writes.
            outputLockFlags=0;
            Check(IOSurfaceLock(output,0,NULL)==kIOReturnSuccess,@"poison lock failed"); outLocked=YES;
            void *outAddress=IOSurfaceGetBaseAddress(output); Check(outAddress!=NULL,@"null output");
            memset(outAddress,0xff,bytes);
            Check(IOSurfaceUnlock(output,0,NULL)==kIOReturnSuccess,@"poison unlock failed"); outLocked=NO;
            double wallStart=ClockMS();
            Check(IOSurfaceLock(input,0,NULL)==kIOReturnSuccess,@"input lock failed"); inLocked=YES;
            _Float16 *dst=IOSurfaceGetBaseAddress(input); const _Float16 *src=x.bytes;
            Check(dst!=NULL,@"null input");
            if(tile==0) {
                // Keep the pre-optimization scalar traversal unchanged.
                for(NSUInteger c=0;c<width;c++) for(NSUInteger t=0;t<tokens;t++) dst[c*tokens+t]=src[t*width+c];
            } else strata_transpose_fp16_bits(src,dst,tokens,width,tile);
            Check(IOSurfaceUnlock(input,0,NULL)==kIOReturnSuccess,@"input unlock failed"); inLocked=NO;
            start=ClockMS();
            ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(model,sel_registerName("evaluateWithQoS:options:request:error:"),kANEQoS,@{},request,&error);
            double elapsed=ClockMS()-start; Check(ok,error.localizedDescription ?: @"dispatch failed"); dispatches++;
            outputLockFlags=kIOSurfaceLockReadOnly;
            Check(IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL)==kIOReturnSuccess,@"output lock failed"); outLocked=YES;
            src=IOSurfaceGetBaseAddress(output); dst=readback.mutableBytes; Check(src!=NULL,@"null output");
            if(tile==0) {
                for(NSUInteger t=0;t<tokens;t++) for(NSUInteger c=0;c<width;c++) dst[t*width+c]=src[c*tokens+t];
            } else strata_transpose_fp16_bits(src,dst,width,tokens,tile);
            Check(IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL)==kIOReturnSuccess,@"output unlock failed"); outLocked=NO;
            double wallMS=ClockMS()-wallStart;
            const _Float16 *expected=reference.bytes;
            for(NSUInteger i=0;i<elements;i++) {
                Check(isfinite((float)dst[i]),@"nonfinite or unwritten output");
                double e=fabs((double)dst[i]-(double)expected[i]);
                maxError=fmax(maxError,e); maxScaled=fmax(maxScaled,e/(0.01+0.02*fabs((double)expected[i])));
            }
            r[@"max_abs_error"]=@(maxError); r[@"max_scaled_error"]=@(maxScaled);
            if(maxScaled>1) {
                double squared=0,refSquared=0; NSUInteger worst=0; double worstScaled=0;
                for(NSUInteger i=0;i<elements;i++) {
                    double delta=(double)dst[i]-(double)expected[i];
                    squared+=delta*delta; refSquared+=(double)expected[i]*(double)expected[i];
                    double scaled=fabs(delta)/(0.01+0.02*fabs((double)expected[i]));
                    if(scaled>worstScaled) { worst=i; worstScaled=scaled; }
                }
                r[@"relative_l2"]=@(sqrt(squared/fmax(refSquared,1e-24)));
                r[@"worst_element"]=@{@"index":@(worst),@"actual":@((float)dst[worst]),@"expected":@((float)expected[worst])};
            }
            Check(maxScaled<=1,@"MLX FP16 reference tolerance exceeded");
            if(iteration>=5) { [dispatch addObject:@(elapsed)]; [wall addObject:@(wallMS)]; }
        }
        Check([readback writeToFile:outputPath options:NSDataWritingWithoutOverwriting error:&error],error.localizedDescription);
        r[@"success"]=@YES;
    } @catch(NSException *e) { r[@"error"]=e.reason ?: e.name; }
    BOOL cleanup=YES;
    if(inLocked) IOSurfaceUnlock(input,0,NULL);
    if(outLocked) IOSurfaceUnlock(output,outputLockFlags,NULL);
    if(loaded && !((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model,sel_registerName("unloadWithQoS:error:"),kANEQoS,NULL)) cleanup=NO;
    if(input) CFRelease(input); if(output) CFRelease(output);
    if(ownsDir && [fm fileExistsAtPath:compilerDir] && ![fm removeItemAtPath:compilerDir error:nil]) cleanup=NO;
    r[@"cleanup_ok"]=@(cleanup); if(!cleanup) r[@"success"]=@NO;
    r[@"verified_direct_ane_dispatches"]=@(dispatches);
    r[@"dispatch_ms_samples"]=dispatch; r[@"resident_with_io_ms_samples"]=wall;
    r[@"thermal_state"]=@(NSProcessInfo.processInfo.thermalState);
    return r;
}
int main(int argc,const char *argv[]) {
    @autoreleasepool {
        if(argc!=5) { fprintf(stderr,"usage: swiglu-bench <64|256|512> <fixture-directory> <sigmoid|tanh> <0|8|16>\n"); return 64; }
        char *end=NULL; unsigned long tokens=strtoul(argv[1],&end,10);
        if(!*argv[1] || *end || (tokens!=64 && tokens!=256 && tokens!=512)) return 64;
        if(strcmp(argv[3],"sigmoid") && strcmp(argv[3],"tanh")) return 64;
        unsigned long tile=strtoul(argv[4],&end,10);
        if(!*argv[4] || *end || (tile!=0 && tile!=8 && tile!=16)) return 64;
        NSDictionary *r=Run(tokens,[NSString stringWithUTF8String:argv[2]],strcmp(argv[3],"tanh")==0,tile);
        NSData *data=[NSJSONSerialization dataWithJSONObject:r options:0 error:nil];
        fwrite(data.bytes,1,data.length,stdout); fputc('\n',stdout);
        return [r[@"success"] boolValue]?0:1;
    }
}
