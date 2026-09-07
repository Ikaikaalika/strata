// Generated dense prefill-shaped work, not a complete LLM or serving backend.
// Reuse Strata's reviewed private lifecycle/surface helpers, not third-party code.
#define main strata_original_probe_main
#include "strata_ane_probe.m"
#undef main
#include <mach/mach_time.h>

static double MonoMS(void) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1e6;
}

static void Require(BOOL ok, NSString *reason) {
    if (!ok) @throw [NSException exceptionWithName:@"StrataPrefillFailure"
        reason:reason ?: @"unknown failure" userInfo:nil];
}

static uint32_t NextRandom(uint32_t *state) {
    *state ^= *state << 13; *state ^= *state >> 17; *state ^= *state << 5;
    return *state;
}

static NSString *ChainMIL(NSUInteger tokens, NSUInteger first, NSUInteger count) {
    NSMutableString *s = [NSMutableString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n func main<ios16>(tensor<fp16, [1, 256, 1, %lu]> x) {\n",
        (unsigned long)tokens];
    [s appendString:
        @" tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        @" tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @" tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @" tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @" tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"];
    NSString *previous = @"x";
    for (NSUInteger i = first; i < first+count; i++) {
        [s appendFormat:@" tensor<fp16, [256, 256, 1, 1]> W%lu = const()[name = tensor<string, []>(\"W%lu\"), val = tensor<fp16, [256, 256, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w%lu.bin\"), offset = tensor<uint64, []>(64)))];\n", (unsigned long)i,(unsigned long)i,(unsigned long)i];
        [s appendFormat:@" tensor<fp16, [1, 256, 1, %lu]> c%lu = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W%lu, x = %@)[name = tensor<string, []>(\"conv_%lu\")];\n",(unsigned long)tokens,(unsigned long)i,(unsigned long)i,previous,(unsigned long)i];
        [s appendFormat:@" tensor<fp16, [1, 256, 1, %lu]> r%lu = relu(x = c%lu)[name = tensor<string, []>(\"relu_%lu\")];\n",(unsigned long)tokens,(unsigned long)i,(unsigned long)i,(unsigned long)i];
        previous = [NSString stringWithFormat:@"r%lu",(unsigned long)i];
    }
    [s appendFormat:@" } -> (%@);\n}\n", previous];
    return s;
}

static NSDictionary *RunChain(NSUInteger tokens, NSUInteger depth, BOOL fused, NSString *outputPath) {
    const NSUInteger width = 256, elements = tokens*width, bytes = elements*sizeof(_Float16);
    NSMutableDictionary *result = [@{@"schema_version":@1,
        @"benchmark":@"strata-direct-ane-prefill-chain",@"evidence_kind":@"hardware",
        @"comparison_scope":@"generated_dense_relu_chain",@"promotion_eligible":@NO,
        @"tokens":@(tokens),@"width":@(width),@"depth":@(depth),@"dtype":@"float16",
        @"variant":fused ? @"fused" : @"split",@"warmups":@5,@"iterations":@20,
        @"seed":@730,@"success":@NO,@"tolerance":@0.002,
        @"claim_boundary":@"Generated dense ReLU chain only; not a transformer, real model, full prefill phase or serving result"} mutableCopy];
    NSMutableArray *models = [NSMutableArray array], *requests = [NSMutableArray array];
    NSMutableArray *directories = [NSMutableArray array], *blobs = [NSMutableArray array];
    NSMutableArray *dispatchSamples = [NSMutableArray array], *wallSamples = [NSMutableArray array];
    IOSurfaceRef surfaces[2] = {NULL,NULL};
    NSMutableData *inputData = [NSMutableData dataWithLength:bytes];
    NSMutableData *referenceData = [NSMutableData dataWithLength:bytes];
    NSMutableData *scratchData = [NSMutableData dataWithLength:bytes];
    NSMutableData *readbackData = [NSMutableData dataWithLength:bytes];
    _Float16 *input = inputData.mutableBytes, *reference = referenceData.mutableBytes;
    _Float16 *scratch = scratchData.mutableBytes, *readback = readbackData.mutableBytes;
    NSMutableArray *allWeights = [NSMutableArray array];
    double maxError = 0, compileMS = 0, loadMS = 0;
    NSFileManager *fm = NSFileManager.defaultManager;
    @try {
        Require(![fm fileExistsAtPath:outputPath], @"output already exists");
        NSDictionary *discovery = BuildReport(NO,0,1);
        result[@"platform"] = discovery[@"platform"];
        Require([discovery[@"required_surface"][@"execution_surface_satisfied"] boolValue], @"private ANE surface unavailable");
        for (NSUInteger i=0;i<elements;i++) input[i]=(_Float16)(((int)((i*13)%67)-33)/64.0f);
        memcpy(reference,input,bytes);
        for (NSUInteger layer=0;layer<depth;layer++) {
            NSMutableData *weights=[NSMutableData dataWithLength:kWeightBytes];
            _Float16 *w=weights.mutableBytes;
            uint32_t state=730+(uint32_t)layer;
            for (NSUInteger i=0;i<width*width;i++) w[i]=(_Float16)(((int)(NextRandom(&state)%65)-32)/512.0f);
            [allWeights addObject:weights];
            [blobs addObject:BuildWeightBlobFromFP16(w)];
            // Independent scalar oracle, rounding at the FP16 layer boundaries.
            for (NSUInteger t=0;t<tokens;t++) for (NSUInteger o=0;o<width;o++) {
                double sum=0;
                for (NSUInteger k=0;k<width;k++) sum+=(double)reference[t*width+k]*(double)w[o*width+k];
                scratch[t*width+o]=(_Float16)fmax(0.0,sum);
            }
            memcpy(reference,scratch,bytes);
        }
        surfaces[0]=CreateTensorSurface(bytes); surfaces[1]=CreateTensorSurface(bytes);
        Require(surfaces[0] && surfaces[1],@"IOSurface allocation failed");
        Class dc=objc_getClass("_ANEInMemoryModelDescriptor"), mc=objc_getClass("_ANEInMemoryModel");
        Class rc=objc_getClass("_ANERequest"), sc=objc_getClass("_ANEIOSurfaceObject");
        NSUInteger modelCount=fused ? 1 : depth;
        for (NSUInteger j=0;j<modelCount;j++) {
            NSUInteger first=fused ? 0 : j, count=fused ? depth : 1;
            NSData *mil=[ChainMIL(tokens,first,count) dataUsingEncoding:NSUTF8StringEncoding];
            NSMutableDictionary *weights=[NSMutableDictionary dictionary];
            for (NSUInteger i=first;i<first+count;i++) weights[[NSString stringWithFormat:@"@model_path/weights/w%lu.bin",(unsigned long)i]]=@{@"offset":@0,@"data":blobs[i]};
            id descriptor=((id(*)(Class,SEL,id,id,id))objc_msgSend)(dc,sel_registerName("modelWithMILText:weights:optionsPlist:"),mil,weights,nil);
            Require(descriptor!=nil,@"descriptor failed");
            id model=((id(*)(Class,SEL,id))objc_msgSend)(mc,sel_registerName("inMemoryModelWithDescriptor:"),descriptor);
            Require(model!=nil,@"model creation failed");
            NSString *identifier=((id(*)(id,SEL))objc_msgSend)(model,sel_registerName("hexStringIdentifier"));
            NSCharacterSet *allowed=[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_."];
            Require([identifier isKindOfClass:NSString.class] && identifier.length>0 && identifier.length<=256 && ![identifier isEqualToString:@"."] && ![identifier isEqualToString:@".."] && [identifier rangeOfCharacterFromSet:allowed.invertedSet].location==NSNotFound,@"unsafe compiler identifier");
            NSString *dir=[NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
            Require(![fm fileExistsAtPath:dir],@"compiler directory already exists");
            NSError *error=nil;
            Require([fm createDirectoryAtPath:[dir stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:&error],error.localizedDescription);
            [directories addObject:dir];
            Require([mil writeToFile:[dir stringByAppendingPathComponent:@"model.mil"] atomically:YES],@"MIL write failed");
            for (NSUInteger i=first;i<first+count;i++) Require([blobs[i] writeToFile:[dir stringByAppendingPathComponent:[NSString stringWithFormat:@"weights/w%lu.bin",(unsigned long)i]] atomically:YES],@"weight write failed");
            double begin=MonoMS();
            BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model,sel_registerName("compileWithQoS:options:error:"),kANEQoS,@{},&error);
            compileMS+=MonoMS()-begin; Require(ok,error.localizedDescription ?: @"compile failed");
            begin=MonoMS();
            ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model,sel_registerName("loadWithQoS:options:error:"),kANEQoS,@{},&error);
            loadMS+=MonoMS()-begin; Require(ok,error.localizedDescription ?: @"load failed");
            [models addObject:model];
            NSUInteger inSlot=fused ? 0 : j%2, outSlot=1-inSlot;
            id inWrap=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(sc,sel_registerName("objectWithIOSurface:"),surfaces[inSlot]);
            id outWrap=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(sc,sel_registerName("objectWithIOSurface:"),surfaces[outSlot]);
            Require(inWrap && outWrap,@"surface wrap failed");
            id request=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(rc,sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),@[inWrap],@[@0],@[outWrap],@[@0],nil,nil,@0);
            Require(request!=nil,@"request creation failed");
            [requests addObject:request];
        }
        NSUInteger finalSlot=fused ? 1 : depth%2;
        for (NSUInteger iteration=0;iteration<25;iteration++) {
            double wallStart=MonoMS();
            Require(IOSurfaceLock(surfaces[0],0,NULL)==kIOReturnSuccess,@"input lock failed");
            _Float16 *physical=IOSurfaceGetBaseAddress(surfaces[0]);
            Require(physical!=NULL,@"null input surface");
            for (NSUInteger t=0;t<tokens;t++) for (NSUInteger c=0;c<width;c++) physical[c*tokens+t]=input[t*width+c];
            IOSurfaceUnlock(surfaces[0],0,NULL);
            double dispatchStart=MonoMS();
            for (NSUInteger j=0;j<models.count;j++) {
                NSError *error=nil;
                BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(models[j],sel_registerName("evaluateWithQoS:options:request:error:"),kANEQoS,@{},requests[j],&error);
                Require(ok,error.localizedDescription ?: @"evaluate failed");
            }
            double dispatchMS=MonoMS()-dispatchStart;
            Require(IOSurfaceLock(surfaces[finalSlot],kIOSurfaceLockReadOnly,NULL)==kIOReturnSuccess,@"readback lock failed");
            physical=IOSurfaceGetBaseAddress(surfaces[finalSlot]);
            Require(physical!=NULL,@"null output surface");
            for (NSUInteger t=0;t<tokens;t++) for (NSUInteger c=0;c<width;c++) readback[t*width+c]=physical[c*tokens+t];
            IOSurfaceUnlock(surfaces[finalSlot],kIOSurfaceLockReadOnly,NULL);
            double wallMS=MonoMS()-wallStart;
            for (NSUInteger i=0;i<elements;i++) {
                Require(isfinite((float)readback[i]),@"nonfinite output");
                maxError=fmax(maxError,fabs((double)readback[i]-(double)reference[i]));
            }
            Require(maxError<=0.002,@"independent oracle tolerance exceeded");
            if(iteration>=5) { [dispatchSamples addObject:@(dispatchMS)]; [wallSamples addObject:@(wallMS)]; }
        }
        NSError *error=nil;
        Require([readbackData writeToFile:outputPath options:NSDataWritingWithoutOverwriting error:&error],error.localizedDescription);
        result[@"success"]=@YES;
        result[@"verified_direct_ane_dispatches"]=@(25*models.count);
    } @catch(NSException *e) {
        result[@"error"]=e.reason ?: e.name;
    }
    BOOL cleanupOK=YES;
    for(id model in models) {
        NSError *error=nil;
        if(!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model,sel_registerName("unloadWithQoS:error:"),kANEQoS,&error)) cleanupOK=NO;
    }
    for(NSUInteger i=0;i<2;i++) if(surfaces[i]) CFRelease(surfaces[i]);
    for(NSString *dir in directories) if([fm fileExistsAtPath:dir] && ![fm removeItemAtPath:dir error:nil]) cleanupOK=NO;
    result[@"cleanup_ok"]=@(cleanupOK);
    if(!cleanupOK) result[@"success"]=@NO;
    result[@"compile_ms"]=@(compileMS); result[@"load_ms"]=@(loadMS);
    result[@"max_abs_error"]=@(maxError);
    result[@"dispatch_ms_samples"]=dispatchSamples;
    result[@"resident_with_io_ms_samples"]=wallSamples;
    result[@"timing_scope"]=@"resident input layout copy + serialized direct ANE evaluations + output layout/readback; excludes compile/load/oracle/file publication";
    result[@"thermal_state"]=@(NSProcessInfo.processInfo.thermalState);
    return result;
}

int main(int argc,const char *argv[]) {
    @autoreleasepool {
        if(argc!=5) { fprintf(stderr,"usage: prefill-chain-bench <64|256|512> <1|2|4> <split|fused> <new-output.bin>\n"); return 64; }
        char *end=NULL; unsigned long tokens=strtoul(argv[1],&end,10);
        if(!*argv[1] || *end || (tokens!=64 && tokens!=256 && tokens!=512)) return 64;
        unsigned long depth=strtoul(argv[2],&end,10);
        if(!*argv[2] || *end || (depth!=1 && depth!=2 && depth!=4)) return 64;
        if(strcmp(argv[3],"split") && strcmp(argv[3],"fused")) return 64;
        NSDictionary *result=RunChain(tokens,depth,strcmp(argv[3],"fused")==0,[NSString stringWithUTF8String:argv[4]]);
        NSData *json=[NSJSONSerialization dataWithJSONObject:result options:0 error:nil];
        fwrite(json.bytes,1,json.length,stdout); fputc('\n',stdout);
        return [result[@"success"] boolValue] ? 0 : 1;
    }
}
