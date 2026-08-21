#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <dlfcn.h>
#include <math.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <sys/sysctl.h>
#include <unistd.h>

static NSString *const kProbeName = @"strata-ane-capability";
static const NSInteger kSchemaVersion = 2;
// These are the smallest shape used by the primary-source in-memory benchmark;
// smaller shapes are not assumed portable across ANE generations.
static const NSUInteger kProjectionChannels = 256;
static const NSUInteger kProjectionSpatial = 64;
static const double kNumericTolerance = 0.002;
static const unsigned int kANEQoS = 21;

static NSString *SysctlString(const char *name) {
    size_t size = 0;
    if (sysctlbyname(name, NULL, &size, NULL, 0) != 0 || size == 0) {
        return @"";
    }

    char *buffer = calloc(size, sizeof(char));
    if (buffer == NULL) {
        return @"";
    }
    if (sysctlbyname(name, buffer, &size, NULL, 0) != 0) {
        free(buffer);
        return @"";
    }
    NSString *value = [NSString stringWithUTF8String:buffer] ?: @"";
    free(buffer);
    return value;
}

static NSString *ArchitectureName(void) {
#if defined(__arm64__)
    return @"arm64";
#elif defined(__x86_64__)
    return @"x86_64";
#else
    return @"unknown";
#endif
}

static NSArray<NSString *> *MethodNames(Class cls) {
    if (cls == Nil) {
        return @[];
    }

    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int index = 0; index < count; index++) {
        SEL selector = method_getName(methods[index]);
        const char *name = sel_getName(selector);
        if (name != NULL) {
            [names addObject:[NSString stringWithUTF8String:name]];
        }
    }
    free(methods);
    [names sortUsingSelector:@selector(compare:)];
    return names;
}

static NSDictionary<NSString *, id> *InspectClass(NSString *name) {
    Class cls = objc_getClass(name.UTF8String);
    if (cls == Nil) {
        return @{
            @"present": @NO,
            @"instance_methods": @[],
            @"class_methods": @[],
        };
    }

    return @{
        @"present": @YES,
        @"instance_methods": MethodNames(cls),
        @"class_methods": MethodNames(object_getClass(cls)),
    };
}

static BOOL HasInstanceMethod(NSString *className, NSString *selectorName) {
    Class cls = objc_getClass(className.UTF8String);
    if (cls == Nil) {
        return NO;
    }
    SEL selector = sel_registerName(selectorName.UTF8String);
    return class_getInstanceMethod(cls, selector) != NULL;
}

static BOOL HasClassMethod(NSString *className, NSString *selectorName) {
    Class cls = objc_getClass(className.UTF8String);
    if (cls == Nil) {
        return NO;
    }
    SEL selector = sel_registerName(selectorName.UTF8String);
    return class_getClassMethod(cls, selector) != NULL;
}

static NSDictionary<NSString *, id> *LoadFramework(
    NSString *name,
    NSString *path,
    NSMutableArray<NSString *> *errors
) {
    dlerror();
    void *handle = dlopen(path.fileSystemRepresentation, RTLD_LAZY | RTLD_LOCAL);
    const char *loadError = handle == NULL ? dlerror() : NULL;
    NSString *error = loadError == NULL ? @"" : [NSString stringWithUTF8String:loadError];
    if (handle == NULL) {
        [errors addObject:[NSString stringWithFormat:@"failed to load %@: %@", name, error]];
    }

    return @{
        @"path": path,
        @"loaded": @((BOOL)(handle != NULL)),
        @"error": error,
    };
}

static NSMutableData *BuildWeightBlob(float *referenceWeights) {
    const NSUInteger elements = kProjectionChannels * kProjectionChannels;
    const NSUInteger headerBytes = 128;
    NSMutableData *blob = [NSMutableData dataWithLength:headerBytes + elements * sizeof(_Float16)];
    uint8_t *bytes = blob.mutableBytes;

    // Core ML weight-blob file header followed by one fp16 data chunk.
    bytes[0] = 0x01;
    bytes[4] = 0x02;
    uint8_t *chunk = bytes + 64;
    chunk[0] = 0xEF;
    chunk[1] = 0xBE;
    chunk[2] = 0xAD;
    chunk[3] = 0xDE;
    chunk[4] = 0x01;
    uint32_t weightBytes = (uint32_t)(elements * sizeof(_Float16));
    uint32_t dataOffset = 128;
    memcpy(chunk + 8, &weightBytes, sizeof(weightBytes));
    memcpy(chunk + 16, &dataOffset, sizeof(dataOffset));

    _Float16 *weights = (_Float16 *)(chunk + 64);
    for (NSUInteger output = 0; output < kProjectionChannels; output++) {
        for (NSUInteger input = 0; input < kProjectionChannels; input++) {
            float value = 0.0f;
            if (input == output) {
                value = 0.5f;
            } else if (input == ((output + 1) % kProjectionChannels)) {
                value = 0.25f;
            }
            _Float16 halfValue = (_Float16)value;
            NSUInteger index = output * kProjectionChannels + input;
            weights[index] = halfValue;
            referenceWeights[index] = (float)halfValue;
        }
    }
    return blob;
}

static NSString *BuildProjectionMIL(void) {
    NSString *nonce = [NSString stringWithFormat:@"strata_probe_%d", getpid()];
    NSMutableString *mil = [NSMutableString string];
    [mil appendString:
        @"program(1.0)\n"
        @"[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        @"{\n"];
    [mil appendFormat:@" func main<ios16>(tensor<fp16, [1, %lu, 1, %lu]> x) {\n",
        (unsigned long)kProjectionChannels, (unsigned long)kProjectionSpatial];
    [mil appendFormat:
        @"  tensor<fp16, [%lu, %lu, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), val = tensor<fp16, [%lu, %lu, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>(64)))];\n",
        (unsigned long)kProjectionChannels, (unsigned long)kProjectionChannels,
        (unsigned long)kProjectionChannels, (unsigned long)kProjectionChannels];
    [mil appendString:
        @"  tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        @"  tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"  tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"  tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"  tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"];
    [mil appendFormat:
        @"  tensor<fp16, [1, %lu, 1, %lu]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = tensor<string, []>(\"%@\")];\n",
        (unsigned long)kProjectionChannels, (unsigned long)kProjectionSpatial, nonce];
    [mil appendString:@" } -> (y);\n}\n"];
    return mil;
}

static IOSurfaceRef CreateTensorSurface(NSUInteger byteCount) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(byteCount),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(byteCount),
        (id)kIOSurfaceAllocSize: @(byteCount),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static void RecordFailure(
    NSMutableDictionary<NSString *, id> *execution,
    NSString *stage,
    NSString *reason
) {
    execution[@"failure_stage"] = stage ?: @"unknown";
    execution[@"error"] = reason ?: @"unknown private ANE failure";
}

static NSString *ErrorDescription(NSError *error, NSString *fallback) {
    return error == nil ? fallback : error.description;
}

static NSMutableDictionary<NSString *, id> *InitialExecutionEvidence(BOOL requested) {
    return [@{
        @"requested": @(requested),
        @"operation": @"fp16_projection",
        @"shape": @[@1, @(kProjectionChannels), @1, @(kProjectionSpatial)],
        @"input_dtype": @"float16",
        @"compute_dtype": @"float16",
        @"output_dtype": @"float16",
        @"tolerance": @(kNumericTolerance),
        @"last_successful_stage": @"discovery",
        @"failure_stage": @"",
        @"error": @"",
        @"mil_generated": @NO,
        @"descriptor_created": @NO,
        @"model_created": @NO,
        @"artifacts_written": @NO,
        @"compile_attempted": @NO,
        @"compile_succeeded": @NO,
        @"load_attempted": @NO,
        @"load_succeeded": @NO,
        @"iosurfaces_created": @NO,
        @"request_created": @NO,
        @"dispatch_attempted": @NO,
        @"dispatch_succeeded": @NO,
        @"output_finite": @NO,
        @"numeric_verified": @NO,
        @"execution_verified": @NO,
        @"max_abs_error": [NSNull null],
        @"dispatch_ms": [NSNull null],
        @"cleanup_error": @"",
    } mutableCopy];
}

static NSDictionary<NSString *, id> *RunProjectionProof(BOOL executionSurfaceSatisfied) {
    NSMutableDictionary<NSString *, id> *evidence = InitialExecutionEvidence(YES);
    if (!executionSurfaceSatisfied) {
        RecordFailure(evidence, @"execution_surface", @"required request or IOSurface selector is missing");
        return evidence;
    }
    evidence[@"last_successful_stage"] = @"surface_validated";

    Class descriptorClass = objc_getClass("_ANEInMemoryModelDescriptor");
    Class modelClass = objc_getClass("_ANEInMemoryModel");
    Class requestClass = objc_getClass("_ANERequest");
    Class surfaceObjectClass = objc_getClass("_ANEIOSurfaceObject");
    const NSUInteger tensorElements = kProjectionChannels * kProjectionSpatial;
    const NSUInteger tensorBytes = tensorElements * sizeof(_Float16);

    float *referenceWeights = calloc(kProjectionChannels * kProjectionChannels, sizeof(float));
    float *referenceOutput = calloc(tensorElements, sizeof(float));
    IOSurfaceRef inputSurface = NULL;
    IOSurfaceRef outputSurface = NULL;
    id model = nil;
    BOOL modelLoaded = NO;
    BOOL createdTemporaryDirectory = NO;
    NSString *temporaryDirectory = nil;
    NSFileManager *fileManager = NSFileManager.defaultManager;

    if (referenceWeights == NULL || referenceOutput == NULL) {
        RecordFailure(evidence, @"allocation", @"failed to allocate CPU reference buffers");
        goto cleanup;
    }

    @try {
        NSString *milText = BuildProjectionMIL();
        NSData *milData = [milText dataUsingEncoding:NSUTF8StringEncoding];
        NSMutableData *weightBlob = BuildWeightBlob(referenceWeights);
        if (milData == nil || weightBlob == nil) {
            RecordFailure(evidence, @"mil_generation", @"failed to generate MIL or weight data");
            goto cleanup;
        }
        evidence[@"mil_generated"] = @YES;
        evidence[@"last_successful_stage"] = @"mil_generated";

        NSDictionary *weightDictionary = @{
            @"@model_path/weights/weight.bin": @{
                @"offset": @0,
                @"data": weightBlob,
            },
        };
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass,
            sel_registerName("modelWithMILText:weights:optionsPlist:"),
            milData,
            weightDictionary,
            nil
        );
        if (descriptor == nil) {
            RecordFailure(evidence, @"descriptor", @"descriptor factory returned nil");
            goto cleanup;
        }
        evidence[@"descriptor_created"] = @YES;
        evidence[@"last_successful_stage"] = @"descriptor_created";

        model = ((id(*)(Class, SEL, id))objc_msgSend)(
            modelClass,
            sel_registerName("inMemoryModelWithDescriptor:"),
            descriptor
        );
        if (model == nil) {
            RecordFailure(evidence, @"model", @"in-memory model factory returned nil");
            goto cleanup;
        }
        evidence[@"model_created"] = @YES;
        evidence[@"last_successful_stage"] = @"model_created";

        NSString *hexIdentifier = ((id(*)(id, SEL))objc_msgSend)(
            model, sel_registerName("hexStringIdentifier"));
        if (![hexIdentifier isKindOfClass:NSString.class] || hexIdentifier.length == 0) {
            RecordFailure(evidence, @"artifacts", @"model did not provide a cache identifier");
            goto cleanup;
        }
        temporaryDirectory = [NSTemporaryDirectory() stringByAppendingPathComponent:hexIdentifier];
        if ([fileManager fileExistsAtPath:temporaryDirectory]) {
            RecordFailure(evidence, @"artifacts", @"refusing to overwrite an existing compiler directory");
            goto cleanup;
        }
        NSError *error = nil;
        NSString *weightsDirectory = [temporaryDirectory stringByAppendingPathComponent:@"weights"];
        if (![fileManager createDirectoryAtPath:weightsDirectory
                    withIntermediateDirectories:YES attributes:nil error:&error]) {
            RecordFailure(evidence, @"artifacts", ErrorDescription(error, @"failed to create compiler directory"));
            goto cleanup;
        }
        createdTemporaryDirectory = YES;
        BOOL milWritten = [milData writeToFile:
            [temporaryDirectory stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        BOOL weightsWritten = [weightBlob writeToFile:
            [weightsDirectory stringByAppendingPathComponent:@"weight.bin"] atomically:YES];
        if (!milWritten || !weightsWritten) {
            RecordFailure(evidence, @"artifacts", @"failed to write generated compiler artifacts");
            goto cleanup;
        }
        evidence[@"artifacts_written"] = @YES;
        evidence[@"last_successful_stage"] = @"artifacts_written";

        error = nil;
        evidence[@"compile_attempted"] = @YES;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, sel_registerName("compileWithQoS:options:error:"), kANEQoS, @{}, &error);
        if (!ok) {
            RecordFailure(evidence, @"compile", ErrorDescription(error, @"private ANE compile returned false"));
            goto cleanup;
        }
        evidence[@"compile_succeeded"] = @YES;
        evidence[@"last_successful_stage"] = @"compile_succeeded";

        error = nil;
        evidence[@"load_attempted"] = @YES;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, sel_registerName("loadWithQoS:options:error:"), kANEQoS, @{}, &error);
        if (!ok) {
            RecordFailure(evidence, @"load", ErrorDescription(error, @"private ANE load returned false"));
            goto cleanup;
        }
        modelLoaded = YES;
        evidence[@"load_succeeded"] = @YES;
        evidence[@"last_successful_stage"] = @"load_succeeded";

        inputSurface = CreateTensorSurface(tensorBytes);
        outputSurface = CreateTensorSurface(tensorBytes);
        if (inputSurface == NULL || outputSurface == NULL) {
            RecordFailure(evidence, @"iosurface", @"IOSurfaceCreate returned nil");
            goto cleanup;
        }
        evidence[@"iosurfaces_created"] = @YES;
        evidence[@"last_successful_stage"] = @"iosurfaces_created";

        BOOL inputLocked = IOSurfaceLock(inputSurface, 0, NULL) == kIOReturnSuccess;
        BOOL outputLocked = IOSurfaceLock(outputSurface, 0, NULL) == kIOReturnSuccess;
        if (!inputLocked || !outputLocked) {
            if (inputLocked) {
                IOSurfaceUnlock(inputSurface, 0, NULL);
            }
            if (outputLocked) {
                IOSurfaceUnlock(outputSurface, 0, NULL);
            }
            RecordFailure(evidence, @"iosurface", @"failed to lock tensor IOSurface");
            goto cleanup;
        }
        _Float16 *input = IOSurfaceGetBaseAddress(inputSurface);
        _Float16 *output = IOSurfaceGetBaseAddress(outputSurface);
        if (input == NULL || output == NULL) {
            IOSurfaceUnlock(inputSurface, 0, NULL);
            IOSurfaceUnlock(outputSurface, 0, NULL);
            RecordFailure(evidence, @"iosurface", @"IOSurface base address is nil");
            goto cleanup;
        }
        for (NSUInteger channel = 0; channel < kProjectionChannels; channel++) {
            for (NSUInteger spatial = 0; spatial < kProjectionSpatial; spatial++) {
                NSInteger pattern = (NSInteger)((channel * 3 + spatial * 5) % 17) - 8;
                _Float16 inputHalf = (_Float16)((float)pattern / 16.0f);
                input[channel * kProjectionSpatial + spatial] = inputHalf;
                output[channel * kProjectionSpatial + spatial] = (_Float16)NAN;
            }
        }
        for (NSUInteger outputChannel = 0; outputChannel < kProjectionChannels; outputChannel++) {
            for (NSUInteger spatial = 0; spatial < kProjectionSpatial; spatial++) {
                float sum = 0.0f;
                for (NSUInteger inputChannel = 0; inputChannel < kProjectionChannels; inputChannel++) {
                    float x = (float)input[inputChannel * kProjectionSpatial + spatial];
                    float w = referenceWeights[outputChannel * kProjectionChannels + inputChannel];
                    sum += x * w;
                }
                referenceOutput[outputChannel * kProjectionSpatial + spatial] =
                    (float)(_Float16)sum;
            }
        }
        IOSurfaceUnlock(inputSurface, 0, NULL);
        IOSurfaceUnlock(outputSurface, 0, NULL);

        id wrappedInput = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceObjectClass, sel_registerName("objectWithIOSurface:"), inputSurface);
        id wrappedOutput = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceObjectClass, sel_registerName("objectWithIOSurface:"), outputSurface);
        if (wrappedInput == nil || wrappedOutput == nil) {
            RecordFailure(evidence, @"request", @"failed to wrap tensor IOSurfaces");
            goto cleanup;
        }
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass,
            sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
            @[wrappedInput], @[@0], @[wrappedOutput], @[@0], nil, nil, @0
        );
        if (request == nil) {
            RecordFailure(evidence, @"request", @"private ANE request factory returned nil");
            goto cleanup;
        }
        evidence[@"request_created"] = @YES;
        evidence[@"last_successful_stage"] = @"request_created";

        error = nil;
        evidence[@"dispatch_attempted"] = @YES;
        CFAbsoluteTime dispatchStart = CFAbsoluteTimeGetCurrent();
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            model,
            sel_registerName("evaluateWithQoS:options:request:error:"),
            kANEQoS,
            @{},
            request,
            &error
        );
        evidence[@"dispatch_ms"] = @((CFAbsoluteTimeGetCurrent() - dispatchStart) * 1000.0);
        if (!ok) {
            RecordFailure(evidence, @"dispatch", ErrorDescription(error, @"private ANE evaluate returned false"));
            goto cleanup;
        }
        evidence[@"dispatch_succeeded"] = @YES;
        evidence[@"last_successful_stage"] = @"dispatch_succeeded";

        if (IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
            RecordFailure(evidence, @"readback", @"failed to lock output IOSurface for readback");
            goto cleanup;
        }
        output = IOSurfaceGetBaseAddress(outputSurface);
        BOOL finiteOutput = output != NULL;
        double maxAbsError = 0.0;
        if (output != NULL) {
            for (NSUInteger index = 0; index < tensorElements; index++) {
                float observed = (float)output[index];
                if (!isfinite(observed)) {
                    finiteOutput = NO;
                    break;
                }
                maxAbsError = fmax(maxAbsError, fabs((double)observed - referenceOutput[index]));
            }
        }
        IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        evidence[@"output_finite"] = @(finiteOutput);
        evidence[@"max_abs_error"] = finiteOutput ? @(maxAbsError) : [NSNull null];
        BOOL numericVerified = finiteOutput && maxAbsError <= kNumericTolerance;
        evidence[@"numeric_verified"] = @(numericVerified);
        if (!numericVerified) {
            RecordFailure(evidence, @"numeric_compare", finiteOutput ?
                @"ANE output exceeded numerical tolerance" : @"ANE output contained non-finite values");
            goto cleanup;
        }
        evidence[@"execution_verified"] = @YES;
        evidence[@"last_successful_stage"] = @"numeric_verified";
    } @catch (NSException *exception) {
        RecordFailure(evidence, @"objective_c_exception", exception.reason ?: exception.name);
    }

cleanup:
    if (modelLoaded && model != nil) {
        NSError *unloadError = nil;
        BOOL unloaded = ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, sel_registerName("unloadWithQoS:error:"), kANEQoS, &unloadError);
        if (!unloaded) {
            evidence[@"cleanup_error"] = ErrorDescription(
                unloadError, @"private ANE unload returned false");
        }
    }
    if (inputSurface != NULL) {
        CFRelease(inputSurface);
    }
    if (outputSurface != NULL) {
        CFRelease(outputSurface);
    }
    if (createdTemporaryDirectory && temporaryDirectory != nil &&
        [fileManager fileExistsAtPath:temporaryDirectory]) {
        NSError *cleanupError = nil;
        if (![fileManager removeItemAtPath:temporaryDirectory error:&cleanupError] &&
            [evidence[@"cleanup_error"] length] == 0) {
            evidence[@"cleanup_error"] = ErrorDescription(
                cleanupError, @"failed to remove generated compiler directory");
        }
    }
    free(referenceWeights);
    free(referenceOutput);
    return evidence;
}

static NSDictionary<NSString *, id> *BuildReport(BOOL executeProjection) {
    NSMutableArray<NSString *> *errors = [NSMutableArray array];

    NSDictionary<NSString *, id> *aneFramework = LoadFramework(
        @"AppleNeuralEngine",
        @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        errors
    );
    NSDictionary<NSString *, id> *compilerFramework = LoadFramework(
        @"ANECompiler",
        @"/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
        errors
    );

    NSArray<NSString *> *classNames = @[
        @"_ANEClient",
        @"_ANECompiler",
        @"_ANEInMemoryModelDescriptor",
        @"_ANEModel",
        @"_ANEInMemoryModel",
        @"_ANERequest",
        @"_ANEIOSurfaceObject",
    ];
    NSMutableDictionary<NSString *, NSDictionary<NSString *, id> *> *classes =
        [NSMutableDictionary dictionaryWithCapacity:classNames.count];
    for (NSString *name in classNames) {
        classes[name] = InspectClass(name);
    }

    NSArray<NSString *> *descriptorCandidates = @[
        @"_ANEInMemoryModelDescriptor",
        @"_ANEInMemoryModel",
        @"_ANEModel",
    ];
    NSString *descriptorClass = @"";
    for (NSString *candidate in descriptorCandidates) {
        if ([classes[candidate][@"present"] boolValue]) {
            descriptorClass = candidate;
            break;
        }
    }

    BOOL frameworksLoaded =
        [aneFramework[@"loaded"] boolValue] && [compilerFramework[@"loaded"] boolValue];
    BOOL clientPresent = [classes[@"_ANEClient"][@"present"] boolValue];
    BOOL compilerPresent = [classes[@"_ANECompiler"][@"present"] boolValue];
    BOOL descriptorPresent = descriptorClass.length > 0;
    BOOL clientEntrypointsPresent =
        HasClassMethod(@"_ANEClient", @"sharedConnection") &&
        HasInstanceMethod(@"_ANEClient", @"loadModel:options:qos:error:") &&
        HasInstanceMethod(@"_ANEClient", @"evaluateWithModel:options:request:qos:error:");
    BOOL descriptorFactoryPresent = HasClassMethod(
        @"_ANEInMemoryModelDescriptor", @"modelWithMILText:weights:optionsPlist:");
    BOOL inMemoryCompilerPresent =
        HasClassMethod(@"_ANEInMemoryModel", @"inMemoryModelWithDescriptor:") &&
        HasInstanceMethod(@"_ANEInMemoryModel", @"compileWithQoS:options:error:");
    NSString *compilerClass = compilerPresent ? @"_ANECompiler" :
        (inMemoryCompilerPresent ? @"_ANEInMemoryModel" : @"");
    BOOL compilerEquivalentPresent = compilerClass.length > 0;
    BOOL modelLifecyclePresent =
        HasInstanceMethod(@"_ANEInMemoryModel", @"loadWithQoS:options:error:") &&
        HasInstanceMethod(@"_ANEInMemoryModel", @"evaluateWithQoS:options:request:error:") &&
        HasInstanceMethod(@"_ANEInMemoryModel", @"unloadWithQoS:error:");
    BOOL requestClassPresent = [classes[@"_ANERequest"][@"present"] boolValue];
    BOOL surfaceObjectClassPresent = [classes[@"_ANEIOSurfaceObject"][@"present"] boolValue];
    BOOL requestFactoryPresent = HasClassMethod(
        @"_ANERequest",
        @"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"
    );
    BOOL surfaceObjectFactoryPresent = HasClassMethod(
        @"_ANEIOSurfaceObject", @"objectWithIOSurface:");
    BOOL surfaceSatisfied = frameworksLoaded && clientPresent && clientEntrypointsPresent &&
        descriptorPresent && descriptorFactoryPresent && compilerEquivalentPresent &&
        modelLifecyclePresent;
    BOOL executionSurfaceSatisfied = surfaceSatisfied && requestClassPresent &&
        surfaceObjectClassPresent && requestFactoryPresent && surfaceObjectFactoryPresent;

    NSDictionary<NSString *, id> *execution = executeProjection ?
        RunProjectionProof(executionSurfaceSatisfied) : InitialExecutionEvidence(NO);
    NSString *executionError = execution[@"error"];
    if (executionError.length > 0) {
        [errors addObject:[NSString stringWithFormat:@"projection proof: %@", executionError]];
    }
    NSString *cleanupError = execution[@"cleanup_error"];
    if (cleanupError.length > 0) {
        [errors addObject:[NSString stringWithFormat:@"projection cleanup: %@", cleanupError]];
    }

    NSOperatingSystemVersion version = NSProcessInfo.processInfo.operatingSystemVersion;
    NSString *versionString = [NSString stringWithFormat:@"%ld.%ld.%ld",
        (long)version.majorVersion,
        (long)version.minorVersion,
        (long)version.patchVersion];
    NSString *build = SysctlString("kern.osversion");
    NSString *chip = SysctlString("machdep.cpu.brand_string");
    if (chip.length == 0) {
        chip = SysctlString("hw.model");
    }

    return @{
        @"schema_version": @(kSchemaVersion),
        @"probe": kProbeName,
        @"platform": @{
            @"os": @"macOS",
            @"version": versionString,
            @"build": build,
            @"architecture": ArchitectureName(),
            @"chip": chip,
        },
        @"frameworks": @{
            @"AppleNeuralEngine": aneFramework,
            @"ANECompiler": compilerFramework,
        },
        @"objective_c": @{
            @"classes": classes,
            @"descriptor_candidates": descriptorCandidates,
            @"descriptor_class": descriptorClass,
        },
        @"required_surface": @{
            @"private_frameworks_loaded": @(frameworksLoaded),
            @"ane_client_present": @(clientPresent),
            @"ane_client_entrypoints_present": @(clientEntrypointsPresent),
            @"ane_compiler_present": @(compilerPresent),
            @"compiler_equivalent_present": @(compilerEquivalentPresent),
            @"compiler_class": compilerClass,
            @"descriptor_present": @(descriptorPresent),
            @"descriptor_factory_present": @(descriptorFactoryPresent),
            @"descriptor_class": descriptorClass,
            @"model_lifecycle_entrypoints_present": @(modelLifecyclePresent),
            @"satisfied": @(surfaceSatisfied),
            @"request_class_present": @(requestClassPresent),
            @"iosurface_object_class_present": @(surfaceObjectClassPresent),
            @"request_factory_present": @(requestFactoryPresent),
            @"iosurface_object_factory_present": @(surfaceObjectFactoryPresent),
            @"execution_surface_satisfied": @(executionSurfaceSatisfied),
        },
        @"execution": execution,
        @"errors": errors,
    };
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        BOOL executeProjection = NO;
        if (argc == 2 && strcmp(argv[1], "--execute-projection") == 0) {
            executeProjection = YES;
        } else if (argc != 1) {
            fprintf(stderr, "usage: strata-ane-probe [--execute-projection]\n");
            return 64;
        }
        NSDictionary<NSString *, id> *report = BuildReport(executeProjection);
        NSError *error = nil;
        NSData *json = [NSJSONSerialization dataWithJSONObject:report options:0 error:&error];
        if (json == nil) {
            fprintf(stderr, "failed to serialize ANE probe report: %s\n",
                error.localizedDescription.UTF8String);
            return 2;
        }
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
        return 0;
    }
}
