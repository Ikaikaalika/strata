#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <objc/runtime.h>
#include <sys/sysctl.h>

static NSString *const kProbeName = @"strata-ane-capability";
static const NSInteger kSchemaVersion = 1;

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

static NSDictionary<NSString *, id> *BuildReport(void) {
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
    BOOL surfaceSatisfied = frameworksLoaded && clientPresent && clientEntrypointsPresent &&
        descriptorPresent && descriptorFactoryPresent && compilerEquivalentPresent &&
        modelLifecyclePresent;

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
        },
        @"execution": @{
            @"compile_attempted": @NO,
            @"dispatch_attempted": @NO,
            @"execution_verified": @NO,
            @"numeric_verified": @NO,
        },
        @"errors": errors,
    };
}

int main(int argc, const char *argv[]) {
    (void)argc;
    (void)argv;
    @autoreleasepool {
        NSDictionary<NSString *, id> *report = BuildReport();
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
