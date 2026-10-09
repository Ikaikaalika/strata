#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <sys/sysctl.h>
#include <vector>

namespace {

using Float16 = _Float16;

constexpr uint32_t kTokenCount = 64;
constexpr uint32_t kInputWidth = 256;
constexpr uint32_t kOutputWidth = 256;
constexpr uint32_t kWarmupIterations = 5;
constexpr uint32_t kMeasuredIterations = 50;
constexpr uint32_t kPhaseDispatchCount = 16;
constexpr double kMaximumAllowedError = 0.01;

struct Distribution {
  double minimum = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double mean = 0.0;
};

void EmitJSON(NSDictionary *payload) {
  NSError *error = nil;
  NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                 options:NSJSONWritingSortedKeys
                                                   error:&error];
  if (data == nil) {
    std::fprintf(stderr, "failed to serialize benchmark result: %s\n",
                 error.localizedDescription.UTF8String);
    return;
  }
  std::fwrite(data.bytes, 1, data.length, stdout);
  std::fputc('\n', stdout);
}

std::string SysctlString(const char *name) {
  size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
    return "unknown";
  }
  std::string value(size, '\0');
  if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
    return "unknown";
  }
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

NSString *EnvironmentString(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? @"unknown" : [NSString stringWithUTF8String:value];
}

NSString *ThermalStateString(NSProcessInfoThermalState state) {
  switch (state) {
    case NSProcessInfoThermalStateNominal:
      return @"nominal";
    case NSProcessInfoThermalStateFair:
      return @"fair";
    case NSProcessInfoThermalStateSerious:
      return @"serious";
    case NSProcessInfoThermalStateCritical:
      return @"critical";
  }
  return @"unknown";
}

Distribution Summarize(std::vector<double> samples) {
  Distribution result;
  if (samples.empty()) {
    return result;
  }
  result.mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  result.minimum = samples.front();
  const size_t middle = samples.size() / 2;
  result.median = samples.size() % 2 == 0
                      ? 0.5 * (samples[middle - 1] + samples[middle])
                      : samples[middle];
  const size_t p95Index =
      std::min(samples.size() - 1,
               static_cast<size_t>(std::ceil(0.95 * samples.size())) - 1);
  result.p95 = samples[p95Index];
  return result;
}

NSDictionary *DistributionJSON(const Distribution &distribution) {
  return @{
    @"minimum_ms" : @(distribution.minimum),
    @"median_ms" : @(distribution.median),
    @"p95_ms" : @(distribution.p95),
    @"mean_ms" : @(distribution.mean),
  };
}

double GFLOPs(double milliseconds) {
  if (milliseconds <= 0.0) {
    return 0.0;
  }
  const double operations = 2.0 * double(kTokenCount) * double(kInputWidth) *
                            double(kOutputWidth);
  return operations / (milliseconds * 1.0e6);
}

__attribute__((noinline)) void CPUReference(const Float16 *input,
                                             const Float16 *weight,
                                             Float16 *output) {
  for (uint32_t token = 0; token < kTokenCount; ++token) {
    for (uint32_t outputColumn = 0; outputColumn < kOutputWidth;
         ++outputColumn) {
      float sum = 0.0f;
      const size_t inputOffset = size_t(token) * kInputWidth;
      const size_t weightOffset = size_t(outputColumn) * kInputWidth;
      for (uint32_t column = 0; column < kInputWidth; ++column) {
        sum = std::fma(float(input[inputOffset + column]),
                       float(weight[weightOffset + column]), sum);
      }
      output[size_t(token) * kOutputWidth + outputColumn] = Float16(sum);
    }
  }
}

int Fail(NSString *reason, NSString *deviceName = @"") {
  const std::string osBuild = SysctlString("kern.osversion");
  EmitJSON(@{
    @"schema_version" : @1,
    @"benchmark" : @"lokahi-native-linear-projection",
    @"success" : @NO,
    @"failure_reason" : reason,
    @"software" : @{
      @"revision" : EnvironmentString("LOKAHI_BENCH_REVISION"),
      @"dirty" : @([EnvironmentString("LOKAHI_BENCH_DIRTY") isEqualToString:@"true"]),
    },
    @"hardware" : @{
      @"chip" : deviceName,
      @"architecture" : @"arm64",
      @"macos_version" : NSProcessInfo.processInfo.operatingSystemVersionString,
      @"macos_build" : [NSString stringWithUTF8String:osBuild.c_str()],
    },
  });
  return EXIT_FAILURE;
}

std::filesystem::path DefaultMetallibPath(const char *executable) {
  std::error_code error;
  const std::filesystem::path executablePath =
      std::filesystem::weakly_canonical(executable, error);
  const std::filesystem::path parent =
      error ? std::filesystem::path(executable).parent_path()
            : executablePath.parent_path();
  return parent / "lokahi-linear.metallib";
}

}  // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    const char *configuredPath = std::getenv("LOKAHI_LINEAR_METALLIB_PATH");
    const std::filesystem::path metallibPath =
        argc > 1 ? std::filesystem::path(argv[1])
                 : (configuredPath != nullptr
                        ? std::filesystem::path(configuredPath)
                        : DefaultMetallibPath(argv[0]));

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return Fail(@"MTLCreateSystemDefaultDevice returned nil");
    }
    NSString *deviceName = device.name ?: @"unknown";

    NSError *error = nil;
    NSURL *libraryURL = [NSURL fileURLWithPath:
        [NSString stringWithUTF8String:metallibPath.c_str()]];
    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
    if (library == nil) {
      return Fail([NSString stringWithFormat:@"failed to load metallib: %@",
                                             error.localizedDescription],
                  deviceName);
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@"linear_projection_f16"];
    if (function == nil) {
      return Fail(@"metallib does not contain linear_projection_f16",
                  deviceName);
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      return Fail([NSString stringWithFormat:@"failed to create pipeline: %@",
                                             error.localizedDescription],
                  deviceName);
    }
    id<MTLFunction> tiledFunction =
        [library newFunctionWithName:@"linear_projection_f16_tiled"];
    if (tiledFunction == nil) {
      return Fail(@"metallib does not contain linear_projection_f16_tiled",
                  deviceName);
    }
    id<MTLComputePipelineState> tiledPipeline =
        [device newComputePipelineStateWithFunction:tiledFunction error:&error];
    if (tiledPipeline == nil) {
      return Fail([NSString stringWithFormat:
          @"failed to create tiled pipeline: %@", error.localizedDescription],
          deviceName);
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      return Fail(@"failed to create Metal command queue", deviceName);
    }

    const size_t inputCount = size_t(kTokenCount) * kInputWidth;
    const size_t weightCount = size_t(kOutputWidth) * kInputWidth;
    const size_t outputCount = size_t(kTokenCount) * kOutputWidth;
    std::vector<Float16> input(inputCount);
    std::vector<Float16> weight(weightCount);
    std::vector<Float16> reference(outputCount);
    std::vector<Float16> cpuOutput(outputCount);

    for (size_t index = 0; index < inputCount; ++index) {
      input[index] = Float16(0.20f * std::sin(float(index) * 0.017f) +
                             0.05f * std::cos(float(index) * 0.011f));
    }
    for (size_t index = 0; index < weightCount; ++index) {
      weight[index] = Float16(0.08f * std::sin(float(index) * 0.007f + 0.3f) -
                              0.03f * std::cos(float(index) * 0.013f));
    }
    CPUReference(input.data(), weight.data(), reference.data());

    for (uint32_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
      CPUReference(input.data(), weight.data(), cpuOutput.data());
    }
    std::vector<double> cpuSamples;
    cpuSamples.reserve(kMeasuredIterations);
    volatile float cpuSink = 0.0f;
    for (uint32_t iteration = 0; iteration < kMeasuredIterations; ++iteration) {
      const auto started = std::chrono::steady_clock::now();
      CPUReference(input.data(), weight.data(), cpuOutput.data());
      const auto ended = std::chrono::steady_clock::now();
      cpuSamples.push_back(
          std::chrono::duration<double, std::milli>(ended - started).count());
      cpuSink = cpuSink + float(cpuOutput[iteration % outputCount]);
    }

    const MTLResourceOptions shared = MTLResourceStorageModeShared;
    id<MTLBuffer> inputBuffer =
        [device newBufferWithBytes:input.data()
                            length:inputCount * sizeof(Float16)
                           options:shared];
    id<MTLBuffer> weightBuffer =
        [device newBufferWithBytes:weight.data()
                            length:weightCount * sizeof(Float16)
                           options:shared];
    id<MTLBuffer> outputBuffer =
        [device newBufferWithLength:outputCount * sizeof(Float16)
                            options:shared];
    id<MTLBuffer> tiledOutputBuffer =
        [device newBufferWithLength:outputCount * sizeof(Float16)
                            options:shared];
    id<MTLBuffer> phaseOutputBuffer =
        [device newBufferWithLength:outputCount * sizeof(Float16) *
                                    kPhaseDispatchCount
                            options:shared];
    if (inputBuffer == nil || weightBuffer == nil || outputBuffer == nil ||
        tiledOutputBuffer == nil || phaseOutputBuffer == nil) {
      return Fail(@"failed to allocate shared Metal buffers", deviceName);
    }

    auto dispatchOnce = [&](id<MTLComputePipelineState> selectedPipeline,
                            id<MTLBuffer> selectedOutput) -> NSDictionary * {
      @autoreleasepool {
        const auto started = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil) {
          return @{@"failure" : @"failed to create Metal command buffer"};
        }
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
          return @{@"failure" : @"failed to create Metal command encoder"};
        }
        [encoder setComputePipelineState:selectedPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:selectedOutput offset:0 atIndex:2];
        uint32_t tokens = kTokenCount;
        uint32_t inputWidth = kInputWidth;
        uint32_t outputWidth = kOutputWidth;
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:3];
        [encoder setBytes:&inputWidth length:sizeof(inputWidth) atIndex:4];
        [encoder setBytes:&outputWidth length:sizeof(outputWidth) atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(kOutputWidth, kTokenCount, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const auto ended = std::chrono::steady_clock::now();
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
          return @{
            @"failure" : commandBuffer.error.localizedDescription
                ?: @"Metal command failed"
          };
        }
        const double wallMilliseconds =
            std::chrono::duration<double, std::milli>(ended - started).count();
        double gpuMilliseconds = 0.0;
        if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime &&
            commandBuffer.GPUStartTime > 0.0) {
          gpuMilliseconds =
              (commandBuffer.GPUEndTime - commandBuffer.GPUStartTime) * 1000.0;
        }
        return @{
          @"wall_ms" : @(wallMilliseconds),
          @"gpu_ms" : @(gpuMilliseconds),
        };
      }
    };

    auto dispatchPhase = [&]() -> NSDictionary * {
      @autoreleasepool {
        const auto started = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil) {
          return @{ @"failure" : @"failed to create phase command buffer" };
        }
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
          return @{ @"failure" : @"failed to create phase compute encoder" };
        }
        [encoder setComputePipelineState:tiledPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:weightBuffer offset:0 atIndex:1];
        uint32_t tokens = kTokenCount;
        uint32_t inputWidth = kInputWidth;
        uint32_t outputWidth = kOutputWidth;
        [encoder setBytes:&tokens length:sizeof(tokens) atIndex:3];
        [encoder setBytes:&inputWidth length:sizeof(inputWidth) atIndex:4];
        [encoder setBytes:&outputWidth length:sizeof(outputWidth) atIndex:5];
        const size_t outputBytes = outputCount * sizeof(Float16);
        for (uint32_t dispatch = 0; dispatch < kPhaseDispatchCount; ++dispatch) {
          [encoder setBuffer:phaseOutputBuffer
                     offset:size_t(dispatch) * outputBytes
                    atIndex:2];
          [encoder dispatchThreads:MTLSizeMake(kOutputWidth, kTokenCount, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        }
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const auto ended = std::chrono::steady_clock::now();
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
          return @{
            @"failure" : commandBuffer.error.localizedDescription
                ?: @"Metal phase command failed"
          };
        }
        const double wallMilliseconds =
            std::chrono::duration<double, std::milli>(ended - started).count();
        double gpuMilliseconds = 0.0;
        if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime &&
            commandBuffer.GPUStartTime > 0.0) {
          gpuMilliseconds =
              (commandBuffer.GPUEndTime - commandBuffer.GPUStartTime) * 1000.0;
        }
        return @{
          @"wall_ms" : @(wallMilliseconds),
          @"gpu_ms" : @(gpuMilliseconds),
        };
      }
    };

    for (uint32_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
      NSDictionary *baselineSample = dispatchOnce(pipeline, outputBuffer);
      if (baselineSample[@"failure"] != nil) {
        return Fail(baselineSample[@"failure"], deviceName);
      }
      NSDictionary *tiledSample =
          dispatchOnce(tiledPipeline, tiledOutputBuffer);
      if (tiledSample[@"failure"] != nil) {
        return Fail(tiledSample[@"failure"], deviceName);
      }
      NSDictionary *phaseSample = dispatchPhase();
      if (phaseSample[@"failure"] != nil) {
        return Fail(phaseSample[@"failure"], deviceName);
      }
    }

    std::vector<double> gpuWallSamples;
    std::vector<double> gpuDeviceSamples;
    std::vector<double> tiledWallSamples;
    std::vector<double> tiledDeviceSamples;
    std::vector<double> phaseWallSamples;
    std::vector<double> phaseDeviceSamples;
    gpuWallSamples.reserve(kMeasuredIterations);
    gpuDeviceSamples.reserve(kMeasuredIterations);
    tiledWallSamples.reserve(kMeasuredIterations);
    tiledDeviceSamples.reserve(kMeasuredIterations);
    phaseWallSamples.reserve(kMeasuredIterations);
    phaseDeviceSamples.reserve(kMeasuredIterations);
    for (uint32_t iteration = 0; iteration < kMeasuredIterations; ++iteration) {
      NSDictionary *sample = dispatchOnce(pipeline, outputBuffer);
      if (sample[@"failure"] != nil) {
        return Fail(sample[@"failure"], deviceName);
      }
      gpuWallSamples.push_back([sample[@"wall_ms"] doubleValue]);
      const double gpuMilliseconds = [sample[@"gpu_ms"] doubleValue];
      if (gpuMilliseconds > 0.0) {
        gpuDeviceSamples.push_back(gpuMilliseconds);
      }
      NSDictionary *tiledSample =
          dispatchOnce(tiledPipeline, tiledOutputBuffer);
      if (tiledSample[@"failure"] != nil) {
        return Fail(tiledSample[@"failure"], deviceName);
      }
      tiledWallSamples.push_back([tiledSample[@"wall_ms"] doubleValue]);
      const double tiledMilliseconds = [tiledSample[@"gpu_ms"] doubleValue];
      if (tiledMilliseconds > 0.0) {
        tiledDeviceSamples.push_back(tiledMilliseconds);
      }
      NSDictionary *phaseSample = dispatchPhase();
      if (phaseSample[@"failure"] != nil) {
        return Fail(phaseSample[@"failure"], deviceName);
      }
      phaseWallSamples.push_back([phaseSample[@"wall_ms"] doubleValue]);
      const double phaseMilliseconds = [phaseSample[@"gpu_ms"] doubleValue];
      if (phaseMilliseconds > 0.0) {
        phaseDeviceSamples.push_back(phaseMilliseconds);
      }
    }

    const Float16 *actual = static_cast<const Float16 *>(outputBuffer.contents);
    const Float16 *tiledActual =
        static_cast<const Float16 *>(tiledOutputBuffer.contents);
    const Float16 *phaseActual =
        static_cast<const Float16 *>(phaseOutputBuffer.contents);
    double maxError = 0.0;
    double sumError = 0.0;
    double tiledMaxError = 0.0;
    double tiledSumError = 0.0;
    double phaseMaxError = 0.0;
    double phaseSumError = 0.0;
    for (size_t index = 0; index < outputCount; ++index) {
      const double errorValue =
          std::abs(double(float(actual[index])) -
                   double(float(reference[index])));
      maxError = std::max(maxError, errorValue);
      sumError += errorValue;
      const double tiledErrorValue =
          std::abs(double(float(tiledActual[index])) -
                   double(float(reference[index])));
      tiledMaxError = std::max(tiledMaxError, tiledErrorValue);
      tiledSumError += tiledErrorValue;
      for (uint32_t dispatch = 0; dispatch < kPhaseDispatchCount; ++dispatch) {
        const double phaseErrorValue = std::abs(
            double(float(phaseActual[size_t(dispatch) * outputCount + index])) -
            double(float(reference[index])));
        phaseMaxError = std::max(phaseMaxError, phaseErrorValue);
        phaseSumError += phaseErrorValue;
      }
    }
    const double meanError = sumError / double(outputCount);
    const double tiledMeanError = tiledSumError / double(outputCount);
    const double phaseMeanError =
        phaseSumError / double(outputCount * kPhaseDispatchCount);
    const bool correctnessPassed =
        std::isfinite(maxError) && maxError <= kMaximumAllowedError &&
        std::isfinite(tiledMaxError) &&
        tiledMaxError <= kMaximumAllowedError &&
        std::isfinite(phaseMaxError) &&
        phaseMaxError <= kMaximumAllowedError;
    if (!correctnessPassed) {
      return Fail([NSString stringWithFormat:
          @"Metal result exceeded maximum error: untiled %.9f tiled %.9f phase %.9f",
          maxError, tiledMaxError, phaseMaxError], deviceName);
    }

    const Distribution cpu = Summarize(cpuSamples);
    const Distribution gpuWall = Summarize(gpuWallSamples);
    const Distribution gpuDevice = Summarize(gpuDeviceSamples);
    const Distribution tiledWall = Summarize(tiledWallSamples);
    const Distribution tiledDevice = Summarize(tiledDeviceSamples);
    const Distribution phaseWall = Summarize(phaseWallSamples);
    const Distribution phaseDevice = Summarize(phaseDeviceSamples);
    const double phaseWallPerProjection =
        phaseWall.median / double(kPhaseDispatchCount);
    const double phaseDevicePerProjection =
        phaseDevice.median / double(kPhaseDispatchCount);
    const std::string osBuild = SysctlString("kern.osversion");
    EmitJSON(@{
      @"schema_version" : @1,
      @"benchmark" : @"lokahi-native-linear-projection",
      @"success" : @YES,
      @"failure_reason" : [NSNull null],
      @"evidence_kind" : @"hardware",
      @"hardware" : @{
        @"chip" : deviceName,
        @"architecture" : @"arm64",
        @"macos_version" : NSProcessInfo.processInfo.operatingSystemVersionString,
        @"macos_build" : [NSString stringWithUTF8String:osBuild.c_str()],
        @"logical_cpu_count" : @(NSProcessInfo.processInfo.processorCount),
        @"physical_memory_bytes" : @(NSProcessInfo.processInfo.physicalMemory),
        @"thermal_state_after_run" : ThermalStateString(NSProcessInfo.processInfo.thermalState),
      },
      @"software" : @{
        @"revision" : EnvironmentString("LOKAHI_BENCH_REVISION"),
        @"dirty" : @([EnvironmentString("LOKAHI_BENCH_DIRTY") isEqualToString:@"true"]),
        @"executable" : @"native Objective-C++ and direct Metal; no Python hot path",
        @"sdk" : EnvironmentString("LOKAHI_BENCH_SDK"),
        @"metal_language_standard" : @"metal3.1",
      },
      @"operation" : @{
        @"kind" : @"linear",
        @"phase" : @"prefill",
        @"input_shape" : @[@(kTokenCount), @(kInputWidth)],
        @"weight_shape" : @[@(kOutputWidth), @(kInputWidth)],
        @"weight_layout" : @"out_in",
        @"output_shape" : @[@(kTokenCount), @(kOutputWidth)],
        @"dtype" : @"float16",
        @"accumulator_dtype" : @"float32",
        @"fixture" : @"analytic_sine_cosine_v1",
      },
      @"correctness" : @{
        @"reference" : @"native scalar CPU float32 accumulation followed by float16 cast",
        @"passed" : @(correctnessPassed),
        @"maximum_allowed_abs_error" : @(kMaximumAllowedError),
        @"untiled_max_abs_error" : @(maxError),
        @"untiled_mean_abs_error" : @(meanError),
        @"tiled_max_abs_error" : @(tiledMaxError),
        @"tiled_mean_abs_error" : @(tiledMeanError),
        @"phase_max_abs_error" : @(phaseMaxError),
        @"phase_mean_abs_error" : @(phaseMeanError),
      },
      @"warmup_iterations" : @(kWarmupIterations),
      @"measured_iterations" : @(kMeasuredIterations),
      @"lanes" : @{
        @"cpu_scalar_reference" : @{
          @"timing" : DistributionJSON(cpu),
          @"median_gflops" : @(GFLOPs(cpu.median)),
          @"timing_boundary" : @"one complete native scalar projection; output retained; excludes allocation and fixture generation",
          @"optimization_role" : @"correctness reference, not optimized CPU baseline",
          @"sink" : @(cpuSink),
        },
        @"metal_direct_untiled" : @{
          @"wall_timing" : DistributionJSON(gpuWall),
          @"gpu_device_timing" : DistributionJSON(gpuDevice),
          @"median_wall_gflops" : @(GFLOPs(gpuWall.median)),
          @"median_device_gflops" : @(GFLOPs(gpuDevice.median)),
          @"timing_boundary" : @"fresh command buffer and encoder creation, encode, commit, GPU execution, and wait; persistent pipeline and buffers; excludes compilation, allocation, and warmup",
          @"kernel" : @"one output element per thread; untiled correctness baseline",
        },
        @"metal_direct_tiled" : @{
          @"wall_timing" : DistributionJSON(tiledWall),
          @"gpu_device_timing" : DistributionJSON(tiledDevice),
          @"median_wall_gflops" : @(GFLOPs(tiledWall.median)),
          @"median_device_gflops" : @(GFLOPs(tiledDevice.median)),
          @"timing_boundary" : @"fresh command buffer and encoder creation, encode, commit, GPU execution, and wait; persistent pipeline and buffers; excludes compilation, allocation, and warmup",
          @"kernel" : @"16-output by 8-token threadgroup tile with 32-wide staged K slices",
          @"median_wall_latency_reduction_percent_vs_untiled" :
              @(100.0 * (gpuWall.median - tiledWall.median) / gpuWall.median),
          @"median_device_latency_reduction_percent_vs_untiled" :
              @(100.0 * (gpuDevice.median - tiledDevice.median) /
                gpuDevice.median),
        },
        @"metal_tiled_phase_program" : @{
          @"dispatches_per_command_buffer" : @(kPhaseDispatchCount),
          @"total_wall_timing" : DistributionJSON(phaseWall),
          @"total_gpu_device_timing" : DistributionJSON(phaseDevice),
          @"median_wall_ms_per_projection" : @(phaseWallPerProjection),
          @"median_gpu_device_ms_per_projection" : @(phaseDevicePerProjection),
          @"median_wall_gflops" : @(GFLOPs(phaseWallPerProjection)),
          @"median_device_gflops" : @(GFLOPs(phaseDevicePerProjection)),
          @"median_wall_latency_reduction_percent_vs_single_tiled" :
              @(100.0 * (tiledWall.median - phaseWallPerProjection) /
                tiledWall.median),
          @"timing_boundary" : @"one command buffer and compute encoder containing sixteen independent tiled projections; reports total and amortized per-projection latency; includes encode, commit, GPU execution, and wait",
          @"optimization_role" : @"command submission amortization proxy for a persistent phase program",
        },
      },
      @"limitations" : @[
        @"The CPU lane is a scalar correctness reference and does not represent Accelerate or BNNS performance.",
        @"Both Metal kernels are learning baselines, not production transformer kernels.",
        @"The phase-program lane repeats one independent projection sixteen times; it measures dispatch amortization, not a transformer layer graph or token throughput.",
        @"The fixed shape matches the bounded ANE prefill envelope but this executable does not dispatch ANE.",
        @"This generated fixture is not a token-throughput or real-model benchmark.",
      ],
    });
    return EXIT_SUCCESS;
  }
}
