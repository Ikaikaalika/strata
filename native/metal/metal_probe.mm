#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRows = 64;
constexpr uint32_t kHiddenSize = 1024;
constexpr float kEpsilon = 1.0e-5f;
constexpr uint32_t kWarmupIterations = 3;
constexpr uint32_t kMeasuredIterations = 20;

void EmitJSON(NSDictionary *payload) {
  NSError *error = nil;
  NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                 options:NSJSONWritingSortedKeys
                                                   error:&error];
  if (data == nil) {
    std::fprintf(stderr, "failed to serialize probe result: %s\n",
                 error.localizedDescription.UTF8String);
    return;
  }
  std::fwrite(data.bytes, 1, data.length, stdout);
  std::fputc('\n', stdout);
}

int Fail(NSString *reason, NSString *deviceName = @"") {
  EmitJSON(@{
    @"schema_version" : @1,
    @"backend" : @"metal",
    @"operation" : @"rmsnorm_residual_f32",
    @"device_name" : deviceName,
    @"dispatch_success" : @NO,
    @"failure_reason" : reason,
    @"numerical_max_error" : [NSNull null],
    @"timing" : @{
      @"warmup_iterations" : @(kWarmupIterations),
      @"measured_iterations" : @(kMeasuredIterations),
      @"method" : @"serial wall clock around commit plus waitUntilCompleted; excludes library load, allocation, and warmup"
    }
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
  return parent / "lokahi-metal.metallib";
}

}  // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    const char *configuredPath = std::getenv("LOKAHI_METALLIB_PATH");
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

    id<MTLFunction> function = [library newFunctionWithName:@"rmsnorm_residual_f32"];
    if (function == nil) {
      return Fail(@"metallib does not contain rmsnorm_residual_f32", deviceName);
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      return Fail([NSString stringWithFormat:@"failed to create pipeline: %@",
                                             error.localizedDescription],
                  deviceName);
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      return Fail(@"failed to create Metal command queue", deviceName);
    }

    const size_t elementCount = size_t(kRows) * size_t(kHiddenSize);
    const size_t tensorBytes = elementCount * sizeof(float);
    const size_t weightBytes = size_t(kHiddenSize) * sizeof(float);
    std::vector<float> input(elementCount);
    std::vector<float> residual(elementCount);
    std::vector<float> weight(kHiddenSize);
    std::vector<float> reference(elementCount);

    for (size_t index = 0; index < elementCount; ++index) {
      input[index] = 0.75f * std::sin(float(index) * 0.013f) +
                     0.20f * std::cos(float(index) * 0.007f);
      residual[index] = 0.10f * std::sin(float(index) * 0.003f + 0.5f);
    }
    for (size_t column = 0; column < kHiddenSize; ++column) {
      weight[column] = 0.85f + 0.30f * float(column) / float(kHiddenSize - 1);
    }

    for (uint32_t row = 0; row < kRows; ++row) {
      const size_t offset = size_t(row) * size_t(kHiddenSize);
      float sumSquares = 0.0f;
      for (uint32_t column = 0; column < kHiddenSize; ++column) {
        const float value = input[offset + column];
        sumSquares += value * value;
      }
      const float inverseRms =
          1.0f / std::sqrt(sumSquares / float(kHiddenSize) + kEpsilon);
      for (uint32_t column = 0; column < kHiddenSize; ++column) {
        const size_t index = offset + column;
        reference[index] =
            input[index] * inverseRms * weight[column] + residual[index];
      }
    }

    const MTLResourceOptions shared = MTLResourceStorageModeShared;
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                   length:tensorBytes
                                                  options:shared];
    id<MTLBuffer> residualBuffer = [device newBufferWithBytes:residual.data()
                                                      length:tensorBytes
                                                     options:shared];
    id<MTLBuffer> weightBuffer = [device newBufferWithBytes:weight.data()
                                                    length:weightBytes
                                                   options:shared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:tensorBytes
                                                    options:shared];
    if (inputBuffer == nil || residualBuffer == nil || weightBuffer == nil ||
        outputBuffer == nil) {
      return Fail(@"failed to allocate shared Metal buffers", deviceName);
    }

    auto dispatchOnce = [&]() -> NSString * {
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
          return @"failed to create Metal command buffer or encoder";
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:residualBuffer offset:0 atIndex:1];
        [encoder setBuffer:weightBuffer offset:0 atIndex:2];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];
        uint32_t hiddenSize = kHiddenSize;
        float epsilon = kEpsilon;
        uint32_t rows = kRows;
        [encoder setBytes:&hiddenSize length:sizeof(hiddenSize) atIndex:4];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:6];
        const NSUInteger width =
            std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, kRows);
        [encoder dispatchThreads:MTLSizeMake(kRows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
          return commandBuffer.error.localizedDescription ?: @"Metal command failed";
        }
      }
      return nil;
    };

    for (uint32_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
      NSString *failure = dispatchOnce();
      if (failure != nil) {
        return Fail(failure, deviceName);
      }
    }

    const auto start = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < kMeasuredIterations; ++iteration) {
      NSString *failure = dispatchOnce();
      if (failure != nil) {
        return Fail(failure, deviceName);
      }
    }
    const auto end = std::chrono::steady_clock::now();
    const double totalMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    const float *actual = static_cast<const float *>(outputBuffer.contents);
    double maxError = 0.0;
    for (size_t index = 0; index < elementCount; ++index) {
      maxError = std::max(
          maxError,
          std::abs(double(actual[index]) - double(reference[index])));
    }
    if (!std::isfinite(maxError)) {
      return Fail(@"GPU output contains a non-finite numerical error", deviceName);
    }

    EmitJSON(@{
      @"schema_version" : @1,
      @"backend" : @"metal",
      @"operation" : @"rmsnorm_residual_f32",
      @"device_name" : deviceName,
      @"dispatch_success" : @YES,
      @"failure_reason" : [NSNull null],
      @"numerical_max_error" : @(maxError),
      @"shape" : @[@(kRows), @(kHiddenSize)],
      @"dtype" : @"float32",
      @"timing" : @{
        @"warmup_iterations" : @(kWarmupIterations),
        @"measured_iterations" : @(kMeasuredIterations),
        @"total_wall_time_ms" : @(totalMilliseconds),
        @"average_dispatch_wall_time_ms" :
            @(totalMilliseconds / double(kMeasuredIterations)),
        @"method" : @"serial wall clock around commit plus waitUntilCompleted; excludes library load, allocation, and warmup"
      }
    });
    return EXIT_SUCCESS;
  }
}
