#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "affine_q4_contract.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mount.h>
#include <sys/sysctl.h>
#include <vector>

namespace {
using namespace strata::q4;
constexpr double kAtol = 0.0005, kRtol = 0.005;
constexpr uint64_t kSSDReserve = 40ull * 1024 * 1024 * 1024;

void Emit(NSDictionary *value) {
  NSError *error = nil;
  NSData *json = [NSJSONSerialization dataWithJSONObject:value
      options:NSJSONWritingSortedKeys error:&error];
  if (!json) { std::fprintf(stderr, "JSON serialization failed\n"); return; }
  std::fwrite(json.bytes, 1, json.length, stdout);
  std::fputc('\n', stdout);
}

int Fail(NSString *reason, int code = 1) {
  Emit(@{@"schema_version": @1, @"operation": @"affine_q4_projection",
         @"execution_verified": @NO, @"promotion_eligible": @NO,
         @"failure_reason": reason});
  return code;
}

NSString *Env(const char *key) {
  const char *value = std::getenv(key);
  return value ? [NSString stringWithUTF8String:value] : @"unknown";
}

NSString *Chip() {
  char value[256]{}; size_t size = sizeof(value);
  if (sysctlbyname("machdep.cpu.brand_string", value, &size, nullptr, 0) != 0)
    return @"unknown";
  return [NSString stringWithUTF8String:value];
}

struct Fixture {
  Shape shape;
  Footprint sizes{};
  std::vector<Half> input, scales, biases;
  std::vector<uint32_t> packed;
  std::vector<float> expected;
  explicit Fixture(Shape s) : shape(s) {
    if (!Validate(shape, &sizes)) throw std::runtime_error("invalid fixture shape");
    input.resize(sizes.input_elements); scales.resize(sizes.group_elements);
    biases.resize(sizes.group_elements); packed.resize(sizes.packed_words);
    expected.resize(sizes.output_elements);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = Half((int((i * 17) % 113) - 56) / 128.0f);
    for (size_t i = 0; i < scales.size(); ++i) {
      scales[i] = Half((i % 7 + 1) / 1024.0f);
      biases[i] = Half(-7.0f * float(scales[i]));
    }
    for (uint32_t out = 0; out < shape.output_width; ++out) {
      for (uint32_t word = 0; word < shape.input_width / 8; ++word) {
        uint32_t value = 0;
        for (uint32_t bit = 0; bit < 8; ++bit)
          value |= (((word * 8 + bit) * 5 + out * 3) % 16) << (4 * bit);
        packed[size_t(out) * (shape.input_width / 8) + word] = value;
      }
    }
    if (!Reference(shape, input.data(), input.size(), packed.data(), packed.size(),
                   scales.data(), biases.data(), scales.size(), expected.data(), expected.size()))
      throw std::runtime_error("CPU oracle failed");
    for (float &value : expected) value = float(Half(value));
  }
};

NSDictionary *Error(const Half *actual, const std::vector<float> &expected) {
  double max_abs = 0, max_scaled = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!std::isfinite(float(actual[i])) || !std::isfinite(expected[i]))
      return @{@"passed": @NO, @"reason": @"nonfinite or unwritten output"};
    const double delta = std::abs(double(actual[i]) - expected[i]);
    max_abs = std::max(max_abs, delta);
    max_scaled = std::max(max_scaled, delta / (kAtol + kRtol * std::abs(expected[i])));
  }
  return @{@"passed": @(max_scaled <= 1), @"max_abs": @(max_abs),
           @"max_scaled": @(max_scaled), @"atol": @(kAtol), @"rtol": @(kRtol)};
}
}  // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 3 || (std::strcmp(argv[1], "--check") && std::strcmp(argv[1], "--benchmark")))
      return Fail(@"usage: strata-q4-bench --check|--benchmark /absolute/path/library.metallib");
    const bool benchmark = !std::strcmp(argv[1], "--benchmark");
    struct statfs disk{};
    if (statfs(NSHomeDirectory().fileSystemRepresentation, &disk) != 0)
      return Fail(@"cannot verify internal SSD reserve", 78);
    const uint64_t available = uint64_t(disk.f_bavail) * uint64_t(disk.f_bsize);
    if (available < kSSDReserve) {
      Emit(@{@"schema_version": @1, @"operation": @"affine_q4_projection",
             @"execution_verified": @NO, @"promotion_eligible": @NO,
             @"failure_reason": @"internal SSD free-space reserve is below 40 GiB",
             @"available_bytes": @(available), @"required_bytes": @(kSSDReserve)});
      return 78;
    }
    try {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) return Fail(@"Metal device unavailable");
      NSError *error = nil;
      id<MTLLibrary> library = [device newLibraryWithURL:
          [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]] error:&error];
      if (!library) return Fail(error.localizedDescription ?: @"library load failed");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (!queue) return Fail(@"command queue creation failed");
      NSArray<NSString *> *names = @[@"affine_q4_scalar", @"affine_q4_simd", @"affine_q4_tiled"];
      NSMutableArray<id<MTLComputePipelineState>> *pipelines = [NSMutableArray array];
      for (NSString *name in names) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (!function) return Fail(@"missing bounded Q4 kernel");
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) return Fail(error.localizedDescription ?: @"pipeline creation failed");
        if (pipeline.maxTotalThreadsPerThreadgroup < 128)
          return Fail(@"device cannot dispatch the declared bounded geometry");
        [pipelines addObject:pipeline];
      }
      NSMutableArray *cells = [NSMutableArray array];
      for (Shape shape : {Shape{1, 1024, 128}, Shape{7, 128, 17}, Shape{64, 1024, 128}}) {
        Fixture fixture(shape);
        const MTLResourceOptions options = MTLResourceStorageModeShared;
        id<MTLBuffer> x = [device newBufferWithBytes:fixture.input.data() length:fixture.input.size() * sizeof(Half) options:options];
        id<MTLBuffer> q = [device newBufferWithBytes:fixture.packed.data() length:fixture.packed.size() * sizeof(uint32_t) options:options];
        id<MTLBuffer> s = [device newBufferWithBytes:fixture.scales.data() length:fixture.scales.size() * sizeof(Half) options:options];
        id<MTLBuffer> b = [device newBufferWithBytes:fixture.biases.data() length:fixture.biases.size() * sizeof(Half) options:options];
        id<MTLBuffer> y = [device newBufferWithLength:fixture.expected.size() * sizeof(Half) options:options];
        if (!x || !q || !s || !b || !y) return Fail(@"bounded buffer allocation failed");
        NSMutableArray *variants = [NSMutableArray array];
        for (NSString *name in names)
          [variants addObject:[@{@"kernel": name, @"wall_ms": [NSMutableArray array],
                                @"gpu_ms": [NSMutableArray array]} mutableCopy]];
        const uint32_t warmups = benchmark ? 5 : 0, iterations = benchmark ? 20 : 1;
        for (uint32_t step = 0; step < warmups + iterations; ++step) {
          for (uint32_t order = 0; order < 3; ++order) {
            @autoreleasepool {
              const uint32_t v = (step + order) % 3;
              std::fill_n(static_cast<Half *>(y.contents), fixture.expected.size(),
                          Half(std::numeric_limits<float>::quiet_NaN()));
              const auto started = std::chrono::steady_clock::now();
              id<MTLCommandBuffer> command = [queue commandBuffer];
              if (!command) return Fail(@"command buffer creation failed");
              id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
              if (!encoder) return Fail(@"encoder creation failed");
              [encoder setComputePipelineState:pipelines[v]];
              [encoder setBuffer:x offset:0 atIndex:0]; [encoder setBuffer:q offset:0 atIndex:1];
              [encoder setBuffer:s offset:0 atIndex:2]; [encoder setBuffer:b offset:0 atIndex:3];
              [encoder setBuffer:y offset:0 atIndex:4];
              [encoder setBytes:&shape.rows length:4 atIndex:5];
              [encoder setBytes:&shape.input_width length:4 atIndex:6];
              [encoder setBytes:&shape.output_width length:4 atIndex:7];
              if (v == 0) {
                [encoder dispatchThreads:MTLSizeMake(shape.output_width, shape.rows, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
              } else if (v == 1) {
                [encoder dispatchThreadgroups:MTLSizeMake(shape.output_width, shape.rows, 1)
                    threadsPerThreadgroup:MTLSizeMake(pipelines[v].threadExecutionWidth, 1, 1)];
              } else {
                [encoder dispatchThreadgroups:MTLSizeMake((shape.output_width + 15) / 16, (shape.rows + 7) / 8, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
              }
              [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
              const auto ended = std::chrono::steady_clock::now();
              if (command.status != MTLCommandBufferStatusCompleted)
                return Fail(command.error.localizedDescription ?: @"Metal dispatch failed");
              NSDictionary *numerical = Error(static_cast<const Half *>(y.contents), fixture.expected);
              if (![numerical[@"passed"] boolValue]) return Fail(@"independent FP64 CPU oracle gate failed");
              variants[v][@"numerical"] = numerical;
              if (benchmark && step >= warmups) {
                [variants[v][@"wall_ms"] addObject:@(std::chrono::duration<double, std::milli>(ended - started).count())];
                id gpu = command.GPUStartTime > 0 && command.GPUEndTime > command.GPUStartTime
                    ? @(1000 * (command.GPUEndTime - command.GPUStartTime)) : [NSNull null];
                [variants[v][@"gpu_ms"] addObject:gpu];
              }
            }
          }
        }
        [cells addObject:@{@"shape": @[@(shape.rows), @(shape.input_width), @(shape.output_width)],
                           @"phase": shape.rows == 1 ? @"decode_projection" : @"prefill_projection",
                           @"resident_bytes": @(fixture.sizes.total_bytes), @"variants": variants}];
      }
      Emit(@{@"schema_version": @1, @"evidence_kind": benchmark ? @"hardware_component" : @"correctness",
             @"operation": @"affine_q4_projection", @"fixture_id": @"integer_affine_q4_v1",
             @"execution_verified": @YES, @"promotion_eligible": @NO, @"ssd_offload": @"disabled",
             @"ane_dispatches": @0, @"chip": Chip(), @"device": device.name,
             @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
             @"dtype": @"fp16_input_scale_bias_output_fp32_accumulation", @"bits": @4, @"group_size": @64,
             @"source_sha256": Env("STRATA_Q4_SOURCE_SHA256"),
             @"contract_sha256": Env("STRATA_Q4_CONTRACT_SHA256"),
             @"host_sha256": Env("STRATA_Q4_HOST_SHA256"),
             @"binary_sha256": Env("STRATA_Q4_BINARY_SHA256"),
             @"metallib_sha256": Env("STRATA_Q4_METALLIB_SHA256"),
             @"compiler_flags": @"metal3.1 -O3 -fno-fast-math; c++20 -O3 -Wall -Wextra -Werror",
             @"warmups": @(benchmark ? 5 : 0), @"iterations": @(benchmark ? 20 : 0),
             @"timing_boundary": @"encode+commit+wait; excludes compile/load/allocation/poison/readback verification",
             @"memory_boundary": @"declared device tensor buffers only; not process peak footprint",
             @"energy_joules": [NSNull null], @"cells": cells});
      return 0;
    } catch (const std::exception &error) {
      return Fail([NSString stringWithUTF8String:error.what()]);
    }
  }
}
