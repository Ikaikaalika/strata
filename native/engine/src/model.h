// Backend-neutral model interface used by the C ABI and the CLI.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lokahi {

struct LoadOptions {
  std::string backend = "auto";  // auto | cpu | metal
  int max_context = 4096;
  int prefill_chunk = 512;
  unsigned cpu_threads = 0;  // 0: hardware concurrency
};

struct GenerationTiming {
  double prefill_seconds = 0.0;      // admitted prompt to first token available
  double decode_seconds = 0.0;       // first token available to last token available
  std::vector<double> token_times;   // host time each generated token became available
};

class Model {
 public:
  virtual ~Model() = default;

  virtual const char* backend() const = 0;
  virtual const char* architecture() const = 0;
  virtual int vocab_size() const = 0;
  virtual int max_context() const = 0;
  virtual int position() const = 0;
  virtual size_t weight_bytes() const = 0;
  virtual const std::vector<int32_t>& eos_token_ids() const = 0;

  virtual void reset() = 0;

  // Append `n` tokens at the current position. Returns the logits of the last
  // token; the pointer stays valid until the next call.
  virtual const float* prefill(const int32_t* tokens, int n) = 0;
  virtual const float* decode(int32_t token) = 0;

  // Greedy generation. Backends may keep tokens on the device and pipeline
  // steps; the default runs prefill + decode with host-side argmax.
  virtual std::vector<int32_t> generate_greedy(const std::vector<int32_t>& prompt, int max_new,
                                               bool stop_at_eos, GenerationTiming* timing);
};

int32_t argmax(const float* values, int count);

std::unique_ptr<Model> load_model(const std::string& directory, const LoadOptions& options);

}  // namespace lokahi
