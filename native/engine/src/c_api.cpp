#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lokahi/lokahi.h"
#include "model.h"

struct lokahi_model {
  std::unique_ptr<lokahi::Model> model;
};

namespace {

thread_local std::string g_last_error;

template <typename Fn>
int guarded(Fn&& fn) {
  try {
    fn();
    g_last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    g_last_error = error.what();
  } catch (...) {
    g_last_error = "lokahi: unknown error";
  }
  return 1;
}

}  // namespace

extern "C" {

int32_t lokahi_abi_version(void) { return LOKAHI_ABI_VERSION; }

const char* lokahi_last_error(void) { return g_last_error.c_str(); }

int lokahi_model_load(const char* directory, const lokahi_load_options* options,
                      lokahi_model** out_model) {
  return guarded([&] {
    if (directory == nullptr || out_model == nullptr) throw std::invalid_argument("null argument");
    lokahi::LoadOptions opts;
    if (options != nullptr) {
      if (options->backend != nullptr) opts.backend = options->backend;
      if (options->max_context > 0) opts.max_context = options->max_context;
      if (options->prefill_chunk > 0) opts.prefill_chunk = options->prefill_chunk;
      if (options->cpu_threads > 0) opts.cpu_threads = static_cast<unsigned>(options->cpu_threads);
    }
    auto handle = std::make_unique<lokahi_model>();
    handle->model = lokahi::load_model(directory, opts);
    *out_model = handle.release();
  });
}

void lokahi_model_free(lokahi_model* model) { delete model; }

const char* lokahi_model_backend(const lokahi_model* model) {
  return model ? model->model->backend() : "";
}

int32_t lokahi_model_vocab_size(const lokahi_model* model) {
  return model ? model->model->vocab_size() : 0;
}

int32_t lokahi_model_position(const lokahi_model* model) {
  return model ? model->model->position() : 0;
}

void lokahi_model_reset(lokahi_model* model) {
  if (model) model->model->reset();
}

int lokahi_prefill(lokahi_model* model, const int32_t* tokens, int32_t count, float* logits_out) {
  return guarded([&] {
    if (model == nullptr || tokens == nullptr || count <= 0) throw std::invalid_argument("bad arguments");
    const float* logits = model->model->prefill(tokens, count);
    if (logits_out) std::memcpy(logits_out, logits, sizeof(float) * model->model->vocab_size());
  });
}

int lokahi_decode(lokahi_model* model, int32_t token, float* logits_out) {
  return guarded([&] {
    if (model == nullptr) throw std::invalid_argument("null model");
    const float* logits = model->model->decode(token);
    if (logits_out) std::memcpy(logits_out, logits, sizeof(float) * model->model->vocab_size());
  });
}

int lokahi_generate_greedy(lokahi_model* model, const int32_t* prompt, int32_t prompt_count,
                           int32_t max_new, int32_t stop_at_eos, int32_t* out_tokens,
                           int32_t* out_count, lokahi_timing* timing) {
  return guarded([&] {
    if (model == nullptr || prompt == nullptr || prompt_count <= 0 || out_tokens == nullptr ||
        out_count == nullptr) {
      throw std::invalid_argument("bad arguments");
    }
    lokahi::GenerationTiming t;
    std::vector<int32_t> input(prompt, prompt + prompt_count);
    auto tokens = model->model->generate_greedy(input, max_new, stop_at_eos != 0, &t);
    std::copy(tokens.begin(), tokens.end(), out_tokens);
    *out_count = static_cast<int32_t>(tokens.size());
    if (timing) {
      timing->prefill_seconds = t.prefill_seconds;
      timing->decode_seconds = t.decode_seconds;
      timing->generated_tokens = static_cast<int32_t>(tokens.size());
    }
  });
}

}  // extern "C"
