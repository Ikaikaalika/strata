#include "model.h"

#include <algorithm>

#include "common.h"
#include "gemma3.h"
#include "json.h"
#include "safetensors.h"

#if defined(LOKAHI_WITH_METAL)
#include "metal/metal_gemma3.h"
#endif

namespace lokahi {

int32_t argmax(const float* values, int count) {
  int32_t best = 0;
  for (int i = 1; i < count; ++i) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

std::vector<int32_t> Model::generate_greedy(const std::vector<int32_t>& prompt, int max_new,
                                            bool stop_at_eos, GenerationTiming* timing) {
  std::vector<int32_t> out;
  if (max_new <= 0) return out;
  const auto& eos = eos_token_ids();
  const double start = now_seconds();
  int32_t token = argmax(prefill(prompt.data(), static_cast<int>(prompt.size())), vocab_size());
  const double first = now_seconds();
  out.push_back(token);
  if (timing) {
    timing->prefill_seconds = first - start;
    timing->token_times = {first};
  }
  while (static_cast<int>(out.size()) < max_new) {
    if (stop_at_eos && std::find(eos.begin(), eos.end(), token) != eos.end()) break;
    token = argmax(decode(token), vocab_size());
    out.push_back(token);
    if (timing) timing->token_times.push_back(now_seconds());
  }
  if (timing) timing->decode_seconds = timing->token_times.back() - first;
  return out;
}

std::unique_ptr<Model> load_model(const std::string& directory, const LoadOptions& options) {
  Json config = Json::parse(read_text_file(directory + "/config.json"));
  LK_CHECK(is_gemma3_config(config),
           "unsupported architecture '" + config.string_or("model_type", "?") +
               "'; this build supports Gemma 3 text models");
  auto weights = Gemma3Weights::load(directory);
  std::string backend = options.backend;
#if defined(LOKAHI_WITH_METAL)
  if (backend == "auto") backend = metal_available() ? "metal" : "cpu";
  if (backend == "metal") return make_gemma3_metal(std::move(weights), options);
#else
  if (backend == "auto") backend = "cpu";
  LK_CHECK(backend != "metal", "this build has no Metal backend");
#endif
  LK_CHECK(backend == "cpu", "unknown backend '" + backend + "'");
  return make_gemma3_cpu(std::move(weights), options);
}

}  // namespace lokahi
