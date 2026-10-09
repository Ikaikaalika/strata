// lokahi: command-line driver for the native engine.
//
//   lokahi info     --model DIR [--backend B]
//   lokahi logits   --model DIR --tokens 2,5,9 [--prefill N] --out FILE
//   lokahi generate --model DIR --tokens 2,5,9 --max-new N [--no-eos-stop]
//   lokahi bench    --model DIR (--tokens ... | --tokens-file F) --max-new N
//                   [--warmups W] [--repetitions R]
//
// Every command prints one JSON object on stdout.
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "model.h"
#include "safetensors.h"

using namespace lokahi;

namespace {

struct Args {
  std::string command;
  std::map<std::string, std::string> values;
  bool has(const std::string& key) const { return values.count(key) != 0; }
  std::string get(const std::string& key, const std::string& fallback = "") const {
    auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  }
  int get_int(const std::string& key, int fallback) const {
    return has(key) ? std::atoi(get(key).c_str()) : fallback;
  }
};

Args parse_args(int argc, char** argv) {
  Args args;
  LK_CHECK(argc >= 2, "usage: lokahi <info|logits|generate|bench> --model DIR ...");
  args.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string key = argv[i];
    LK_CHECK(key.rfind("--", 0) == 0, "unexpected argument " + key);
    key = key.substr(2);
    if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
      args.values[key] = argv[++i];
    } else {
      args.values[key] = "1";
    }
  }
  return args;
}

std::vector<int32_t> parse_tokens(const std::string& text) {
  std::vector<int32_t> tokens;
  std::string item;
  for (char c : text + ",") {
    if (c == ',' || c == ' ' || c == '\n' || c == '\t' || c == '[' || c == ']') {
      if (!item.empty()) tokens.push_back(static_cast<int32_t>(std::stol(item)));
      item.clear();
    } else {
      item.push_back(c);
    }
  }
  return tokens;
}

std::vector<int32_t> prompt_tokens(const Args& args) {
  if (args.has("tokens-file")) return parse_tokens(read_text_file(args.get("tokens-file")));
  LK_CHECK(args.has("tokens"), "--tokens or --tokens-file is required");
  return parse_tokens(args.get("tokens"));
}

LoadOptions load_options(const Args& args) {
  LoadOptions options;
  options.backend = args.get("backend", "auto");
  options.max_context = args.get_int("max-context", 4096);
  options.prefill_chunk = args.get_int("prefill-chunk", 512);
  options.cpu_threads = static_cast<unsigned>(args.get_int("threads", 0));
  return options;
}

std::string json_tokens(const std::vector<int32_t>& tokens) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < tokens.size(); ++i) out << (i ? "," : "") << tokens[i];
  out << "]";
  return out.str();
}

double peak_rss_bytes() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss);  // bytes
#else
  return static_cast<double>(usage.ru_maxrss) * 1024.0;  // kilobytes
#endif
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

int cmd_info(const Args& args) {
  auto model = load_model(args.get("model"), load_options(args));
  std::printf(
      "{\"architecture\":\"%s\",\"backend\":\"%s\",\"vocab_size\":%d,\"max_context\":%d,"
      "\"weight_bytes\":%zu}\n",
      model->architecture(), model->backend(), model->vocab_size(), model->max_context(),
      model->weight_bytes());
  return 0;
}

int cmd_logits(const Args& args) {
  auto model = load_model(args.get("model"), load_options(args));
  std::vector<int32_t> tokens = prompt_tokens(args);
  int prefill = args.get_int("prefill", static_cast<int>(tokens.size()));
  LK_CHECK(prefill >= 1 && prefill <= static_cast<int>(tokens.size()), "--prefill out of range");
  LK_CHECK(args.has("out"), "--out is required");
  std::ofstream out(args.get("out"), std::ios::binary);
  LK_CHECK(out.good(), "cannot write " + args.get("out"));
  const int vocab = model->vocab_size();
  int rows = 0;
  const float* logits = model->prefill(tokens.data(), prefill);
  out.write(reinterpret_cast<const char*>(logits), sizeof(float) * vocab);
  ++rows;
  for (size_t i = static_cast<size_t>(prefill); i < tokens.size(); ++i) {
    logits = model->decode(tokens[i]);
    out.write(reinterpret_cast<const char*>(logits), sizeof(float) * vocab);
    ++rows;
  }
  std::printf("{\"backend\":\"%s\",\"rows\":%d,\"vocab_size\":%d,\"prefill\":%d}\n", model->backend(),
              rows, vocab, prefill);
  return 0;
}

int cmd_generate(const Args& args) {
  auto model = load_model(args.get("model"), load_options(args));
  std::vector<int32_t> tokens = prompt_tokens(args);
  GenerationTiming timing;
  auto generated =
      model->generate_greedy(tokens, args.get_int("max-new", 16), !args.has("no-eos-stop"), &timing);
  std::printf("{\"backend\":\"%s\",\"tokens\":%s,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f}\n",
              model->backend(), json_tokens(generated).c_str(), timing.prefill_seconds,
              timing.decode_seconds);
  return 0;
}

int cmd_bench(const Args& args) {
  LoadOptions options = load_options(args);
  const double load_start = now_seconds();
  auto model = load_model(args.get("model"), options);
  const double load_seconds = now_seconds() - load_start;
  std::vector<int32_t> prompt = prompt_tokens(args);
  const int max_new = args.get_int("max-new", 128);
  const int warmups = args.get_int("warmups", 2);
  const int repetitions = args.get_int("repetitions", 5);
  LK_CHECK(max_new >= 2, "--max-new must be at least 2");

  std::ostringstream reps;
  reps << "[";
  bool invariant = true;
  std::vector<int32_t> reference;
  for (int run = 0; run < warmups + repetitions; ++run) {
    model->reset();
    GenerationTiming timing;
    auto tokens = model->generate_greedy(prompt, max_new, false, &timing);
    LK_CHECK(static_cast<int>(tokens.size()) == max_new, "generation stopped early");
    if (run < warmups) continue;
    if (reference.empty()) reference = tokens;
    invariant = invariant && tokens == reference;
    std::vector<double> intervals;
    for (size_t i = 1; i < timing.token_times.size(); ++i) {
      intervals.push_back(1000.0 * (timing.token_times[i] - timing.token_times[i - 1]));
    }
    const double ttft_ms = 1000.0 * timing.prefill_seconds;
    const double total = timing.prefill_seconds + timing.decode_seconds;
    reps << (run > warmups ? "," : "") << "{\"ttft_ms\":" << ttft_ms
         << ",\"inter_token_latency_ms\":" << median(intervals)
         << ",\"prompt_tokens_per_second\":" << prompt.size() / timing.prefill_seconds
         << ",\"decode_tokens_per_second\":" << (max_new - 1) / timing.decode_seconds
         << ",\"aggregate_tokens_per_second_including_prefill\":" << max_new / total
         << ",\"peak_memory_gb\":" << peak_rss_bytes() / 1e9
         << ",\"token_ids\":" << json_tokens(tokens) << "}";
  }
  reps << "]";
  std::printf(
      "{\"runner\":\"lokahi-native-%s\",\"backend\":\"%s\",\"architecture\":\"%s\","
      "\"load_seconds\":%.6f,\"weight_bytes\":%zu,\"prompt_tokens\":%zu,\"output_tokens\":%d,"
      "\"warmups\":%d,\"repetition_token_invariance\":%s,\"peak_memory_semantics\":"
      "\"process peak resident set size\",\"repetitions\":%s}\n",
      model->backend(), model->backend(), model->architecture(), load_seconds, model->weight_bytes(),
      prompt.size(), max_new, warmups, invariant ? "true" : "false", reps.str().c_str());
  return invariant ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    LK_CHECK(args.has("model"), "--model is required");
    if (args.command == "info") return cmd_info(args);
    if (args.command == "logits") return cmd_logits(args);
    if (args.command == "generate") return cmd_generate(args);
    if (args.command == "bench") return cmd_bench(args);
    fail("unknown command " + args.command);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
}
