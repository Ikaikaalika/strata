// Dependency-free unit tests for the Lokahi engine core.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include "half.h"
#include "json.h"
#include "quant.h"
#include "safetensors.h"
#include "thread_pool.h"

using namespace lokahi;

namespace {

int g_failures = 0;

#define EXPECT(cond)                                                    \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "%s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                     \
    }                                                                   \
  } while (0)

bool throws(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

void test_json() {
  Json value = Json::parse(R"({"a": [1, 2.5, -3e2], "b": {"c": "xé\n"}, "d": null, "e": true})");
  EXPECT(value.at("a").as_array().size() == 3);
  EXPECT(value.at("a").as_array()[2].as_number() == -300.0);
  EXPECT(value.at("b").at("c").as_string() == "x\xc3\xa9\n");
  EXPECT(value.at("d").is_null());
  EXPECT(value.at("e").as_bool());
  EXPECT(value.int_or("missing", 7) == 7);
  EXPECT(throws([] { Json::parse("{\"a\": 1,}"); }));
  EXPECT(throws([] { Json::parse("[1, 2"); }));
  EXPECT(throws([] { Json::parse("{} extra"); }));
}

void test_half() {
  EXPECT(bf16_to_f32(f32_to_bf16(1.0f)) == 1.0f);
  EXPECT(bf16_to_f32(f32_to_bf16(33.941125f)) == 34.0f);  // sqrt(1152) in BF16
  EXPECT(f16_to_f32(0x3C00) == 1.0f);
  EXPECT(f16_to_f32(0xC000) == -2.0f);
  EXPECT(f16_to_f32(0x0001) > 0.0f && f16_to_f32(0x0001) < 1e-7f);  // subnormal
  EXPECT(std::isinf(f16_to_f32(0x7C00)));
}

struct QuantFixture {
  std::vector<uint32_t> words;
  std::vector<uint16_t> scales, biases;
  std::vector<float> dense;  // reference dequantized weights
  Linear linear;
};

QuantFixture make_quant(int rows, int cols, int bits, int group_size, unsigned seed) {
  QuantFixture f;
  std::mt19937 rng(seed);
  const int per_word = 32 / bits;
  const int groups = cols / group_size;
  f.words.assign(static_cast<size_t>(rows) * cols / per_word, 0);
  f.scales.resize(static_cast<size_t>(rows) * groups);
  f.biases.resize(static_cast<size_t>(rows) * groups);
  f.dense.resize(static_cast<size_t>(rows) * cols);
  std::uniform_int_distribution<uint32_t> code(0, (1u << bits) - 1);
  std::uniform_real_distribution<float> uniform(-0.05f, 0.05f);
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      f.scales[static_cast<size_t>(r) * groups + g] = f32_to_bf16(0.01f + std::fabs(uniform(rng)));
      f.biases[static_cast<size_t>(r) * groups + g] = f32_to_bf16(uniform(rng));
    }
    for (int c = 0; c < cols; ++c) {
      uint32_t q = code(rng);
      f.words[(static_cast<size_t>(r) * cols + c) / per_word] |= q << (bits * (c % per_word));
      int g = c / group_size;
      f.dense[static_cast<size_t>(r) * cols + c] =
          bf16_to_f32(f.scales[static_cast<size_t>(r) * groups + g]) * static_cast<float>(q) +
          bf16_to_f32(f.biases[static_cast<size_t>(r) * groups + g]);
    }
  }
  f.linear.rows = rows;
  f.linear.cols = cols;
  f.linear.quant = {group_size, bits};
  f.linear.words = f.words.data();
  f.linear.scales = f.scales.data();
  f.linear.biases = f.biases.data();
  f.linear.scale_dtype = DType::BF16;
  return f;
}

void test_quant_kernels() {
  ThreadPool pool(3);
  for (int bits : {2, 4, 8}) {
    const int rows = 37, cols = 256, group = 64;
    QuantFixture f = make_quant(rows, cols, bits, group, 1234u + bits);
    std::vector<float> row(cols);
    dequantize_row(f.linear, 5, row.data());
    for (int c = 0; c < cols; ++c) EXPECT(row[c] == f.dense[5 * cols + c]);

    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int n = 3;
    std::vector<float> x(static_cast<size_t>(n) * cols);
    for (float& v : x) v = normal(rng);
    std::vector<float> y(static_cast<size_t>(n) * rows), yv(rows);
    matmul(f.linear, x.data(), n, y.data(), pool);
    matvec(f.linear, x.data() + cols, yv.data(), pool);
    for (int t = 0; t < n; ++t) {
      for (int r = 0; r < rows; ++r) {
        double ref = 0.0;
        for (int c = 0; c < cols; ++c) ref += static_cast<double>(f.dense[r * cols + c]) * x[t * cols + c];
        EXPECT(std::fabs(y[t * rows + r] - ref) < 1e-3 * (1.0 + std::fabs(ref)));
        if (t == 1) EXPECT(std::fabs(yv[r] - ref) < 1e-3 * (1.0 + std::fabs(ref)));
      }
    }
  }
}

void test_quant_config() {
  Json config = Json::parse(R"({"quantization": {"group_size": 64, "bits": 4,
      "model.layers.0.mlp.up_proj": {"bits": 8}, "model.norm": false}})");
  EXPECT(quant_spec_for(config, "model.layers.1.mlp.up_proj").bits == 4);
  EXPECT(quant_spec_for(config, "model.layers.0.mlp.up_proj").bits == 8);
  EXPECT(quant_spec_for(config, "model.layers.0.mlp.up_proj").group_size == 64);
  EXPECT(!quant_spec_for(config, "model.norm").quantized());
  EXPECT(!quant_spec_for(Json::parse("{}"), "x").quantized());
  EXPECT(throws([] {
    quant_spec_for(Json::parse(R"({"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}})"), "x");
  }));
}

std::string write_safetensors(const std::string& header, const std::vector<uint8_t>& payload) {
  char path[] = "/tmp/lokahi_test_XXXXXX";
  int fd = mkstemp(path);
  EXPECT(fd >= 0);
  close(fd);
  std::string dir = std::string(path) + ".d";
  std::string cmd = "mkdir -p " + dir;
  EXPECT(std::system(cmd.c_str()) == 0);
  std::ofstream out(dir + "/model.safetensors", std::ios::binary);
  uint64_t len = header.size();
  out.write(reinterpret_cast<const char*>(&len), 8);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
  unlink(path);
  return dir;
}

void test_safetensors() {
  std::vector<uint8_t> payload(8 + 4);
  float values[2] = {1.5f, -2.0f};
  std::memcpy(payload.data(), values, 8);
  uint16_t halfs[2] = {0x3C00, 0x4000};
  std::memcpy(payload.data() + 8, halfs, 4);
  std::string dir = write_safetensors(
      R"({"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
      R"("b":{"dtype":"F16","shape":[1,2],"data_offsets":[8,12]}})",
      payload);
  Checkpoint ck = Checkpoint::open_directory(dir);
  EXPECT(ck.tensors().size() == 2);
  EXPECT(ck.get("a").as<float>()[1] == -2.0f);
  EXPECT(ck.get("b").shape.size() == 2 && ck.get("b").dim(1) == 2);
  float out[2];
  load_vector(ck, "b", 2, out);
  EXPECT(out[0] == 1.0f && out[1] == 2.0f);
  EXPECT(throws([&] { ck.get("missing"); }));

  std::string bad = write_safetensors(R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})", payload);
  EXPECT(throws([&] { Checkpoint::open_directory(bad); }));
}

void test_thread_pool() {
  ThreadPool pool(4);
  std::vector<int> hits(1000, 0);
  for (int round = 0; round < 50; ++round) {
    pool.parallel_for(hits.size(), [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) ++hits[i];
    });
  }
  bool all = true;
  for (int h : hits) all = all && h == 50;
  EXPECT(all);
}

}  // namespace

int main() {
  test_json();
  test_half();
  test_quant_kernels();
  test_quant_config();
  test_safetensors();
  test_thread_pool();
  if (g_failures) {
    std::fprintf(stderr, "%d expectation(s) failed\n", g_failures);
    return 1;
  }
  std::printf("lokahi_engine_tests: all passed\n");
  return 0;
}
