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
#include "tokenizer.h"
#include "unicode.h"

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

void test_unicode() {
  using namespace unicode;
  auto length = [](std::string_view bytes) { return utf8_sequence_length(bytes, 0); };
  EXPECT(length("A") == 1 && length("\xC3\xA9") == 2 && length("\xE2\x82\xAC") == 3);
  EXPECT(length("\xF0\x9F\x99\x82") == 4 && length("\xF4\x8F\xBF\xBF") == 4);
  EXPECT(length("\xC0\x80") == 0);          // overlong
  EXPECT(length("\xE0\x80\x80") == 0);      // overlong
  EXPECT(length("\xED\xA0\x80") == 0);      // surrogate
  EXPECT(length("\xF4\x90\x80\x80") == 0);  // above U+10FFFF
  EXPECT(length("\xE2\x82") == 0 && length("\x80") == 0 && length("\xFF") == 0);
  EXPECT(throws([] { decode_utf8("a\xED\xBF\xBF" "b"); }));
  EXPECT(decode_utf8("a\xC3\xA9\xF0\x9F\x99\x82") == std::u32string({'a', 0xE9, 0x1F642}));

  EXPECT(category('A') == Lu && category('a') == Ll && category(0x01C5) == Lt && category('7') == Nd);
  EXPECT(category(0x4E00) == Lo && category(0x0301) == Mn && category(0x00A0) == Zs && category(0x0378) == Cn);
  EXPECT(category(0xE000) == Co && category(0x10FFFF) == Cn && category(0x200D) == Cf);
  EXPECT(category(0x13460) == Lo);  // Egyptian Hieroglyphs Extended-A, as the reference's Unicode 16 data says
  EXPECT(is_space('\t') && is_space(0x0B) && is_space(0x85) && is_space(0x3000) && is_space(0x2029));
  EXPECT(!is_space(0x200B) && !is_space(0x180E) && !is_space('x'));

  EXPECT(nfc("e\xCC\x81") == "\xC3\xA9");                            // e + U+0301 -> U+00E9
  EXPECT(nfc("\xE2\x84\xAB") == "\xC3\x85");                        // U+212B ANGSTROM SIGN -> U+00C5
  EXPECT(nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8") == "\xEA\xB0\x81");  // jamo -> U+AC01
  // a + U+0301 (230) + U+0316 (220): reordered, then the acute composes past
  // the lower-class mark.
  EXPECT(nfc("a\xCC\x81\xCC\x96") == "\xC3\xA1\xCC\x96");
  // Dives Akuru U+11935 U+11930 compose in Unicode 13+, but not in the
  // reference normalizer, so they stay apart.
  EXPECT(nfc("\xF0\x91\xA4\xB5\xF0\x91\xA4\xB0") == "\xF0\x91\xA4\xB5\xF0\x91\xA4\xB0");
  EXPECT(std::string(reference()).rfind("tokenizers ", 0) == 0);
}

// Streaming must never emit text that a later token rewrites: for every
// prefix, decode(ids[0, stable_prefix)) must be a prefix of every later
// full decode. Byte-fallback groups are the case that makes this nontrivial.
void test_tokenizer_stable_prefix() {
  std::string vocab = R"("<unk>": 0, "<s>": 1, "a": 258, "\u2581": 259, "b": 260)";
  for (int b = 0; b < 256; ++b) {
    char entry[32];
    std::snprintf(entry, sizeof(entry), ", \"<0x%02X>\": %d", b, b + 2);
    vocab += entry;
  }
  Json root = Json::parse(
      R"({"added_tokens": [{"id": 1, "content": "<s>", "special": true}],
          "model": {"type": "BPE", "unk_token": "<unk>", "byte_fallback": true, "merges": [], "vocab": {)" +
      vocab +
      R"(}}, "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]}})");
  Tokenizer tokenizer = Tokenizer::from_json(root);
  // Bytes of "é" (C3 A9) and "€" (E2 82 AC) plus stray continuation bytes.
  const std::vector<int32_t> pool = {258, 259, 260, 1, 0xC3 + 2, 0xA9 + 2, 0xE2 + 2, 0x82 + 2, 0xAC + 2, 0x91 + 2, 'Z' + 2};
  std::mt19937 rng(5);
  bool ok = true;
  for (int trial = 0; trial < 400 && ok; ++trial) {
    std::vector<int32_t> ids;
    for (int i = 0; i < 12; ++i) ids.push_back(pool[rng() % pool.size()]);
    for (bool skip : {true, false}) {
      for (size_t n = 0; n <= ids.size() && ok; ++n) {
        std::vector<int32_t> head(ids.begin(), ids.begin() + static_cast<long>(n));
        const size_t stable = tokenizer.stable_prefix(head, skip);
        const std::string emitted =
            tokenizer.decode(std::vector<int32_t>(head.begin(), head.begin() + static_cast<long>(stable)), skip);
        for (size_t m = n; m <= ids.size(); ++m) {
          const std::string later =
              tokenizer.decode(std::vector<int32_t>(ids.begin(), ids.begin() + static_cast<long>(m)), skip);
          ok = ok && later.compare(0, emitted.size(), emitted) == 0;
        }
      }
    }
  }
  EXPECT(ok);
  // Without the hold-back, "Z" then 0x91 would rewrite "Z" as two U+FFFD.
  EXPECT(tokenizer.decode({'Z' + 2}, true) == "Z");
  EXPECT(tokenizer.decode({'Z' + 2, 0x91 + 2}, true) == "\xEF\xBF\xBD\xEF\xBF\xBD");
  EXPECT(tokenizer.stable_prefix({258, 'Z' + 2}, true) == 1);
}

}  // namespace

int main() {
  test_json();
  test_half();
  test_quant_kernels();
  test_quant_config();
  test_safetensors();
  test_thread_pool();
  test_unicode();
  test_tokenizer_stable_prefix();
  if (g_failures) {
    std::fprintf(stderr, "%d expectation(s) failed\n", g_failures);
    return 1;
  }
  std::printf("lokahi_engine_tests: all passed\n");
  return 0;
}
