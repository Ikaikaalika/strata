// Gemma 3 text decoder: configuration, weight binding, and CPU backend.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "model.h"
#include "quant.h"
#include "safetensors.h"

namespace lokahi {

struct Gemma3Config {
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1e6f;
  float rope_local_base_freq = 1e4f;
  float rope_linear_factor = 1.0f;
  float query_pre_attn_scalar = 256.0f;
  int sliding_window = 4096;
  std::vector<bool> layer_is_sliding;
  float final_logit_softcapping = 0.0f;  // 0 disables
  float attn_logit_softcapping = 0.0f;
  int bos_token_id = 2;
  std::vector<int32_t> eos_token_ids;
  std::string weight_prefix;  // "" or "language_model."

  static Gemma3Config from_json(const Json& raw);
  float embed_scale() const;  // sqrt(hidden) rounded to BF16
  int kv_capacity(int layer, int max_context) const;
};

bool is_gemma3_config(const Json& raw);

struct Gemma3Layer {
  std::vector<float> input_norm, post_attn_norm, pre_ff_norm, post_ff_norm, q_norm, k_norm;
  Linear q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;
};

// Weights bound to a mapped checkpoint (zero-copy for matrices).
struct Gemma3Weights {
  Gemma3Config config;
  Json raw_config;
  std::shared_ptr<Checkpoint> checkpoint;
  Linear embed;
  Linear lm_head;
  std::vector<float> final_norm;
  std::vector<Gemma3Layer> layers;
  bool tied_lm_head = false;

  static std::shared_ptr<Gemma3Weights> load(const std::string& directory);
  size_t matrix_bytes() const;
};

std::unique_ptr<Model> make_gemma3_cpu(std::shared_ptr<Gemma3Weights> weights,
                                       const LoadOptions& options);

}  // namespace lokahi
