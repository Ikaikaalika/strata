#include "gemma3.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common.h"
#include "half.h"
#include "thread_pool.h"

namespace lokahi {

static const Json& text_config(const Json& raw) {
  const Json* text = raw.find("text_config");
  return text != nullptr && text->is_object() ? *text : raw;
}

bool is_gemma3_config(const Json& raw) {
  std::string type = text_config(raw).string_or("model_type", raw.string_or("model_type", ""));
  return type == "gemma3_text" || type == "gemma3";
}

Gemma3Config Gemma3Config::from_json(const Json& raw) {
  const Json& cfg = text_config(raw);
  Gemma3Config c;
  c.vocab_size = static_cast<int>(cfg.at("vocab_size").as_int());
  c.hidden_size = static_cast<int>(cfg.at("hidden_size").as_int());
  c.intermediate_size = static_cast<int>(cfg.at("intermediate_size").as_int());
  c.num_layers = static_cast<int>(cfg.at("num_hidden_layers").as_int());
  c.num_heads = static_cast<int>(cfg.at("num_attention_heads").as_int());
  c.num_kv_heads = static_cast<int>(cfg.at("num_key_value_heads").as_int());
  c.head_dim = static_cast<int>(cfg.int_or("head_dim", 256));
  c.rms_norm_eps = static_cast<float>(cfg.number_or("rms_norm_eps", 1e-6));
  c.rope_theta = static_cast<float>(cfg.number_or("rope_theta", 1e6));
  c.rope_local_base_freq = static_cast<float>(cfg.number_or("rope_local_base_freq", 1e4));
  c.query_pre_attn_scalar = static_cast<float>(cfg.number_or("query_pre_attn_scalar", c.head_dim));
  c.sliding_window = static_cast<int>(cfg.int_or("sliding_window", 4096));
  if (const Json* scaling = cfg.find("rope_scaling"); scaling != nullptr && scaling->is_object()) {
    std::string kind = scaling->string_or("rope_type", scaling->string_or("type", "default"));
    if (kind == "linear") {
      c.rope_linear_factor = static_cast<float>(scaling->at("factor").as_number());
    } else {
      LK_CHECK(kind == "default", "gemma3: rope_scaling type '" + kind + "' is not supported");
    }
  }
  std::string activation = cfg.string_or("hidden_activation", "gelu_pytorch_tanh");
  LK_CHECK(activation == "gelu_pytorch_tanh", "gemma3: unsupported activation " + activation);
  if (const Json* types = cfg.find("layer_types"); types != nullptr && types->is_array()) {
    for (const Json& kind : types->as_array()) {
      c.layer_is_sliding.push_back(kind.as_string() == "sliding_attention");
    }
  } else {
    int pattern = static_cast<int>(cfg.int_or("sliding_window_pattern", 6));
    for (int i = 0; i < c.num_layers; ++i) c.layer_is_sliding.push_back((i + 1) % pattern != 0);
  }
  LK_CHECK(static_cast<int>(c.layer_is_sliding.size()) == c.num_layers,
           "gemma3: layer_types length mismatch");
  if (const Json* cap = cfg.find("final_logit_softcapping"); cap && cap->is_number()) {
    c.final_logit_softcapping = static_cast<float>(cap->as_number());
  }
  if (const Json* cap = cfg.find("attn_logit_softcapping"); cap && cap->is_number()) {
    c.attn_logit_softcapping = static_cast<float>(cap->as_number());
  }
  c.bos_token_id = static_cast<int>(raw.int_or("bos_token_id", cfg.int_or("bos_token_id", 2)));
  const Json* eos = raw.find("eos_token_id");
  if (eos == nullptr) eos = cfg.find("eos_token_id");
  if (eos != nullptr && eos->is_array()) {
    for (const Json& id : eos->as_array()) c.eos_token_ids.push_back(static_cast<int32_t>(id.as_int()));
  } else if (eos != nullptr && eos->is_number()) {
    c.eos_token_ids.push_back(static_cast<int32_t>(eos->as_int()));
  }
  LK_CHECK(c.num_heads % c.num_kv_heads == 0, "gemma3: heads must be a multiple of kv heads");
  LK_CHECK(c.head_dim % 2 == 0, "gemma3: head_dim must be even");
  return c;
}

float Gemma3Config::embed_scale() const {
  return bf16_to_f32(f32_to_bf16(std::sqrt(static_cast<float>(hidden_size))));
}

int Gemma3Config::kv_capacity(int layer, int max_context) const {
  return layer_is_sliding[static_cast<size_t>(layer)] ? std::min(sliding_window, max_context)
                                                       : max_context;
}

std::shared_ptr<Gemma3Weights> Gemma3Weights::load(const std::string& directory) {
  auto w = std::make_shared<Gemma3Weights>();
  w->raw_config = Json::parse(read_text_file(directory + "/config.json"));
  LK_CHECK(is_gemma3_config(w->raw_config), "model is not a Gemma 3 checkpoint");
  w->checkpoint = std::make_shared<Checkpoint>(Checkpoint::open_directory(directory));
  const Checkpoint& ck = *w->checkpoint;
  Gemma3Config& c = w->config;
  c = Gemma3Config::from_json(w->raw_config);
  if (ck.contains("language_model.model.embed_tokens.weight")) c.weight_prefix = "language_model.";

  const std::string& p = c.weight_prefix;
  auto linear = [&](const std::string& module) {
    return bind_linear(ck, w->raw_config, p + module, p + module);
  };
  auto vec = [&](const std::string& name, int64_t size) {
    std::vector<float> out(static_cast<size_t>(size));
    load_vector(ck, p + name + ".weight", size, out.data());
    return out;
  };

  w->embed = linear("model.embed_tokens");
  LK_CHECK(w->embed.rows == c.vocab_size && w->embed.cols == c.hidden_size,
           "gemma3: embed_tokens shape mismatch");
  if (ck.contains(p + "lm_head.weight")) {
    w->lm_head = linear("lm_head");
  } else if (ck.contains("lm_head.weight")) {
    w->lm_head = bind_linear(ck, w->raw_config, "lm_head", "lm_head");
  } else {
    w->lm_head = w->embed;
    w->tied_lm_head = true;
  }
  w->final_norm = vec("model.norm", c.hidden_size);
  const int64_t q_out = static_cast<int64_t>(c.num_heads) * c.head_dim;
  const int64_t kv_out = static_cast<int64_t>(c.num_kv_heads) * c.head_dim;
  for (int i = 0; i < c.num_layers; ++i) {
    const std::string base = "model.layers." + std::to_string(i);
    Gemma3Layer layer;
    layer.input_norm = vec(base + ".input_layernorm", c.hidden_size);
    layer.post_attn_norm = vec(base + ".post_attention_layernorm", c.hidden_size);
    layer.pre_ff_norm = vec(base + ".pre_feedforward_layernorm", c.hidden_size);
    layer.post_ff_norm = vec(base + ".post_feedforward_layernorm", c.hidden_size);
    layer.q_norm = vec(base + ".self_attn.q_norm", c.head_dim);
    layer.k_norm = vec(base + ".self_attn.k_norm", c.head_dim);
    layer.q_proj = linear(base + ".self_attn.q_proj");
    layer.k_proj = linear(base + ".self_attn.k_proj");
    layer.v_proj = linear(base + ".self_attn.v_proj");
    layer.o_proj = linear(base + ".self_attn.o_proj");
    layer.gate_proj = linear(base + ".mlp.gate_proj");
    layer.up_proj = linear(base + ".mlp.up_proj");
    layer.down_proj = linear(base + ".mlp.down_proj");
    LK_CHECK(layer.q_proj.rows == q_out && layer.q_proj.cols == c.hidden_size, base + ": q_proj shape");
    LK_CHECK(layer.k_proj.rows == kv_out && layer.v_proj.rows == kv_out, base + ": k/v shape");
    LK_CHECK(layer.o_proj.rows == c.hidden_size && layer.o_proj.cols == q_out, base + ": o_proj shape");
    LK_CHECK(layer.gate_proj.rows == c.intermediate_size && layer.up_proj.rows == c.intermediate_size &&
                 layer.down_proj.cols == c.intermediate_size,
             base + ": mlp shape");
    w->layers.push_back(std::move(layer));
  }
  return w;
}

size_t Gemma3Weights::matrix_bytes() const {
  size_t total = embed.weight_bytes() + (tied_lm_head ? 0 : lm_head.weight_bytes());
  for (const auto& l : layers) {
    total += l.q_proj.weight_bytes() + l.k_proj.weight_bytes() + l.v_proj.weight_bytes() +
             l.o_proj.weight_bytes() + l.gate_proj.weight_bytes() + l.up_proj.weight_bytes() +
             l.down_proj.weight_bytes();
  }
  return total;
}

// ---------------------------------------------------------------------------
// CPU backend
// ---------------------------------------------------------------------------

namespace {

void rms_norm(const float* x, const float* weight, int64_t size, float eps, float* out) {
  double sum = 0.0;
  for (int64_t i = 0; i < size; ++i) sum += static_cast<double>(x[i]) * x[i];
  const float inv = static_cast<float>(1.0 / std::sqrt(sum / static_cast<double>(size) + eps));
  for (int64_t i = 0; i < size; ++i) out[i] = x[i] * inv * (1.0f + weight[i]);
}

float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2 / pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

class Gemma3Cpu final : public Model {
 public:
  Gemma3Cpu(std::shared_ptr<Gemma3Weights> weights, const LoadOptions& options)
      : w_(std::move(weights)), c_(w_->config), pool_(options.cpu_threads),
        max_context_(options.max_context), chunk_(std::max(1, options.prefill_chunk)) {
    LK_CHECK(max_context_ > 0, "max_context must be positive");
    kv_k_.resize(static_cast<size_t>(c_.num_layers));
    kv_v_.resize(static_cast<size_t>(c_.num_layers));
    for (int l = 0; l < c_.num_layers; ++l) {
      size_t size = static_cast<size_t>(c_.num_kv_heads) * c_.kv_capacity(l, max_context_) * c_.head_dim;
      kv_k_[static_cast<size_t>(l)].assign(size, 0.0f);
      kv_v_[static_cast<size_t>(l)].assign(size, 0.0f);
    }
    logits_.resize(static_cast<size_t>(c_.vocab_size));
    // RoPE inverse frequencies for local (sliding) and global layers.
    const int half = c_.head_dim / 2;
    inv_freq_local_.resize(static_cast<size_t>(half));
    inv_freq_global_.resize(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i) {
      double exponent = static_cast<double>(2 * i) / c_.head_dim;
      inv_freq_local_[static_cast<size_t>(i)] = 1.0 / std::pow(c_.rope_local_base_freq, exponent);
      inv_freq_global_[static_cast<size_t>(i)] =
          1.0 / std::pow(c_.rope_theta, exponent) / c_.rope_linear_factor;
    }
  }

  const char* backend() const override { return "cpu"; }
  const char* architecture() const override { return "gemma3"; }
  int vocab_size() const override { return c_.vocab_size; }
  int max_context() const override { return max_context_; }
  int position() const override { return position_; }
  size_t weight_bytes() const override { return w_->matrix_bytes(); }
  const std::vector<int32_t>& eos_token_ids() const override { return c_.eos_token_ids; }
  void reset() override { position_ = 0; }

  const float* prefill(const int32_t* tokens, int n) override {
    LK_CHECK(n > 0, "prefill requires at least one token");
    for (int start = 0; start < n; start += chunk_) {
      int count = std::min(chunk_, n - start);
      forward(tokens + start, count, start + count == n);
    }
    return logits_.data();
  }

  const float* decode(int32_t token) override {
    forward(&token, 1, true);
    return logits_.data();
  }

 private:
  const float* key_ptr(int layer, int kv, int pos, int chunk_start, const float* chunk_k) const {
    if (c_.layer_is_sliding[static_cast<size_t>(layer)] && pos >= chunk_start) {
      return chunk_k + (static_cast<size_t>(pos - chunk_start) * c_.num_kv_heads + kv) * c_.head_dim;
    }
    int capacity = c_.kv_capacity(layer, max_context_);
    int slot = pos % capacity;
    return kv_k_[static_cast<size_t>(layer)].data() +
           (static_cast<size_t>(kv) * capacity + slot) * c_.head_dim;
  }

  const float* value_ptr(int layer, int kv, int pos, int chunk_start, const float* chunk_v) const {
    if (c_.layer_is_sliding[static_cast<size_t>(layer)] && pos >= chunk_start) {
      return chunk_v + (static_cast<size_t>(pos - chunk_start) * c_.num_kv_heads + kv) * c_.head_dim;
    }
    int capacity = c_.kv_capacity(layer, max_context_);
    int slot = pos % capacity;
    return kv_v_[static_cast<size_t>(layer)].data() +
           (static_cast<size_t>(kv) * capacity + slot) * c_.head_dim;
  }

  void store_kv(int layer, int pos, const float* k, const float* v) {
    int capacity = c_.kv_capacity(layer, max_context_);
    int slot = pos % capacity;
    for (int kv = 0; kv < c_.num_kv_heads; ++kv) {
      size_t dst = (static_cast<size_t>(kv) * capacity + slot) * c_.head_dim;
      std::copy_n(k + static_cast<size_t>(kv) * c_.head_dim, c_.head_dim,
                  kv_k_[static_cast<size_t>(layer)].data() + dst);
      std::copy_n(v + static_cast<size_t>(kv) * c_.head_dim, c_.head_dim,
                  kv_v_[static_cast<size_t>(layer)].data() + dst);
    }
  }

  void apply_rope(float* vec, int pos, const std::vector<double>& inv_freq) const {
    const int half = c_.head_dim / 2;
    for (int i = 0; i < half; ++i) {
      double angle = static_cast<double>(pos) * inv_freq[static_cast<size_t>(i)];
      float cs = static_cast<float>(std::cos(angle));
      float sn = static_cast<float>(std::sin(angle));
      float a = vec[i], b = vec[i + half];
      vec[i] = a * cs - b * sn;
      vec[i + half] = b * cs + a * sn;
    }
  }

  void forward(const int32_t* tokens, int n, bool want_logits) {
    LK_CHECK(position_ + n <= max_context_, "context length exceeded");
    const int H = c_.num_heads, KV = c_.num_kv_heads, D = c_.head_dim;
    const int64_t hidden = c_.hidden_size, inter = c_.intermediate_size;
    const size_t sn = static_cast<size_t>(n);
    x_.resize(sn * hidden);
    h_.resize(sn * hidden);
    r_.resize(sn * hidden);
    q_.resize(sn * H * D);
    k_.resize(sn * KV * D);
    v_.resize(sn * KV * D);
    attn_.resize(sn * H * D);
    gate_.resize(sn * inter);
    up_.resize(sn * inter);

    const float embed_scale = c_.embed_scale();
    for (int t = 0; t < n; ++t) {
      LK_CHECK(tokens[t] >= 0 && tokens[t] < c_.vocab_size, "token id out of range");
      float* row = x_.data() + t * hidden;
      dequantize_row(w_->embed, tokens[t], row);
      for (int64_t i = 0; i < hidden; ++i) row[i] *= embed_scale;
    }

    const int start = position_;
    const float scale = 1.0f / std::sqrt(c_.query_pre_attn_scalar);
    const int group = H / KV;
    for (int l = 0; l < c_.num_layers; ++l) {
      const Gemma3Layer& L = w_->layers[static_cast<size_t>(l)];
      const bool sliding = c_.layer_is_sliding[static_cast<size_t>(l)];
      for (int t = 0; t < n; ++t) {
        rms_norm(x_.data() + t * hidden, L.input_norm.data(), hidden, c_.rms_norm_eps,
                 h_.data() + t * hidden);
      }
      matmul(L.q_proj, h_.data(), n, q_.data(), pool_);
      matmul(L.k_proj, h_.data(), n, k_.data(), pool_);
      matmul(L.v_proj, h_.data(), n, v_.data(), pool_);
      const auto& inv_freq = sliding ? inv_freq_local_ : inv_freq_global_;
      for (int t = 0; t < n; ++t) {
        for (int hh = 0; hh < H; ++hh) {
          float* qv = q_.data() + (static_cast<size_t>(t) * H + hh) * D;
          rms_norm(qv, L.q_norm.data(), D, c_.rms_norm_eps, qv);
          apply_rope(qv, start + t, inv_freq);
        }
        for (int kv = 0; kv < KV; ++kv) {
          float* kvec = k_.data() + (static_cast<size_t>(t) * KV + kv) * D;
          rms_norm(kvec, L.k_norm.data(), D, c_.rms_norm_eps, kvec);
          apply_rope(kvec, start + t, inv_freq);
        }
      }
      // Global layers attend through the cache, so write the chunk first.
      // Sliding layers read the chunk directly and update the ring after.
      if (!sliding) {
        for (int t = 0; t < n; ++t) {
          store_kv(l, start + t, k_.data() + static_cast<size_t>(t) * KV * D,
                   v_.data() + static_cast<size_t>(t) * KV * D);
        }
      }
      pool_.parallel_for(static_cast<size_t>(n) * H, [&](size_t begin, size_t end) {
        std::vector<float> scores;
        for (size_t item = begin; item < end; ++item) {
          const int t = static_cast<int>(item / H);
          const int hh = static_cast<int>(item % H);
          const int kv = hh / group;
          const int pos = start + t;
          const int lo = sliding ? std::max(0, pos - c_.sliding_window + 1) : 0;
          const float* qv = q_.data() + (static_cast<size_t>(t) * H + hh) * D;
          scores.resize(static_cast<size_t>(pos - lo + 1));
          float max_score = -std::numeric_limits<float>::infinity();
          for (int j = lo; j <= pos; ++j) {
            const float* kvec = key_ptr(l, kv, j, start, k_.data());
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += qv[d] * kvec[d];
            dot *= scale;
            if (c_.attn_logit_softcapping > 0.0f) {
              dot = std::tanh(dot / c_.attn_logit_softcapping) * c_.attn_logit_softcapping;
            }
            scores[static_cast<size_t>(j - lo)] = dot;
            max_score = std::max(max_score, dot);
          }
          float denom = 0.0f;
          for (float& s : scores) {
            s = std::exp(s - max_score);
            denom += s;
          }
          float* out = attn_.data() + (static_cast<size_t>(t) * H + hh) * D;
          std::fill(out, out + D, 0.0f);
          for (int j = lo; j <= pos; ++j) {
            const float p = scores[static_cast<size_t>(j - lo)] / denom;
            const float* vvec = value_ptr(l, kv, j, start, v_.data());
            for (int d = 0; d < D; ++d) out[d] += p * vvec[d];
          }
        }
      });
      if (sliding) {
        const int first = std::max(0, n - c_.kv_capacity(l, max_context_));
        for (int t = first; t < n; ++t) {
          store_kv(l, start + t, k_.data() + static_cast<size_t>(t) * KV * D,
                   v_.data() + static_cast<size_t>(t) * KV * D);
        }
      }
      matmul(L.o_proj, attn_.data(), n, r_.data(), pool_);
      for (int t = 0; t < n; ++t) {
        float* xr = x_.data() + t * hidden;
        float* rr = r_.data() + t * hidden;
        rms_norm(rr, L.post_attn_norm.data(), hidden, c_.rms_norm_eps, rr);
        for (int64_t i = 0; i < hidden; ++i) xr[i] += rr[i];
        rms_norm(xr, L.pre_ff_norm.data(), hidden, c_.rms_norm_eps, h_.data() + t * hidden);
      }
      matmul(L.gate_proj, h_.data(), n, gate_.data(), pool_);
      matmul(L.up_proj, h_.data(), n, up_.data(), pool_);
      for (size_t i = 0; i < sn * inter; ++i) gate_[i] = gelu_tanh(gate_[i]) * up_[i];
      matmul(L.down_proj, gate_.data(), n, r_.data(), pool_);
      for (int t = 0; t < n; ++t) {
        float* xr = x_.data() + t * hidden;
        float* rr = r_.data() + t * hidden;
        rms_norm(rr, L.post_ff_norm.data(), hidden, c_.rms_norm_eps, rr);
        for (int64_t i = 0; i < hidden; ++i) xr[i] += rr[i];
      }
    }
    position_ += n;
    if (!want_logits) return;
    // Only the last row's logits are needed: skip the vocabulary projection
    // for every other prompt token.
    std::vector<float> last(static_cast<size_t>(hidden));
    rms_norm(x_.data() + static_cast<size_t>(n - 1) * hidden, w_->final_norm.data(), hidden,
             c_.rms_norm_eps, last.data());
    matvec(w_->lm_head, last.data(), logits_.data(), pool_);
    if (c_.final_logit_softcapping > 0.0f) {
      const float cap = c_.final_logit_softcapping;
      for (float& v : logits_) v = std::tanh(v / cap) * cap;
    }
  }

  std::shared_ptr<Gemma3Weights> w_;
  const Gemma3Config& c_;
  ThreadPool pool_;
  int max_context_;
  int chunk_;
  int position_ = 0;
  std::vector<std::vector<float>> kv_k_, kv_v_;
  std::vector<double> inv_freq_local_, inv_freq_global_;
  std::vector<float> x_, h_, r_, q_, k_, v_, attn_, gate_, up_, logits_;
};

}  // namespace

std::unique_ptr<Model> make_gemma3_cpu(std::shared_ptr<Gemma3Weights> weights,
                                       const LoadOptions& options) {
  return std::make_unique<Gemma3Cpu>(std::move(weights), options);
}

}  // namespace lokahi
