/*
 * Lokahi C ABI.
 *
 * Copyright (c) 2025-2026 Common Compute LLC.
 * Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
 *
 * All functions return 0 on success and a nonzero code on failure; call
 * lokahi_last_error() on the same thread for a description. A model handle
 * is not thread-safe; serialize calls per handle.
 */
#ifndef LOKAHI_H
#define LOKAHI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOKAHI_ABI_VERSION 1

typedef struct lokahi_model lokahi_model;

typedef struct {
  const char* backend; /* "auto", "cpu", or "metal"; NULL means "auto" */
  int32_t max_context; /* tokens; <= 0 means 4096 */
  int32_t prefill_chunk; /* tokens per prefill submission; <= 0 means 512 */
  int32_t cpu_threads; /* <= 0 means hardware concurrency */
} lokahi_load_options;

typedef struct {
  double prefill_seconds; /* admitted prompt to first token available */
  double decode_seconds;  /* first token to last token available */
  int32_t generated_tokens;
} lokahi_timing;

int32_t lokahi_abi_version(void);
const char* lokahi_last_error(void);

int lokahi_model_load(const char* directory, const lokahi_load_options* options,
                      lokahi_model** out_model);
void lokahi_model_free(lokahi_model* model);

const char* lokahi_model_backend(const lokahi_model* model);
int32_t lokahi_model_vocab_size(const lokahi_model* model);
int32_t lokahi_model_position(const lokahi_model* model);
void lokahi_model_reset(lokahi_model* model);

/* Append tokens; copies the last token's logits into logits_out when it is
 * not NULL (vocab_size floats). */
int lokahi_prefill(lokahi_model* model, const int32_t* tokens, int32_t count, float* logits_out);
int lokahi_decode(lokahi_model* model, int32_t token, float* logits_out);

/* Greedy generation from the current position. Writes up to max_new tokens
 * and stores the count in *out_count. */
int lokahi_generate_greedy(lokahi_model* model, const int32_t* prompt, int32_t prompt_count,
                           int32_t max_new, int32_t stop_at_eos, int32_t* out_tokens,
                           int32_t* out_count, lokahi_timing* timing);

#ifdef __cplusplus
}
#endif

#endif /* LOKAHI_H */
