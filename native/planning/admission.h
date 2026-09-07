#ifndef STRATA_NATIVE_ADMISSION_H
#define STRATA_NATIVE_ADMISSION_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Capacity planning only: a successful result is NOT model/backend qualification.
// Inputs come from a trusted host probe + complete model adapter accounting,
// never directly from remote customer JSON or self-reported fleet telemetry.
typedef enum {
    STRATA_ADMIT_OK = 0,
    STRATA_ADMIT_INVALID = 1,
    STRATA_ADMIT_UNKNOWN_CAPACITY = 2,
    STRATA_ADMIT_INCOMPLETE_ACCOUNTING = 3,
    STRATA_ADMIT_OVERFLOW = 4,
    STRATA_ADMIT_NO_CAPACITY = 5,
    STRATA_ADMIT_SSD_UNIMPLEMENTED = 6
} strata_admit_status;

typedef struct {
    uint64_t physical_bytes;
    uint64_t metal_recommended_working_set_bytes;
    // Conservative headroom for THIS plan after other workloads/reservations.
    // Not merely free pages; a serving allocator must atomically reserve it.
    uint64_t available_plan_bytes;
} strata_host_budget_v1;

typedef struct {
    uint64_t resident_weight_bytes; // ALL selected shards, not active MoE weights
    uint64_t ane_extra_bytes;       // FP16 expansions + resident compiled programs
    uint64_t fixed_runtime_bytes;   // tensors, tokenizer, runtime overhead
    uint64_t cache_budget_bytes;    // bounded prefix/program caches, counted once
    uint64_t staging_bytes;         // CPU/GPU/ANE copies and bounded I/O staging
    uint64_t state_bytes_per_session; // KV/MLA/recurrent state at admitted context
    uint64_t scratch_bytes_per_session; // worst-phase prefill/decode workspace
    uint32_t requested_sessions;    // bounded search: 1..64
    uint32_t accounting_complete;  // exactly 1; adapter validated all terms
    uint32_t ssd_offload;           // 0=disabled; 1=requested experimental paging
    uint32_t reserved;              // must be zero for ABI v1
} strata_model_budget_v1;

typedef struct {
    uint64_t plan_limit_bytes;
    uint64_t os_reserve_bytes;
    uint64_t minimum_peak_bytes;
    uint64_t selected_peak_bytes;
    uint32_t admitted_sessions;
    uint32_t native_runtime_qualified; // always zero; independent promotion gate
} strata_admission_v1;

// No allocation, framework calls, downloads, state mutation or paging.
// All arithmetic checked. Out is zeroed even when validation fails.
strata_admit_status strata_admit_v1(const strata_host_budget_v1 *host,
                                   const strata_model_budget_v1 *model,
                                   strata_admission_v1 *out);
#ifdef __cplusplus
}
#endif
#endif
