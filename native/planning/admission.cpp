#include "admission.h"
#include <algorithm>
#include <limits>

namespace {
bool add(uint64_t &sum, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max()-sum) return false;
    sum += value;
    return true;
}
}

extern "C" strata_admit_status strata_admit_v1(const strata_host_budget_v1 *h,
                                               const strata_model_budget_v1 *m,
                                               strata_admission_v1 *out) {
    if (!out) return STRATA_ADMIT_INVALID;
    *out = {};
    if (!h || !m || m->reserved || !m->requested_sessions ||
        m->requested_sessions>64 || m->ssd_offload>1) return STRATA_ADMIT_INVALID;
    if (m->ssd_offload) return STRATA_ADMIT_SSD_UNIMPLEMENTED;
    if (!h->physical_bytes || !h->metal_recommended_working_set_bytes ||
        !h->available_plan_bytes) return STRATA_ADMIT_UNKNOWN_CAPACITY;
    if (h->available_plan_bytes>h->physical_bytes ||
        h->metal_recommended_working_set_bytes>h->physical_bytes) return STRATA_ADMIT_INVALID;
    if (m->accounting_complete!=1 || !m->resident_weight_bytes ||
        !m->state_bytes_per_session || !m->scratch_bytes_per_session)
        return STRATA_ADMIT_INCOMPLETE_ACCOUNTING;
    constexpr uint64_t gib=uint64_t{1}<<30;
    // Conservative initial policy, not an Apple framework guarantee. Never
    // override the OS GPU working-set recommendation to manufacture capacity.
    out->os_reserve_bytes=std::max(4*gib,h->physical_bytes/8);
    if (out->os_reserve_bytes>=h->physical_bytes) return STRATA_ADMIT_NO_CAPACITY;
    out->plan_limit_bytes=std::min({h->physical_bytes-out->os_reserve_bytes,
                                  h->metal_recommended_working_set_bytes,
                                  h->available_plan_bytes});
    uint64_t fixed=m->resident_weight_bytes;
    if (!add(fixed,m->ane_extra_bytes) || !add(fixed,m->fixed_runtime_bytes) ||
        !add(fixed,m->cache_budget_bytes) || !add(fixed,m->staging_bytes)) return STRATA_ADMIT_OVERFLOW;
    uint64_t per_session=m->state_bytes_per_session;
    if (!add(per_session,m->scratch_bytes_per_session)) return STRATA_ADMIT_OVERFLOW;
    out->minimum_peak_bytes=fixed;
    if (!add(out->minimum_peak_bytes,per_session)) return STRATA_ADMIT_OVERFLOW;
    if (out->minimum_peak_bytes>out->plan_limit_bytes) return STRATA_ADMIT_NO_CAPACITY;
    uint64_t sessions=std::min(uint64_t{m->requested_sessions},
                             (out->plan_limit_bytes-fixed)/per_session);
    // Division bounds the product and the sum below by plan_limit_bytes.
    out->admitted_sessions=static_cast<uint32_t>(sessions);
    out->selected_peak_bytes=fixed+sessions*per_session;
    return STRATA_ADMIT_OK;
}
