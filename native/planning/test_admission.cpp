#include "admission.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <initializer_list>

int main() {
    constexpr uint64_t G=uint64_t{1}<<30;
    strata_host_budget_v1 h{512*G,464*G,480*G};
    strata_model_budget_v1 m{360*G,16*G,4*G,8*G,4*G,8*G,4*G,8,1,0,0};
    strata_admission_v1 p{};
    size_t cases=0;
    auto check=[&](strata_admit_status want){assert(strata_admit_v1(&h,&m,&p)==want);++cases;};
    check(STRATA_ADMIT_OK);
    assert(p.plan_limit_bytes==448*G && p.admitted_sessions==4 && p.selected_peak_bytes==440*G);
    assert(!p.native_runtime_qualified);
    m.ane_extra_bytes=80*G;
    check(STRATA_ADMIT_NO_CAPACITY); // expansion defeats otherwise fitting weights
    m.ane_extra_bytes=16*G; m.ssd_offload=1;
    check(STRATA_ADMIT_SSD_UNIMPLEMENTED);
    m.ssd_offload=2; check(STRATA_ADMIT_INVALID); m.ssd_offload=0;
    m.accounting_complete=0; check(STRATA_ADMIT_INCOMPLETE_ACCOUNTING); m.accounting_complete=1;
    h.metal_recommended_working_set_bytes=0; check(STRATA_ADMIT_UNKNOWN_CAPACITY);
    h.metal_recommended_working_set_bytes=513*G; check(STRATA_ADMIT_INVALID);
    h.metal_recommended_working_set_bytes=464*G;
    m.resident_weight_bytes=std::numeric_limits<uint64_t>::max(); check(STRATA_ADMIT_OVERFLOW);
    m.resident_weight_bytes=G; m.state_bytes_per_session=std::numeric_limits<uint64_t>::max();
    check(STRATA_ADMIT_OVERFLOW); m.state_bytes_per_session=G;
    m.requested_sessions=65; check(STRATA_ADMIT_INVALID); m.requested_sessions=0; check(STRATA_ADMIT_INVALID);
    m={2*G,0,G,G,G,G,G,4,1,0,0};
    for (uint64_t tier : {16,18,24,32,36,48,64,96,128,192,256,384,512}) {
        h={tier*G,tier*G*3/4,tier*G*3/4};
        check(STRATA_ADMIT_OK);
        assert(p.admitted_sessions>0 && p.admitted_sessions<=4 && p.selected_peak_bytes<=p.plan_limit_bytes);
        // Exhaustively test reservation boundaries without large allocations.
        for(uint64_t available=1;available<=tier;++available) {
            h.available_plan_bytes=available*G;
            auto result=strata_admit_v1(&h,&m,&p); ++cases;
            assert(result==STRATA_ADMIT_OK || result==STRATA_ADMIT_NO_CAPACITY);
            if(result==STRATA_ADMIT_OK) assert(p.selected_peak_bytes<=h.available_plan_bytes);
        }
    }
    h={16*G,12*G,8*G}; check(STRATA_ADMIT_OK); assert(p.admitted_sessions==1);
    h.available_plan_bytes=6*G; check(STRATA_ADMIT_NO_CAPACITY);
    h.available_plan_bytes=0; check(STRATA_ADMIT_UNKNOWN_CAPACITY);
    m.state_bytes_per_session=0; h.available_plan_bytes=8*G;
    check(STRATA_ADMIT_INCOMPLETE_ACCOUNTING);
    assert(strata_admit_v1(nullptr,&m,&p)==STRATA_ADMIT_INVALID && p.admitted_sessions==0);
    assert(strata_admit_v1(&h,nullptr,&p)==STRATA_ADMIT_INVALID);
    assert(strata_admit_v1(&h,&m,nullptr)==STRATA_ADMIT_INVALID);
    std::printf("%zu native admission cases passed; synthetic budgets, no model execution\n",cases+3);
}
