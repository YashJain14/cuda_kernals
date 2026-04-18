#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "common.h"

namespace flash {

FA_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

template <int ngroups>
FA_DEVICE void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
}

FA_DEVICE void cp_async_commit_and_wait_all() {
    cp_async_commit();
    cp_async_wait<0>();
}

template <int size, typename T>
FA_DEVICE void cp_async(T *smem_to, T *gmem_from) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    // The .cg option bypasses the L1 cache.
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_from), "n"(size));
}

template <bool transpose, typename T>
FA_DEVICE void ldmatrix_x4(T *load_from, uint32_t &a1, uint32_t &a2,
                           uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    if constexpr (transpose) {
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                     : "r"(smem_ptr));
    } else {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                     : "r"(smem_ptr));
    }
}

template <typename value_t>
FA_DEVICE void
mma_m16n8k16_f32_accum(float &d1, float &d2, float &d3, float &d4,
                       uint32_t const &a1, uint32_t const &a2,
                       uint32_t const &a3, uint32_t const &a4,
                       uint32_t const &b1, uint32_t const &b2, float const &c1,
                       float const &c2, float const &c3, float const &c4) {
    static_assert(std::is_same_v<value_t, half> ||
                      std::is_same_v<value_t, nv_bfloat16>,
                  "value_t must be either half or nv_bfloat16");

    if constexpr (std::is_same_v<value_t, half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 }; "
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 }; "
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

//
// st functions
//
// Adapted from cccl:
// https://github.com/NVIDIA/cccl/blob/29e6e2fc0a2fcaf3e8b2451fe89e3a40edf8c2b0/libcudacxx/include/cuda/__ptx/instructions/generated/st.h
//

// SM 80 (A100) does not support .reg .b128 / mov.b128.
// Use v2.u64 vectorized stores instead, which achieve the same 128-bit
// coalesced write and are valid from SM 20 onwards.

template <typename _B128>
FA_DEVICE void st_global(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.global.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_relaxed_gpu(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.relaxed.gpu.global.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_relaxed_sys(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_ef(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.global.L1::evict_first.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_ef_relaxed_gpu(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.relaxed.gpu.global.L1::evict_first.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_ef_relaxed_sys(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.relaxed.sys.global.L1::evict_first.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_na(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.global.L1::no_allocate.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_na_relaxed_gpu(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_na_relaxed_sys(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");
    longlong2 src = *reinterpret_cast<longlong2 *>(&__src);
    asm volatile("st.relaxed.sys.global.L1::no_allocate.v2.u64 [%0], {%1, %2};"
                 :
                 : "l"(__addr), "l"(src.x), "l"(src.y)
                 : "memory");
}

} // namespace flash