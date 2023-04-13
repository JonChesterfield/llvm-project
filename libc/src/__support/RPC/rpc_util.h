//===-- Shared memory RPC client / server utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_RPC_RPC_UTILS_H
#define LLVM_LIBC_SRC_SUPPORT_RPC_RPC_UTILS_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"

#include <stdint.h>

namespace __llvm_libc {
namespace rpc {

/// Suspend the thread briefly to assist the thread scheduler during busy loops.
LIBC_INLINE void sleep_briefly() {
#if defined(LIBC_TARGET_ARCH_IS_NVPTX) && __CUDA_ARCH__ >= 700
  asm("nanosleep.u32 64;" ::: "memory");
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  __builtin_amdgcn_s_sleep(2);
#else
  // Simply do nothing if sleeping isn't supported on this platform.
#endif
}

LIBC_INLINE uint64_t get_lane_id() {
#if defined(LIBC_TARGET_ARCH_IS_NVPTX)
  return __nvvm_read_ptx_sreg_tid_x() /*threadIdx.x*/ & (32 - 1);

#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
#if __AMDGCN_WAVEFRONT_SIZE == 64
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
#elif __AMDGCN_WAVEFRONT_SIZE == 32
  return __builtin_amdgcn_mbcnt_lo(~0u, 0u);
#else
#error ""
#endif
#else
  return 0;
#endif
}

template <typename T> LIBC_INLINE uint64_t get_first_lane_id(T active_threads) {
  return __builtin_ffsl(active_threads) - 1;
}

template <typename T> LIBC_INLINE bool is_first_lane(T active_threads) {

  return get_lane_id() == get_first_lane_id(active_threads);
}

template <typename T>
LIBC_INLINE uint32_t broadcast_first_lane(T active_threads, uint32_t x) {
  (void)active_threads;

#if defined(LIBC_TARGET_ARCH_IS_NVPTX) && __CUDA_ARCH__ >= 700

#if 0
#error                                                                         \
    "This doesn't compile, needs target feature ptx60...., despite the cuda arch guard"
  uint32_t first_id = get_first_lane_id(active_threads);
  return __nvvm_shfl_sync_idx_i32(active_threads, x, first_id,
                                  32 - 1);
#else
  return x;
#endif

#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  // reads from lowest set bit in exec mask
  // this is OK from definition of is_first_lane
  return __builtin_amdgcn_readfirstlane(x);
#else
  return x;
#endif
}

} // namespace rpc
} // namespace __llvm_libc

#endif
