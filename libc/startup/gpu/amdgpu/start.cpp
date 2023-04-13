//===-- Implementation of crt for amdgpu ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/RPC/rpc_client.h"

extern "C" int main(int argc, char **argv, char **envp);

extern "C" [[gnu::visibility("protected"), clang::amdgpu_kernel]] void
_start(int argc, char **argv, char **envp, int *ret, void *in, void *out,
       void *buffer, void* more_bits) {
  static __llvm_libc::cpp::Atomic<uint32_t> locks = {0};
  __llvm_libc::rpc::client.reset(&locks, in, out, buffer, more_bits);

  __atomic_fetch_or(ret, main(argc, argv, envp), __ATOMIC_RELAXED);
}
