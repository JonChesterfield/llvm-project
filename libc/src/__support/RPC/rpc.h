//===-- Shared memory RPC client / server interface -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a remote procedure call mechanism to communicate between
// heterogeneous devices that can share an address space atomically. We provide
// a client and a server to facilitate the remote call. The client makes request
// to the server using a shared communication channel. We use separate atomic
// signals to indicate which side, the client or the server is in ownership of
// the buffer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_RPC_RPC_H
#define LLVM_LIBC_SRC_SUPPORT_RPC_RPC_H

#include "rpc_util.h"
#include "src/__support/CPP/atomic.h"

#include <stdint.h>

namespace __llvm_libc {
namespace rpc {

/// A list of opcodes that we use to invoke certain actions on the server. We
/// reserve the first 255 values for internal libc usage.
enum Opcode : uint64_t {
  NOOP = 0,
  PRINT_TO_STDERR = 1,
  EXIT = 2,
  LIBC_LAST = (1UL << 8) - 1,
};

/// A fixed size channel used to communicate between the RPC client and server.
struct Buffer {
  uint64_t data[8];
};

template <unsigned I, unsigned O> struct port_t {
  static_assert(I == 0 || I == 1, "");
  static_assert(O == 0 || O == 1, "");
  uint32_t value;

  port_t(uint32_t value) : value(value) {}

  port_t<!I, O> invert_inbox() { return value; }
  port_t<I, !O> invert_outbox() { return value; }
};

template <bool InvertedLoad, int scope> struct bitmap_t {
private:
  cpp::Atomic<uint32_t> *underlying;
  using Word = uint32_t;

  template <typename Word> inline uint32_t index_to_element(uint32_t x) {
    uint32_t wordBits = 8 * sizeof(Word);
    return x / wordBits;
  }

  template <typename Word> inline uint32_t index_to_subindex(uint32_t x) {
    uint32_t wordBits = 8 * sizeof(Word);
    return x % wordBits;
  }

  inline bool nthbitset(uint32_t x, uint32_t n) {
    return x & (UINT32_C(1) << n);
  }

  inline bool nthbitset(uint64_t x, uint32_t n) {
    return x & (UINT64_C(1) << n);
  }

  inline uint32_t setnthbit(uint32_t x, uint32_t n) {
    return x | (UINT32_C(1) << n);
  }

  inline uint64_t setnthbit(uint64_t x, uint32_t n) {
    return x | (UINT64_C(1) << n);
  }

  static constexpr bool system_scope() {
    return scope == __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES;
  }
  static constexpr bool device_scope() {
    return scope == __OPENCL_MEMORY_SCOPE_DEVICE;
  }

  static_assert(system_scope() || device_scope(), "");
  static_assert(system_scope() != device_scope(), "");

  Word load_word(uint32_t w) const {
    cpp::Atomic<uint32_t> &addr = underlying[w];
    Word tmp = addr.load(cpp::MemoryOrder::RELAXED);
    return InvertedLoad ? ~tmp : tmp;
  }

public:
  bitmap_t() /*: underlying(nullptr)*/ {}
  bitmap_t(cpp::Atomic<uint32_t> *d) : underlying(d) {
    // can't necessarily write to the pointer from this object. if the memory is
    // on a gpu, but this instance is being constructed on a cpu first, then
    // direct writes will fail. However, the data does need to be zeroed for the
    // bitmap to work.
  }

  bool read_slot(uint32_t slot) {
    uint32_t w = index_to_element<Word>(slot);
    uint32_t subindex = index_to_subindex<Word>(slot);
    return nthbitset(load_word(w), subindex);
  }

  template <unsigned I> port_t<I, 1> claim_slot(port_t<I, 0> port) {
    uint32_t i = port.value;

    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);

    cpp::Atomic<uint32_t> &addr = underlying[w];
    Word before;
    if (system_scope()) {
      Word addend = (Word)1 << subindex;
      before = addr.fetch_add(addend, cpp::MemoryOrder::ACQ_REL);
    } else {
      Word mask = setnthbit((Word)0, subindex);
      before = addr.fetch_or(mask, cpp::MemoryOrder::ACQ_REL);
    }

    (void)before;
    return port.invert_outbox();
  }

  void release_slot(uint32_t i) {
    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);

    cpp::Atomic<uint32_t> &addr = underlying[w];

    if (system_scope()) {
      Word addend = 1 + ~((Word)1 << subindex);
      addr.fetch_add(addend, cpp::MemoryOrder::ACQ_REL);
    } else {
      // and with everything other than the slot set
      Word mask = ~setnthbit((Word)0, subindex);
      addr.fetch_and(mask, cpp::MemoryOrder::ACQ_REL);
    }
  }

  template <unsigned I> port_t<I, 0> release_slot(port_t<I, 1> port) {
    release_slot(port.value);
    return port.invert_outbox();
  }

  bool try_claim_slot(uint32_t slot) {
    uint32_t i = slot;

    uint32_t w = index_to_element<Word>(i);
    uint32_t subindex = index_to_subindex<Word>(i);

    static_assert(device_scope(), "");

    Word mask = setnthbit((Word)0, subindex);

    cpp::Atomic<uint32_t> &addr = underlying[w];
    uint32_t before = addr.fetch_or(mask, cpp::MemoryOrder::ACQ_REL);

    return !nthbitset(before, subindex);
  }
};

template <uint32_t WarpSize> struct WarpSizeType;
template <> struct WarpSizeType<32> { using Type = uint32_t; };
template <> struct WarpSizeType<64> { using Type = uint64_t; };

/// A common process used to synchronize communication between a client and a
/// server. The process contains an inbox and an outbox used for signaling
/// ownership of the shared buffer.
template <typename BufferElement, uint32_t WarpSize, uint32_t NumberSlots,
          bool InvertedInboxLoadT>
struct Process {
  static_assert(NumberSlots > 0, "");
  static_assert(WarpSize == 32 || WarpSize == 64, "");

  BufferElement *shared_buffer;
  bitmap_t<false, __OPENCL_MEMORY_SCOPE_DEVICE> active;
  bitmap_t<InvertedInboxLoadT,__OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES> inbox;
  bitmap_t<false, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES> outbox;

  using ThreadMask = typename WarpSizeType<WarpSize>::Type;

  Process() = default;
  ~Process() = default;

  Process(cpp::Atomic<uint32_t> *locks, cpp::Atomic<uint32_t> *inbox,
          cpp::Atomic<uint32_t> *outbox, BufferElement *shared_buffer)
      : shared_buffer(shared_buffer), active(locks), inbox(inbox),
        outbox(outbox) {}

  /// Initialize the communication channels.
  LIBC_INLINE void reset(void *locks, void *inbox, void *outbox, void *buffer) {
    this->active = reinterpret_cast<cpp::Atomic<uint32_t> *>(locks);
    this->inbox = reinterpret_cast<cpp::Atomic<uint32_t> *>(inbox);
    this->outbox = reinterpret_cast<cpp::Atomic<uint32_t> *>(outbox);
    this->shared_buffer = reinterpret_cast<BufferElement *>(buffer);
  }

  struct pair {
    port_t<0, 0> port;
    bool success;
  };

  port_t<0, 0> open(ThreadMask active_threads) {
    for (;;) {
      pair r = try_open(active_threads);
      if (r.success) {
        return r.port;
      }
      sleep_briefly();
    }
  }

  pair try_open(ThreadMask active_threads) {

    for (uint32_t p = 0; p < NumberSlots; p++) {
      bool claim = false;
      if (is_first_lane(active_threads)) {
        claim = active.try_claim_slot(p);
      }
      claim = broadcast_first_lane(active_threads, claim);

      if (!claim) {
        continue;
      }

      atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
      bool in = inbox.read_slot(p);
      bool out = outbox.read_slot(p);

      if (in == 0 && out == 0) {
        return {p, true};
      }

      if (in == 1 && out == 1) {
        // garbage collect from an async call
        outbox.release_slot(p);
      }

      active.release_slot(p);
    }

    return {UINT32_MAX, false};
  }

  template <unsigned I, unsigned O> void close(port_t<I, O> port) {
    active.release_slot(port.value);
  }

  template <unsigned IandO, typename Op>
  port_t<IandO, IandO> use(port_t<IandO, IandO> port, Op op) {
    // Op could be && and forwarded, but Client::run takes it by value
    uint32_t raw = port.value;
    // todo: this depends on how buffer is extended to multiple warps
    uint64_t active_threads = 1;
    if (is_first_lane(active_threads)) {
      op(&shared_buffer[raw]);
    }
    return port;
  }

  template <unsigned I> port_t<!I, !I> wait(port_t<I, !I> port) {
    bool in = inbox.read_slot(port.value);
    while (in == I) {
      sleep_briefly();
      in = inbox.read_slot(port.value);
    }

    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
    return port.invert_inbox();
  }

  template <unsigned IandO>
  port_t<IandO, !IandO> send(port_t<IandO, IandO> port) {
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);

    if constexpr (IandO == 0) {
      return outbox.claim_slot(port);
    } else {
      return outbox.release_slot(port);
    }
  }
};

/// The RPC client used to make requests to the server.
struct Client : public Process<Buffer, 32, 1, false> {
  LIBC_INLINE Client() = default;
  LIBC_INLINE Client(const Client &) = default;
  LIBC_INLINE Client &operator=(const Client &) = default;
  LIBC_INLINE ~Client() = default;

  template <typename F, typename U> LIBC_INLINE void run(F fill, U use);
};

/// The RPC server used to respond to the client.
struct Server : public Process<Buffer, 32, 1, true> {
  LIBC_INLINE Server() = default;
  LIBC_INLINE Server(const Server &) = default;
  LIBC_INLINE Server &operator=(const Server &) = default;
  LIBC_INLINE ~Server() = default;

  template <typename W, typename C> LIBC_INLINE bool handle(W work, C clean);
};

/// Run the RPC client protocol to communicate with the server. We perform the
/// following high level actions to complete a communication:
///   - Apply \p fill to the shared buffer and write 1 to the outbox.
///   - Wait until the inbox is 1.
///   - Apply \p use to the shared buffer and write 0 to the outbox.
///   - Wait until the inbox is 0.
template <typename F, typename U> LIBC_INLINE void Client::run(F fill, U use) {
  uint32_t ThreadMask = 1;

  port_t<0, 0> port0 = open(ThreadMask);

  // Apply the \p fill to the buffer and signal the server.
  port_t<0, 1> port1 = send(this->use(port0, fill));

  // Wait for the server to work on the buffer and respond.
  port_t<1, 1> port2 = wait(port1);

  // Apply \p use to the buffer and signal the server.
  port_t<1, 0> port3 = send(this->use(port2, use));

  // Wait for the server to signal the end of the protocol.
  close(port3);
}

/// Run the RPC server protocol to communicate with the client. This is
/// non-blocking and only checks the server a single time. We perform the
/// following high level actions to complete a communication:
///   - Open a port or exit if there is no work to do.
///   - Apply \p work to the shared buffer and write 1 to the outbox.
///   - Wait until the inbox is 1.
///   - Apply \p clean to the shared buffer and write 0 to the outbox.
template <typename W, typename C>
LIBC_INLINE bool Server::handle(W work, C clean) {
  uint32_t ThreadMask = 1;

  auto maybe_port = try_open(ThreadMask);
  // There is no work to do, exit early.
  if (!maybe_port.success)
    return false;

  port_t<0, 0> port0 = maybe_port.port;

  // Apply \p work to the buffer and signal the client.
  port_t<0, 1> port1 = send(use(port0, work));

  // Wait for the client to use the buffer and respond.
  port_t<1, 1> port2 = wait(port1);

  // Clean up the buffer and signal the end of the protocol.
  port_t<1, 0> port3 = send(use(port2, clean));

  return true;
}

} // namespace rpc
} // namespace __llvm_libc

#endif
