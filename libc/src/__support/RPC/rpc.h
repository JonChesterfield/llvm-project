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

enum : uint32_t {
  NumberPorts = 64,
  NumberUInt32ForBitmaps = 2,
};

/// A fixed size channel used to communicate between the RPC client and server.
struct ThreadBuffer {
  uint64_t data[8];
};

struct Buffer {
  ThreadBuffer data[32]; // TODO: handle wavesize==64 as well
};

// The data structure Process is essentially four arrays of the same length
// indexed by port_t. The operations on process provide mutally exclusive
// access to the Buffer element at index port_t::value. Ownership alternates
// between the client and the server instance.
// The template parameters I, O correspond to the runtime
// values of the state machine implemented here from the perspective of the
// current process. They are tracked in the type system to raise errors on
// attempts to make invalid transitions or to use the protected buffer
// while the other process owns it.

template <unsigned I, unsigned O> struct port_t {
  static_assert(I == 0 || I == 1, "");
  static_assert(O == 0 || O == 1, "");
  uint32_t value;

  port_t(uint32_t value) : value(value) {}

  port_t<!I, O> invert_inbox() { return value; }
  port_t<I, !O> invert_outbox() { return value; }
};


/// Bitmap deals with consistently picking the address that corresponds to a
/// given port instance. 'Slot' is used to mean an index into the shared arrays
/// which may not be currently bound to a port.
template <bool InvertedLoad, int scope> struct bitmap_t {
private:
  cpp::Atomic<uint32_t> *underlying;
  using Word = uint32_t;

  inline uint32_t index_to_element(uint32_t x) {
    uint32_t wordBits = 8 * sizeof(Word);
    return x / wordBits;
  }

  inline uint32_t index_to_subindex(uint32_t x) {
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
    uint32_t w = index_to_element(slot);
    uint32_t subindex = index_to_subindex(slot);
    return nthbitset(load_word(w), subindex);
  }

  // Does not change inbox as no process ever writes to it's own inbox
  // Knows that the outbox is initially zero which allows using fetch_add
  // to set the bit over pci-e, otherwise we would need to use a CAS loop
  template <unsigned I> port_t<I, 1> claim_slot(port_t<I, 0> port) {
    uint32_t slot = port.value;

    uint32_t w = index_to_element(slot);
    uint32_t subindex = index_to_subindex(slot);

    cpp::Atomic<uint32_t> &addr = underlying[w];
    Word before;
    if (system_scope()) {
      // System scope is used here to approximate 'might be over pcie', where
      // the available atomic operations are likely to be CAS, Add, Exchange.
      // Set the bit using the knowledge that it is currently clear.
      Word addend = (Word)1 << subindex;
      before = addr.fetch_add(addend, cpp::MemoryOrder::ACQ_REL);
    } else {
      // Set the bit more clearly. TODO: device scope is missing from atomic.h
      Word mask = setnthbit((Word)0, subindex);
      before = addr.fetch_or(mask, cpp::MemoryOrder::ACQ_REL);
    }

    (void)before;
    return port.invert_outbox();
  }

  // Release also does not change the inbox. Assumes the outbox is set
  template <unsigned I> port_t<I, 0> release_slot(port_t<I, 1> port) {
    release_slot(port.value);
    return port.invert_outbox();
  }

  
  // Not wholly typed as called to drop partially constructed ports, locks
  void release_slot(uint32_t i) {
    uint32_t w = index_to_element(i);
    uint32_t subindex = index_to_subindex(i);

    cpp::Atomic<uint32_t> &addr = underlying[w];

    if (system_scope()) {
      // Clear the bit using the knowledge that it is currently set.
      Word addend = 1 + ~((Word)1 << subindex);
      addr.fetch_add(addend, cpp::MemoryOrder::ACQ_REL);
    } else {
      // Clear the bit more clearly
      Word mask = ~setnthbit((Word)0, subindex);
      addr.fetch_and(mask, cpp::MemoryOrder::ACQ_REL);
    }
  }

  // Only used on the bitmap used for device local mutual exclusion. Does not
  // hit shared memory.
  bool try_claim_slot(uint32_t slot) {
    uint32_t w = index_to_element(slot);
    uint32_t subindex = index_to_subindex(slot);

    static_assert(device_scope(), "");

    Word mask = setnthbit((Word)0, subindex);

    // Fetch or implementing test and set as a faster alternative to CAS
    cpp::Atomic<uint32_t> &addr = underlying[w];
    uint32_t before = addr.fetch_or(mask, cpp::MemoryOrder::ACQ_REL);

    return !nthbitset(before, subindex);
  }
};

// TODO: Work out a reasonable way to abstract over this
template <uint32_t WaveSize> struct WaveSizeType;
template <> struct WaveSizeType<32> { using Type = uint32_t; };
template <> struct WaveSizeType<64> { using Type = uint64_t; };

/// A common process used to synchronize communication between a client and a
/// server. The process contains an inbox and an outbox used for signaling
/// ownership of the shared buffer.
template <typename BufferElement, uint32_t WaveSize, bool InvertedInboxLoadT>
struct Process {
  static_assert(WaveSize == 32 || WaveSize == 64, "");

  static_assert(WaveSize == 32, "64 not yet implemented");

  // The inverted read on inbox determines which of two linked processes
  // initially owns the underlying buffer.
  // The initial memory state is inbox == outbox == 0 which implies ownership,
  // but one process has inbox read bitwise inverted. That starts out believing
  // believing the memory state is inbox == 1, outbox == 0, which implies the
  // other process owns the buffer.
  BufferElement *shared_buffer;
  bitmap_t<false, __OPENCL_MEMORY_SCOPE_DEVICE> active;
  bitmap_t<InvertedInboxLoadT, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES> inbox;
  bitmap_t<false, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES> outbox;

  using ThreadMask = typename WaveSizeType<WaveSize>::Type;

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

  template <typename T> struct maybe {
    T value;
    bool success;
  };

  /// Try to claim one of the buffer elements for this warp/wavefront/wave
  maybe<port_t<0, 0>> try_open(ThreadMask active_threads) {

    // Inefficient try-each-port-in-order for simplicity of initial diff
    for (uint32_t p = 0; p < NumberPorts; p++) {

      // GPUs test-set is per-lane-in-wave so you have to mask off all but one
      // in order to distinguish between a different wave getting the lock vs
      // a different lane in this wave getting the lock.
      // Passing the active threads around is a volta specific quirk, can
      // usually make that a compile time value for minor codegen improvements
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
        // Only return a port in the 0, 0 state
        return {p, true};
      }

      if (in == 1 && out == 1) {
        // Garbage collect from an async call
        outbox.release_slot(p);
      }

      // Other values mean the buffer is not available to this process
      active.release_slot(p);
    }

    return {UINT32_MAX, false};
  }

  /// Release a port. Any inbox/outbox state is acceptable.
  template <unsigned I, unsigned O> void close(port_t<I, O> port) {
    active.release_slot(port.value);
  }

  /// Call a function Op on the owned buffer. Note I==O is required.
  template <unsigned IandO, typename Op>
  void apply(port_t<IandO, IandO> port, Op op) {
    uint32_t raw = port.value;
    op(&shared_buffer[raw].data[get_lane_id()]);
  }


  /// Release ownership of the buffer to the other process.
  /// Requires I==O to call, returns I!=O.
  template <unsigned IandO>
  port_t<IandO, !IandO> post(port_t<IandO, IandO> port) {
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    if constexpr (IandO == 0) {
      return outbox.claim_slot(port);
    } else {
      return outbox.release_slot(port);
    }
  }



  
  /// Wait for the buffer to be returned by the other process.
  /// Equivalently, for the other process to close the port.
  /// Requires I!=O to call, returns I==O
  template <unsigned I> port_t<!I, !I> wait(port_t<I, !I> port) {
    bool in = inbox.read_slot(port.value);
    while (in == I) {
      sleep_briefly();
      in = inbox.read_slot(port.value);
    }

    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
    return port.invert_inbox();
  }


  /// Derivative / convenience functions, possibly better in a derived class
  port_t<0, 0> open(ThreadMask active_threads) {
    for (;;) {
      maybe<port_t<0, 0>> r = try_open(active_threads);
      if (r.success) {
        return r.value;
      }
      sleep_briefly();
    }
  }

};

/// The RPC client used to make requests to the server.
/// The 'false' parameter to Process means this instance can open ports first
struct Client : public Process<Buffer, 32, false> {
  LIBC_INLINE Client() = default;
  LIBC_INLINE Client(const Client &) = default;
  LIBC_INLINE Client &operator=(const Client &) = default;
  LIBC_INLINE ~Client() = default;

  template <typename F, typename U> LIBC_INLINE void run(F fill, U use);
  template <typename F> LIBC_INLINE void run_async(F fill);
};

/// The RPC server used to respond to the client.
/// The 'true' parameter to Process means all ports will be unavailable
/// initially, until Client has opened one and then called post on it.
struct Server : public Process<Buffer, 32, true> {
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
  uint32_t ThreadMask = 1; // TODO: This needs to be passed in.

  port_t<0, 0> port0 = open(ThreadMask);

  // Apply the \p fill to the buffer and signal the server.
  apply(port0, fill);
  port_t<0, 1> port1 = post(port0);

  // Wait for the server to work on the buffer and respond.
  port_t<1, 1> port2 = wait(port1);

  // Apply \p use to the buffer and signal the server.
  apply(port2, use);
  port_t<1, 0> port3 = post(port2);

  // Wait for the server to signal the end of the protocol.
  close(port3);
}

template <typename F> LIBC_INLINE void Client::run_async(F fill) {
  uint32_t ThreadMask = 1;
  port_t<0, 0> port0 = open(ThreadMask);
  // Apply the \p fill to the buffer and signal the server.
  apply(port0, fill);
  port_t<0, 1> port1 = post(port0);
  close(port1);
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

  port_t<0, 0> port0 = maybe_port.value;

  // Apply \p work to the buffer and signal the client.
  apply(port0, work);
  port_t<0, 1> port1 = post(port0);

  // Wait for the client to use the buffer and respond.
  port_t<1, 1> port2 = wait(port1);

  // Clean up the buffer and signal the end of the protocol.
  apply(port2, clean);
  port_t<1, 0> port3 = post(port2);

  close(port3);
  return true;
}

} // namespace rpc
} // namespace __llvm_libc

#endif
