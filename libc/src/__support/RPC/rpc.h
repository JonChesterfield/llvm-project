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
#include "src/__support/CPP/optional.h"
#include "src/string/memory_utils/memcpy_implementations.h"

#include <stdint.h>

namespace __llvm_libc {
namespace rpc {

/// A list of opcodes that we use to invoke certain actions on the server.
enum Opcode : uint16_t {
  NOOP = 0,
  PRINT_TO_STDERR = 1,
  EXIT = 2,
  TEST_INCREMENT = 3,
};

/// A fixed size channel used to communicate between the RPC client and server.
struct alignas(64) Buffer {
  uint8_t data[62];
  uint16_t opcode;
};
static_assert(sizeof(Buffer) == 64, "Buffer size mismatch");

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

template <bool InvertedLoad, int scope = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>
struct bitmap_t {
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

/// A common process used to synchronize communication between a client and a
/// server. The process contains an inbox and an outbox used for signaling
/// ownership of the shared buffer.
template <typename BufferElement, bool InvertedInboxLoadT> struct Process {

  // The inverted read on inbox determines which of two linked processes
  // initially owns the underlying buffer.
  // The initial memory state is inbox == outbox == 0 which implies ownership,
  // but one process has inbox read bitwise inverted. That starts out believing
  // believing the memory state is inbox == 1, outbox == 0, which implies the
  // other process owns the buffer.
  BufferElement *shared_buffer;
  bitmap_t<false, __OPENCL_MEMORY_SCOPE_DEVICE> active;
  bitmap_t<InvertedInboxLoadT> inbox;
  bitmap_t<false> outbox;

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
  maybe<port_t<0, 0>> try_open() {

    // Only one port available at present
    uint32_t p = 0;
    {
      bool claim = active.try_claim_slot(p);
      if (!claim) {
        return {UINT32_MAX, false};
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
        // Missing from previous implementation, would leak on odd numbers of send/recv
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
  void apply(port_t<IandO, IandO>, Op op) {
    op(shared_buffer);
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
};

/// The port provides the interface to communicate between the multiple
/// processes. A port is conceptually an index into the memory provided by the
/// underlying process that is guarded by a lock bit.
template <typename Buffer, bool InvertedInboxLoad> struct PortT {
  // TODO: This should be move-only.
  LIBC_INLINE PortT(Process<Buffer, InvertedInboxLoad> &process, uint64_t index,
                    uint32_t out)
      : process(process), index(index), out(out) {}
  LIBC_INLINE PortT(const PortT &) = default;
  LIBC_INLINE PortT &operator=(const PortT &) = delete;
  LIBC_INLINE ~PortT() = default;

  template <typename U> LIBC_INLINE void recv(U use);
  template <typename F> LIBC_INLINE void send(F fill);
  template <typename F, typename U>
  LIBC_INLINE void send_and_recv(F fill, U use);
  template <typename W> LIBC_INLINE void recv_and_send(W work);
  LIBC_INLINE void send_n(const void *src, uint64_t size);
  template <typename A> LIBC_INLINE void recv_n(A alloc);

  LIBC_INLINE uint16_t get_opcode() const {
    return process.shared_buffer[index].opcode;
  }

  LIBC_INLINE void close() {
    port_t<1, 0> tmp(index);
    process.close(tmp);
  }

private:
  Process<Buffer, InvertedInboxLoad> &process;
  uint32_t index;
  uint32_t out;
};

/// The RPC client used to make requests to the server.
/// The 'false' parameter to Process means this instance can open ports first
struct Client : public Process<Buffer, false> {
  LIBC_INLINE Client() = default;
  LIBC_INLINE Client(const Client &) = default;
  LIBC_INLINE Client &operator=(const Client &) = default;
  LIBC_INLINE ~Client() = default;

  using Port = PortT<Buffer, false>;
  LIBC_INLINE cpp::optional<Port> try_open(uint16_t opcode);
  LIBC_INLINE Port open(uint16_t opcode);
};

/// The RPC server used to respond to the client.
/// The 'true' parameter to Process means all ports will be unavailable
/// initially, until Client has opened one and then called post on it.
struct Server : public Process<Buffer, true> {
  LIBC_INLINE Server() = default;
  LIBC_INLINE Server(const Server &) = default;
  LIBC_INLINE Server &operator=(const Server &) = default;
  LIBC_INLINE ~Server() = default;

  using Port = PortT<Buffer, true>;
  LIBC_INLINE cpp::optional<Port> try_open();
  LIBC_INLINE Port open();
};

/// Applies \p fill to the shared buffer and initiates a send operation.
template <typename Buffer, bool InvertedInboxLoad>
template <typename F>
LIBC_INLINE void PortT<Buffer, InvertedInboxLoad>::send(F fill) {
  // index in Port corresponds to .value in port_t
  // Maintaining the invariant that a port is owned by the other side
  // before and after Port::send or Port::recv
  if (out == 0) {
    port_t<1, 0> port0(index);
    port_t<0, 0> port1 = process.wait(port0);
    process.apply(port1, fill);
    port_t<0, 1> port2 = process.post(port1);
    out = 1;
  } else {
    port_t<0, 1> port0(index);
    port_t<1, 1> port1 = process.wait(port0);
    process.apply(port1, fill);
    port_t<1, 0> port2 = process.post(port1);
    out = 0;
  }
}

/// Applies \p use to the shared buffer and acknowledges the send.
template <typename Buffer, bool InvertedInboxLoad>
template <typename U>
LIBC_INLINE void PortT<Buffer, InvertedInboxLoad>::recv(U use) {
  // it's the same, dispatch implicit in the boolean template parameter
  PortT<Buffer, InvertedInboxLoad>::send(use);
}

/// Combines a send and receive into a single function.
template <typename Buffer, bool InvertedInboxLoad>
template <typename F, typename U>
LIBC_INLINE void PortT<Buffer, InvertedInboxLoad>::send_and_recv(F fill,
                                                                 U use) {
  send(fill);
  recv(use);
}

/// Combines a receive and send operation into a single function. The \p work
/// function modifies the buffer in-place and the send is only used to initiate
/// the copy back.
template <typename Buffer, bool InvertedInboxLoad>
template <typename W>
LIBC_INLINE void PortT<Buffer, InvertedInboxLoad>::recv_and_send(W work) {
  recv(work);
  send([](Buffer *) { /* no-op */ });
}

/// Sends an arbitrarily sized data buffer \p src across the shared channel in
/// multiples of the packet length.
template <typename Buffer, bool InvertedInboxLoad>
LIBC_INLINE void PortT<Buffer, InvertedInboxLoad>::send_n(const void *src,
                                                          uint64_t size) {
  // TODO: We could send the first bytes in this call and potentially save an
  // extra send operation.
  send([=](Buffer *buffer) { buffer->data[0] = size; });
  const uint8_t *ptr = reinterpret_cast<const uint8_t *>(src);
  for (uint64_t idx = 0; idx < size; idx += sizeof(Buffer::data)) {
    send([=](Buffer *buffer) {
      const uint64_t len =
          size - idx > sizeof(Buffer::data) ? sizeof(Buffer::data) : size - idx;
      inline_memcpy(buffer->data, ptr + idx, len);
    });
  }
}

/// Receives an arbitrarily sized data buffer across the shared channel in
/// multiples of the packet length. The \p alloc function is called with the
/// size of the data so that we can initialize the size of the \p dst buffer.
template <typename Buffer, bool InvertedInboxLoad>
template <typename A>
LIBC_INLINE void PortT<Buffer, InvertedInboxLoad>::recv_n(A alloc) {
  uint64_t size = 0;
  recv([&](Buffer *buffer) { size = buffer->data[0]; });
  uint8_t *dst = reinterpret_cast<uint8_t *>(alloc(size));
  for (uint64_t idx = 0; idx < size; idx += sizeof(Buffer::data)) {
    recv([=](Buffer *buffer) {
      uint64_t len =
          size - idx > sizeof(Buffer::data) ? sizeof(Buffer::data) : size - idx;
      inline_memcpy(dst + idx, buffer->data, len);
    });
  }
}

/// Attempts to open a port to use as the client. The client can only open a
/// port if we find an index that is in a valid sending state. That is, there
/// are send operations pending that haven't been serviced on this port. Each
/// port instance uses an associated \p opcode to tell the server what to do.
LIBC_INLINE cpp::optional<Client::Port> Client::try_open(uint16_t opcode) {
  maybe<port_t<0, 0>> p = Process::try_open();
  if (!p.success)
    return cpp::nullopt;

  shared_buffer->opcode = opcode;
  return Port(*this, 0, 0);
}

LIBC_INLINE Client::Port Client::open(uint16_t opcode) {
  for (;;) {
    if (cpp::optional<Port> p = try_open(opcode))
      return p.value();
    sleep_briefly();
  }
}

/// Attempts to open a port to use as the server. The server can only open a
/// port if it has a pending receive operation
LIBC_INLINE cpp::optional<Server::Port> Server::try_open() {
  uint32_t index = 0;
  uint32_t in = inbox.read_slot(index);
  uint32_t out = outbox.read_slot(index);

  // The server is passive, if there is no work pending don't bother
  // opening a port.
  if (in != out)
    return cpp::nullopt;

  maybe<port_t<0, 0>> p = Process::try_open();
  if (!p.success) {
    return cpp::nullopt;
  }

  return Port(*this, index, 0);
}

LIBC_INLINE Server::Port Server::open() {
  for (;;) {
    if (cpp::optional<Server::Port> p = try_open())
      return p.value();
    sleep_briefly();
  }
}

} // namespace rpc
} // namespace __llvm_libc

#endif
