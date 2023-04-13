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

template <typename T> struct maybe {
  T port;
  bool success;
};

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
  bitmap_t<InvertedInboxLoadT, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES> inbox;
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

  port_t<0, 0> open(ThreadMask active_threads) {
    for (;;) {
      maybe<port_t<0, 0>> r = try_open(active_threads);
      if (r.success) {
        return r.port;
      }
      sleep_briefly();
    }
  }

  maybe<port_t<0, 0>> try_open(ThreadMask active_threads) {

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
  void use(port_t<IandO, IandO> port, Op op) {
    // Op could be && and forwarded, but Client::run takes it by value
    uint32_t raw = port.value;
    // todo: this depends on how buffer is extended to multiple warps
    uint64_t active_threads = 1;
    if (is_first_lane(active_threads)) {
      op(&shared_buffer[raw]);
    }
  }

  template <unsigned IandO, typename Op, typename Extra>
  Extra use(port_t<IandO, IandO> port, Extra e, Op op) {
    uint32_t raw = port.value;
    // todo: this depends on how buffer is extended to multiple warps
    uint64_t active_threads = 1;
    Extra r = {};
    if (is_first_lane(active_threads)) {
      r = op(e, &shared_buffer[raw]);
    }
    r = broadcast_first_lane(active_threads, r); // wont need this once threaded
    return r;
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

  template <unsigned IandO>
  port_t<!IandO, !IandO> send_then_wait(port_t<IandO, IandO> port) {
    port_t<IandO, !IandO> tmp = send(port);
    return wait(tmp);
  }
};

template <unsigned M, unsigned I, unsigned O> struct stream_port_t {
  static_assert(I == 0 || I == 1, "");
  static_assert(O == 0 || O == 1, "");
  static_assert(M == 0 || M == 1, "");
  uint32_t value;

  stream_port_t(uint32_t value) : value(value) {}
};

// Would lose a bunch of boilerplate if StreamProcess inherited from Process
// A process instance with an extra bit of state for each slot
template <typename BufferElement, uint32_t WarpSize, uint32_t NumberSlots,
          bool InvertedInboxLoadT>
struct StreamProcess {
  using ProcessType =
      Process<BufferElement, WarpSize, NumberSlots, InvertedInboxLoadT>;
  ProcessType core;
  bitmap_t<false, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES> MORE_BITS;

  using ThreadMask = typename ProcessType::ThreadMask;

  LIBC_INLINE void reset(void *locks, void *inbox, void *outbox, void *buffer,
                         void *extra) {
    core.reset(locks, inbox, outbox, buffer);
    MORE_BITS = reinterpret_cast<cpp::Atomic<uint32_t> *>(extra);
  }


  port_t<0, 0> open(ThreadMask active_threads) {
    return core.open(active_threads);
  }
  
  template <unsigned M>
  stream_port_t<M, 0, 0> open_stream(ThreadMask active_threads) {
    for (;;) {
      maybe<stream_port_t<M, 0, 0>> res = try_open_stream(active_threads);
      if (res.success) {
        return res.port;
      }
    }
  }

  template <unsigned M>
  maybe<stream_port_t<M, 0, 0>> try_open_stream(ThreadMask active_threads) {

    maybe<port_t<0, 0>> maybe_port = core.try_open(active_threads);

    if (maybe_port.success) {
      if (MORE_BITS.read_slot(maybe_port.port.value) == M) {
        return {maybe_port.port.value, true};
      } else {
        core.close(maybe_port.port);
      }
    }
    return {UINT32_MAX, false};
  }

  template <unsigned I, unsigned O> void close(port_t<I, O> port) {
    core.close(port);
  }

  template <unsigned M, unsigned I, unsigned O>
  void close(stream_port_t<M, I, O> mport) {
    port_t<I, O> port = mport.value;
    core.close(port);
  }

  template <unsigned M, unsigned IandO>
  bool read_state_stream(stream_port_t<M, IandO, IandO> port) {
    return MORE_BITS.read_slot(port.value);
  }

  template <unsigned IandO>
  stream_port_t<0, IandO, IandO>
  close_stream(stream_port_t<1, IandO, IandO> port) {
    MORE_BITS.release_slot(port.value);
    return port.value;
  }

  template <unsigned IandO>
  stream_port_t<1, IandO, IandO>
  start_stream(stream_port_t<0, IandO, IandO> port) {
    MORE_BITS.claim_slot(port.value);
    return port.value;
  }

  template <unsigned IandO, typename Op>
  void use(port_t<IandO, IandO> port, Op op) {
    core.use(port, op);
  }

  template <unsigned IandO, typename Op, typename Extra>
  Extra use(port_t<IandO, IandO> port, Extra e, Op op) {
    return core.use(port, e, op);
  }

  
  template <unsigned M, unsigned IandO, typename Op>
  bool use_stream(stream_port_t<M, IandO, IandO> port, Op op) {
    return core.use(port_t<IandO, IandO>(port.value), M, op);
  }

  template <unsigned I> port_t<!I, !I> wait(port_t<I, !I> port) {
    return core.wait(port);
  }

  template <unsigned M, unsigned I>
  stream_port_t<M, !I, !I> wait(stream_port_t<M, I, !I> port) {
    port_t<I, !I> tmp(port.value);
    port_t<!I, !I> res = core.wait(tmp);
    return res.value;
  }

  template <unsigned IandO>
  port_t<IandO, !IandO> send(port_t<IandO, IandO> port) {
    return core.send(port);
  }

  template <unsigned M, unsigned IandO>
  stream_port_t<M, IandO, !IandO> send(stream_port_t<M, IandO, IandO> port) {
    port_t<IandO, IandO> tmp(port.value);
    port_t<IandO, !IandO> res = core.send(tmp);
    return res.value;
  }

  template <unsigned IandO>
  port_t<!IandO, !IandO> send_then_wait(port_t<IandO, IandO> port) {
    return core.send_then_wait(port);
  }

  template <unsigned M, unsigned IandO>
  stream_port_t<M, !IandO, !IandO>
  send_then_wait(stream_port_t<M, IandO, IandO> port) {
    port_t<IandO, IandO> tmp(port.value);
    port_t<!IandO, !IandO> res = core.send_then_wait(tmp);
    return res.value;
  }
};

/// The RPC client used to make requests to the server.
struct Client : public Process<Buffer, 32, 1, false> {
  LIBC_INLINE Client() = default;
  LIBC_INLINE Client(const Client &) = default;
  LIBC_INLINE Client &operator=(const Client &) = default;
  LIBC_INLINE ~Client() = default;

  template <typename F, typename U> LIBC_INLINE void run(F fill, U use);
  template <typename F> LIBC_INLINE void run_async(F fill);
};

struct StreamClient : public StreamProcess<Buffer, 32, 1, false> {
  LIBC_INLINE StreamClient() = default;
  LIBC_INLINE StreamClient(const StreamClient &) = default;
  LIBC_INLINE StreamClient &operator=(const StreamClient &) = default;
  LIBC_INLINE ~StreamClient() = default;

  template <typename F, typename U> LIBC_INLINE void run(F fill, U use);
  template <typename F> LIBC_INLINE void run_async(F fill);

  template <typename F, typename U> LIBC_INLINE void run_multi(F fill, U use);
};

/// The RPC server used to respond to the client.
struct Server : public Process<Buffer, 32, 1, true> {
  LIBC_INLINE Server() = default;
  LIBC_INLINE Server(const Server &) = default;
  LIBC_INLINE Server &operator=(const Server &) = default;
  LIBC_INLINE ~Server() = default;

  template <typename W, typename C> LIBC_INLINE bool handle(W work, C clean);
};

struct StreamServer : public StreamProcess<Buffer, 32, 1, true> {
  LIBC_INLINE StreamServer() = default;
  LIBC_INLINE StreamServer(const StreamServer &) = default;
  LIBC_INLINE StreamServer &operator=(const StreamServer &) = default;
  LIBC_INLINE ~StreamServer() = default;

  template <typename W, typename C> LIBC_INLINE bool handle(W work, C clean);
};

namespace detail {
template <typename P, typename F, typename U>
void run(P &process, F fill, U use) {
  /// Run the RPC client protocol to communicate with the server. We perform the
  /// following high level actions to complete a communication:
  ///   - Apply \p fill to the shared buffer and write 1 to the outbox.
  ///   - Wait until the inbox is 1.
  ///   - Apply \p use to the shared buffer and write 0 to the outbox.
  ///   - Wait until the inbox is 0.

  uint32_t ThreadMask = 1;

  port_t<0, 0> port0 = process.open(ThreadMask);

  // Apply the \p fill to the buffer and signal the server.
  process.use(port0, fill);
  port_t<0, 1> port1 = process.send(port0);

  // Wait for the server to work on the buffer and respond.
  port_t<1, 1> port2 = process.wait(port1);

  // Apply \p use to the buffer and signal the server.
  process.use(port2, use);
  port_t<1, 0> port3 = process.send(port2);

  // Wait for the server to signal the end of the protocol.
  process.close(port3);
}

template <typename P, typename F> void run_async(P &process, F fill) {
  uint32_t ThreadMask = 1;
  port_t<0, 0> port0 = process.open(ThreadMask);
  // Apply the \p fill to the buffer and signal the server.
  process.use(port0, fill);
  port_t<0, 1> port1 = process.send(port0);
  process.close(port1);
}

} // namespace detail

template <typename F, typename U> LIBC_INLINE void Client::run(F fill, U use) {
  detail::run(*this, fill, use);
}
template <typename F> LIBC_INLINE void Client::run_async(F fill) {
  detail::run_async(*this, fill);
}

template <typename F, typename U>
LIBC_INLINE void StreamClient::run(F fill, U use) {
  detail::run(*this, fill, use);
}
template <typename F> LIBC_INLINE void StreamClient::run_async(F fill) {
  detail::run_async(*this, fill);
}

template <typename F, typename U>
LIBC_INLINE void StreamClient::run_multi(F fill, U use) {
  uint32_t ThreadMask = 1;

  // Control flow is driven by the fill callback.
  // Returns true starts a stream, return false ends it.

  stream_port_t<0, 0, 0> port0 = open_stream<0>(ThreadMask);

  bool s = use_stream(port0, fill);

  if (!s) {
    // fill did not want to start a stream, handle as a normal call
    stream_port_t<0, 1, 1> port1 = wait(send(port0));
    // retrieving arguments doesn't get to start a stream
    use_stream(port1, use);
    stream_port_t<0, 1, 0> tmp2 = send(port1);
    close(tmp2);
    return;
  }

  stream_port_t<1, 0, 0> port1 = start_stream(port0);
  while (s) {
    stream_port_t<1, 1, 1> tmp1 = send_then_wait(port1);
    s = use_stream(tmp1, fill);

    if (s) {
      // keep going
      stream_port_t<1, 1, 0> tmp2 = send(tmp1);
      port1 = wait(tmp2);
      s = use_stream(port1, fill);
      continue;
    }

    // that was the last packet in the stream, send it
    // then handle the return value
    stream_port_t<0, 1, 1> tmp2 = close_stream(tmp1);
    stream_port_t<0, 0, 0> tmp3 = send_then_wait(tmp2);

    use_stream(tmp3, use);
    stream_port_t<0, 0, 1> tmp4 = send(tmp3);
    close(tmp4);
    return;
  }

  // Just called use_stream for the last time, send it then handle return value
  stream_port_t<0, 0, 0> tmp2 = close_stream(port1);
  stream_port_t<0, 1, 1> tmp3 = send_then_wait(tmp2);
  use_stream(tmp3, use);
  stream_port_t<0, 1, 0> tmp4 = send(tmp3);
  close(tmp4);
  return;
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
  uint32_t active_threads = 1;

  auto maybe_port = try_open(active_threads);
  // There is no work to do, exit early.
  if (!maybe_port.success)
    return false;

  port_t<0, 0> port0 = maybe_port.port;

  // Apply \p work to the buffer and signal the client.
  use(port0, work);
  port_t<0, 1> port1 = send(port0);

  // Wait for the client to use the buffer and respond.
  port_t<1, 1> port2 = wait(port1);

  // Clean up the buffer and signal the end of the protocol.
  use(port2, clean);
  port_t<1, 0> port3 = send(port2);

  close(port3);
  return true;
}

template <typename W, typename C>
LIBC_INLINE bool StreamServer::handle(W work, C clean) {
  uint32_t active_threads = 1;

  auto maybe_simple_port = try_open_stream<0>(active_threads);
  if (maybe_simple_port.success) {
    // as in Server::handle

    stream_port_t<0, 0, 0> port0 = maybe_simple_port.port;

    use_stream(port0, work);
    stream_port_t<0, 1, 1> port1 = send_then_wait(port0);

    use_stream(port1, clean);
    stream_port_t<0, 1, 0> port2 = send(port1);

    close(port2);
    return true;
  }

  auto maybe_port = try_open_stream<1>(active_threads);
  if (!maybe_port.success) {
    return false;
  }
  stream_port_t<1, 0, 0> port0 = maybe_port.port;

  bool stream_state = true;

  while (stream_state) {
    use_stream(port0, work);
    stream_port_t<1, 1, 1> port1 = send_then_wait(port0);
    stream_state = read_state_stream(port1);

    if (stream_state) {
      // keep going
      use_stream(port1, work);
      stream_port_t<1, 0, 0> port2 = send_then_wait(port1);
      stream_state = read_state_stream(port2);
      port0 = port2;
      continue;
    }

    // Reached the end
    stream_port_t<0, 1, 1> port2 = port1.value;
    use_stream(port2, clean);
    stream_port_t<0, 1, 0> port3 = send(port2);
    close(port3);
    return true;
  }

  // Also reached the end
  stream_port_t<0, 1, 1> port1 = port0.value;
  use_stream(port1, clean);
  stream_port_t<0, 1, 0> port2 = send(port1);
  close(port2);
  return true;
}

} // namespace rpc
} // namespace __llvm_libc

#endif
