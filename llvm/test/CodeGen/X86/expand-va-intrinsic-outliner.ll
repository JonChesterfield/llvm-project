; RUN: opt -mtriple=i386-unknown-linux-gnu -S --passes=expand-variadics --expand-variadics-abi=false --expand-variadics-split=true --expand-variadics-calls=false < %s | FileCheck %s -check-prefix=X86
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -S --passes=expand-variadics --expand-variadics-abi=false --expand-variadics-split=true --expand-variadics-calls=false < %s | FileCheck %s -check-prefix=X64


declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

declare void @sink_valist(ptr)
declare void @sink_i32(i32)


;; Simple function that cannot be used as-is by the pass and is thus split into two functions
; X86-LABEL: define internal void @x86_non_inlinable.valist(
; X86:       entry:
; X86:       %va = alloca ptr, align 4
; X86:       call void @sink_i32(i32 0)
; X86:       store ptr %varargs, ptr %va, align 4
; X86:       %0 = load ptr, ptr %va, align 4
; X86:       call void @sink_valist(ptr noundef %0)
; X86:       ret void
; X86:     }
; X86-LABEL: define void @x86_non_inlinable(
; X86:       entry:
; X86:       %va_list = alloca ptr, align 4
; X86:       call void @llvm.va_start(ptr %va_list)
; X86:       tail call void @x86_non_inlinable.valist(ptr %va_list)
; X86:       ret void
; X86:       }
define void @x86_non_inlinable(...)  {
entry:
  %va = alloca ptr, align 4
  call void @sink_i32(i32 0)
  call void @llvm.va_start(ptr nonnull %va)
  %0 = load ptr, ptr %va, align 4
  call void @sink_valist(ptr noundef %0)
  ret void
}

;; As above, but for x64 - the different va_list type means a missing load.

; X64-LABEL: define internal void @x64_non_inlinable.valist(
; X64:       entry:
; X64:       %va = alloca [1 x %struct.__va_list_tag], align 16
; X64:       call void @sink_i32(i32 0)
; X64:       call void @llvm.memcpy.inline.p0.p0.i32(ptr %va, ptr %varargs, i32 24, i1 false)
; X64:       call void @sink_valist(ptr noundef %va)
; X64:       ret void
; X64:     }
; X64-LABEL: define void @x64_non_inlinable(
; X64:       entry:
; X64:       %va_list = alloca [1 x { i32, i32, ptr, ptr }], align 8
; X64:       call void @llvm.va_start(ptr %va_list)
; X64:       tail call void @x64_non_inlinable.valist(ptr %va_list)
; X64:       ret void
; X64:       }
define void @x64_non_inlinable(...)  {
entry:
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @sink_i32(i32 0)
  call void @llvm.va_start(ptr nonnull %va)
  call void @sink_valist(ptr noundef %va)
  ret void
}


; Following cases are functions which can be used by the expander as-is
; so are not split.

; X86-LABEL: @x86_maximal_inlinable(
; X86-NOT:   @x86_maximal_inlinable.valist(
define void @x86_maximal_inlinable(...)  {
entry:
  %va = alloca ptr, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %va)
  call void @llvm.va_start(ptr nonnull %va)
  %0 = load ptr, ptr %va, align 4
  call void @sink_valist(ptr noundef %0)
  call void @llvm.va_end(ptr %va)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %va)
  ret void
}

; X86-LABEL: @x86_nolifetime_inlinable(
; X86-NOT:   @x86_nolifetime_inlinable.valist(
define void @x86_nolifetime_inlinable(...)  {
entry:
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr nonnull %va)
  %0 = load ptr, ptr %va, align 4
  call void @sink_valist(ptr noundef %0)
  call void @llvm.va_end(ptr %va)
  ret void
}

; X86-LABEL: @x86_minimal_inlinable(
; X86-NOT:   @x86_minimal_inlinable.valist(
define void @x86_minimal_inlinable(...)  {
entry:
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr nonnull %va)
  %0 = load ptr, ptr %va, align 4
  call void @sink_valist(ptr noundef %0)
  ret void
}



%struct.__va_list_tag = type { i32, i32, ptr, ptr }

; X64-LABEL: @x64_maximal_inlinable(
; X64-NOT:   @x64_maximal_inlinable.valist(
define void @x64_maximal_inlinable( ...) {
entry:
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %va)
  call void @llvm.va_start(ptr nonnull %va)
  call void @sink_valist(ptr noundef nonnull %va)
  call void @llvm.va_end(ptr %va)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %va)
  ret void
}

; X64-LABEL: @x64_nolifetime_inlinable(
; X64-NOT:   @x64_nolifetime_inlinable.valist(
define void @x64_nolifetime_inlinable( ...) {
entry:
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr nonnull %va)
  call void @sink_valist(ptr noundef nonnull %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

; X64-LABEL: @x64_minimal_inlinable(
; X64-NOT:   @x64_minimal_inlinable.valist(
define void @x64_minimal_inlinable( ...) {
entry:
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr nonnull %va)
  call void @sink_valist(ptr noundef nonnull %va)
  ret void
}

