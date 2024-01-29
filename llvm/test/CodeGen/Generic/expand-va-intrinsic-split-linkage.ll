; RUN: opt -mtriple=i386-unknown-linux-gnu -S --expand-va-intrinsics --expand-va-intrinsics-abi=false --expand-va-intrinsics-split=true --expand-va-intrinsics-calls=false < %s | FileCheck %s --check-prefixes=OPT
; RUN: opt -mtriple=i386-unknown-linux-gnu -S --expand-va-intrinsics --expand-va-intrinsics-abi=true --expand-va-intrinsics-split=true --expand-va-intrinsics-calls=false < %s | FileCheck %s --check-prefixes=ABI

; Split variadic functions into two functions:
; - one equivalent to the original, same symbol etc
; - one implementing the contents of the original but taking a valist
; IR here is applicable to any target that uses a ptr for valist
;
; Defines a function with each linkage (in the order of the llvm documentation).
; If split applies it does the same transform to each.
; Whether split applies depends on whether the ABI is being changed or not - e.g. a weak
; function is not normally useful to split as the contents cannot be called from elsewhere.
; If the ABI is being rewritten then the function is still converted. Call sites tested elsewhere.

declare void @sink_valist(ptr)
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

declare void @decl_simple(...)
define void @defn_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

; no declare for private
define private void @defn_private_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

; no declare for internal
define internal void @defn_internal_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

; no declare for available_externally
define available_externally void @available_externally_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

; no declare for linkonce
define linkonce void @defn_linkonce_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

; no declare for weak
define weak void @defn_weak_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

; common is not applicable to functions
; appending is not applicable to functions

declare extern_weak void @decl_extern_weak_simple(...)
; no define for extern_weak

; no declare for linkonce_odr
define linkonce_odr void @defn_linkonce_odr_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

; no declare for weak_odr
define weak_odr void @defn_weak_odr_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}

declare external void @decl_external_simple(...)
define external void @defn_external_simple(...) {
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  call void @sink_valist(ptr %va)
  call void @llvm.va_end(ptr %va)
  ret void
}



