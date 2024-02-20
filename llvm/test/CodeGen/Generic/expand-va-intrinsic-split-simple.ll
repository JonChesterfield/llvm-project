; RUN: opt -mtriple=i386-unknown-linux-gnu -S --passes=expand-variadics --expand-variadics-abi=false --expand-variadics-split=true --expand-variadics-calls=false < %s | FileCheck %s

; i386 uses a void* for va_arg
; amdgpu should be the same codegen, nvptx slightly different alignment on the va_arg

; Examples are variadic functions that return the first or the second of an int and a double
; Split the functions into an internal equivalent that takes a va_list and a ABI preserving wrapper

define i32 @variadic_int_double_get_firstz(...) {
entry:
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr nonnull %va)
  %argp.cur = load ptr, ptr %va, align 4
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
  store ptr %argp.next, ptr %va, align 4
  %0 = load i32, ptr %argp.cur, align 4
  call void @llvm.va_end(ptr %va)
  ret i32 %0
}

; CHECK-LABEL: @variadic_int_double_get_firstz.valist(ptr noalias %varargs) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:   %va = alloca ptr, align 4
; CHECK-NEXT:   store ptr %varargs, ptr %va, align 4
; CHECK-NEXT:   %argp.cur = load ptr, ptr %va, align 4
; CHECK-NEXT:   %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
; CHECK-NEXT:   store ptr %argp.next, ptr %va, align 4
; CHECK-NEXT:   %0 = load i32, ptr %argp.cur, align 4
; CHECK-NEXT:   ret i32 %0
; CHECK-NEXT:  }

; CHECK-LABEL: @variadic_int_double_get_firstz(...) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %va_list = alloca ptr, align 4
; CHECK-NEXT:    call void @llvm.va_start(ptr %va_list)
; CHECK-NEXT:    %0 = tail call i32 @variadic_int_double_get_firstz.valist(ptr %va_list)
; CHECK-NEXT:    ret i32 %0
; CHECK-NEXT:  }

define double @variadic_int_double_get_secondz(...) {
entry:
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr nonnull %va)
  %argp.cur = load ptr, ptr %va, align 4
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
  %argp.next2 = getelementptr inbounds i8, ptr %argp.cur, i32 12
  store ptr %argp.next2, ptr %va, align 4
  %0 = load double, ptr %argp.next, align 4
  call void @llvm.va_end(ptr %va)
  ret double %0
}

; CHECK-LABEL: @variadic_int_double_get_secondz.valist(ptr noalias %varargs) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %va = alloca ptr, align 4
; CHECK-NEXT:    store ptr %varargs, ptr %va, align 4
; CHECK-NEXT:    %argp.cur = load ptr, ptr %va, align 4
; CHECK-NEXT:    %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
; CHECK-NEXT:    %argp.next2 = getelementptr inbounds i8, ptr %argp.cur, i32 12
; CHECK-NEXT:    store ptr %argp.next2, ptr %va, align 4
; CHECK-NEXT:    %0 = load double, ptr %argp.next, align 4
; CHECK-NEXT:    ret double %0
; CHECK-NEXT:  }

; CHECK-LABEL: @variadic_int_double_get_secondz(...) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %va_list = alloca ptr, align 4
; CHECK-NEXT:    call void @llvm.va_start(ptr %va_list)
; CHECK-NEXT:    %0 = tail call double @variadic_int_double_get_secondz.valist(ptr %va_list)
; CHECK-NEXT:    ret double %0
; CHECK-NEXT:  }

;; Two call sites are unchanged for this test case because rewrite calls is false
;; They still refer to the ... external function

; CHECK-LABEL: @variadic_can_get_firstIidEEbT_T0_(i32 %x, double %y) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %call = call i32 (...) @variadic_int_double_get_firstz(i32 %x, double %y)
; CHECK-NEXT:    %cmp.i = icmp eq i32 %call, %x
; CHECK-NEXT:    ret i1 %cmp.i
; CHECK-NEXT:  }
define zeroext i1 @variadic_can_get_firstIidEEbT_T0_(i32 %x, double %y) {
entry:
  %call = call i32 (...) @variadic_int_double_get_firstz(i32 %x, double %y)
  %cmp.i = icmp eq i32 %call, %x
  ret i1 %cmp.i
}

; CHECK-LABEL: @variadic_can_get_secondIidEEbT_T0_(i32 %x, double %y) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %call = call double (...) @variadic_int_double_get_secondz(i32 %x, double %y)
; CHECK-NEXT:    %cmp.i = fcmp oeq double %call, %y
; CHECK-NEXT:    ret i1 %cmp.i
; CHECK-NEXT:  }

define zeroext i1 @variadic_can_get_secondIidEEbT_T0_(i32 %x, double %y) {
entry:
  %call = call double (...) @variadic_int_double_get_secondz(i32 %x, double %y)
  %cmp.i = fcmp oeq double %call, %y
  ret i1 %cmp.i
}

; Declaration unchanged
; CHECK: declare void @variadic_without_callers(...)
declare void @variadic_without_callers(...)

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)
