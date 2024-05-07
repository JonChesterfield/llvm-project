; RUN: opt -mtriple=wasm32-unknown-unknown -S --passes=expand-variadics --expand-variadics-override=without-rewriting-calls < %s | FileCheck %s

%struct.__va_list_tag = type { i32, i32, ptr, ptr }

declare void @llvm.va_start.p0(ptr)
declare void @llvm.va_end.p0(ptr)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

declare void @sink(...)

; CHECK: failure
define void @pass_s3(i32 %x.coerce) {
entry:
  tail call void (...) @sink(i32 %x.coerce)
  ret void
}

declare void @valist(ptr noundef)

define void @start_twice(float, ...) {
entry:
  %s0 = alloca [1 x %struct.__va_list_tag], align 16
  %s1 = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %s0) #2
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %s1) #2
  call void @llvm.va_start.p0(ptr nonnull %s0)
  call void @valist(ptr noundef nonnull %s0) #2
  call void @llvm.va_end.p0(ptr %s0)
  call void @llvm.va_start.p0(ptr nonnull %s1)
  call void @valist(ptr noundef nonnull %s1) #2
  call void @llvm.va_end.p0(ptr %s1)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %s1) #2
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %s0) #2
  ret void
}
