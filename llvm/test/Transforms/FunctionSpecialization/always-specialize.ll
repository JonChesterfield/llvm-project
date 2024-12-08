; RUN: opt -S --passes="always-specialize,deadargelim" < %s | FileCheck %s

; CHECK: badger

define dso_local i32 @add(i32 %x, i32 %y) {
entry:
  %add = add nsw i32 %y, %x
  ret i32 %add
}

define internal i32 @f(i32 %x, i32 %y, ptr %u, ptr %v) noinline {
entry:
  %call = tail call i32 %u(i32 %x, i32 %y)
  %call1 = tail call i32 %v(i32 %x, i32 %y)
  %mul = mul nsw i32 %call1, %call
  ret i32 %mul
}

define dso_local i32 @g0(i32 %x, i32 %y) {
; CHECK-LABEL: @g0
; CHECK:       call i32 @f.specialized.3(i32 [[X:%.*]], i32 [[Y:%.*]])
entry:
  %call = tail call i32 @f(i32 %x, i32 %y, ptr @add, ptr @add)
  ret i32 %call
}
