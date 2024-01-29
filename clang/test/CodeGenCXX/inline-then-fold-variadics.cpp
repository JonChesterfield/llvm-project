// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -Wno-varargs -O1 -emit-llvm -o - %s | opt --expand-va-intrinsics | opt -S -O1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -Wno-varargs -O1 -emit-llvm -o - %s | opt --expand-va-intrinsics | opt -S -O1 | FileCheck %s

// The clang test suite has _lots_ of windows related triples in it
// 'x86_64-pc-windows-msvc|i686-windows-msvc|thumbv7-windows|aarch64-windows|i686-windows|x86_64-windows|x86_64-unknown-windows-msvc|i386-windows-pc|x86_64--windows-msvc|i686--windows-msvc|x86_64-unknown-windows-gnu|i686-unknown-windows-msvc|i686-unknown-windows-gnu|arm64ec-pc-windows-msvc19.20.0|i686-pc-windows-msvc19.14.0|i686-pc-windows|x86_64--windows-gnu|i686--windows-gnu|thumbv7--windows|i386-windows|x86_64-unknown-windows-pc|i686--windows|x86_64--windows|i686-w64-windows-gnu'

// Might be detecting an inconsistency - maybe different alignment
// R-N: %clang_cc1 -triple i686-windows-msvc -Wno-varargs -O1 -emit-llvm -o - %s | opt --expand-va-intrinsics | opt -S -O1 | FileCheck %s


// This fails as not-implemented
// MS ABI is very keen on passing-indirectly
// R-N: %clang_cc1 -triple x86_64-pc-windows-msvc -Wno-varargs -O1 -emit-llvm -o - %s | opt --expand-va-intrinsics | opt -S -O1 | FileCheck %s


// Works for first, the ptrtoint fails to fold for second for both of them
// nvptx64-nvidia-cuda
// amdgcn-amd-amdhsa

// Not yet implemented on arm
// Also there are various x86 variants that should be in the triple

// Checks for consistency between clang and expand-va-intrinics
// 1. Use clang to lower va_arg
// 2. Use expand-va-intrinsics to lower the rest of the variadic operations
// 3. Use opt -O1 to simplify the functions to ret %arg
// The simplification to ret %arg will fail when the two are not consistent, modulo bugs elsewhere.

#include <stdarg.h>

template <typename X, typename Y>
static X first(...) {
  va_list va;
  __builtin_va_start(va, 0);
  X r = va_arg(va, X);
  va_end(va);
  return r;
}

template <typename X, typename Y>
static Y second(...) {
  va_list va;
  __builtin_va_start(va, 0);
  va_arg(va, X);
  Y r = va_arg(va, Y);
  va_end(va);
  return r;
}

// Permutations of an int and a double
extern "C"
{
// CHECK-LABEL: define dso_local i32 @first_i32_f64(
// CHECK:       entry:
// CHECK:       ret i32 %x
int first_i32_f64(int x, double y)
{
  return first<int,double>(x, y);
}
  
// CHECK-LABEL: define dso_local double @second_i32_f64(
// CHECK:       entry:
// CHECK:       ret double %y
double second_i32_f64(int x, double y)
{
  return second<int,double>(x, y);
}

// CHECK-LABEL: define dso_local double @first_f64_i32(
// CHECK:       entry:
// CHECK:       ret double %x
double first_f64_i32(double x, int y)
{
  return first<double,int>(x, y);
}

// CHECK-LABEL: define dso_local i32 @second_f64_i32(
// CHECK:       entry:
// CHECK:       ret i32 %y
int second_f64_i32(double x, int y)
{
  return second<double,int>(x, y);
}
    
}

// Non-primitives may be tedious to pattern match as different targets
// do things like convert structs to i64 or use byval etc

extern "C"
{
  // somewhat likely to be passed indirectly
  // likely to want abi-specific checks to match
  struct chararr
  {
    char x[23];
  };
  
  int first_i32_chararr(int x, chararr y)
  {
    return first<int,chararr>(x,y);
  }

  chararr second_i32_chararr(int x, chararr y)
  {
    return second<int,chararr>(x,y);
  }

  chararr first_chararr_i32(chararr x, int y)
  {
    return first<chararr, int>(x, y);
  }

  int second_chararr_i32(chararr x, int y)
  {
    return second<chararr, int>(x, y);
  }

  
}
