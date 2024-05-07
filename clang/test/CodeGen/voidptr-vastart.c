// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// CHECK: Delete this before committing
// $HOME/llvm-build/llvm/bin/clang -cc1 -std=c23  -nostdsysteminc -triple wasm32-unknown-unknown -O1 -emit-llvm  $HOME/llvm-project/clang/test/CodeGen/voidptr-vastart.c -o $HOME/llvm-project/llvm/test/CodeGen/WebAssembly/vararg-frame.ll

void sink(...);

// Scalar types of increasing power two size, where size equal to alignment
typedef struct {
} s0;

typedef struct {
  char x;
} s1;

typedef struct {
  short x;
} s2;

typedef struct {
  int x;
} s3;

typedef struct {
  long long int x;
} s4;

typedef int s5 __attribute__((__vector_size__(16), __aligned__(16)));

void pass_s0(s0 x) {sink(x);}
void pass_s1(s1 x) {sink(x);}
void pass_s2(s2 x) {sink(x);}
void pass_s3(s3 x) {sink(x);}
void pass_s4(s4 x) {sink(x);}
void pass_s5(s5 x) {sink(x);}

void pass_int_s0(int i, s0 x) {sink(i, x);}
void pass_int_s1(int i, s1 x) {sink(i, x);}
void pass_int_s2(int i, s2 x) {sink(i, x);}
void pass_int_s3(int i, s3 x) {sink(i, x);}
void pass_int_s4(int i, s4 x) {sink(i, x);}
void pass_int_s5(int i, s5 x) {sink(i, x);}


void pass_asc(s0 x0,
              s1 x1,
              s2 x2,
              s3 x3,
              s4 x4,
              s5 x5)
{
  sink(x0, x1, x2, x3, x4, x5);
}

void pass_dsc(s5 x0,
              s4 x1,
              s3 x2,
              s2 x3,
              s1 x4,
              s0 x5)
{
  sink(x0, x1, x2, x3, x4, x5);
}

void pass_multiple(int i,
                   s0 x0,
                   s1 x1,
                   s2 x2,
                   s3 x3,
                   s4 x4,
                   s5 x5)
{
  sink(i, x0, x2, x4);
  sink(i, x1, x3, x5);
}
