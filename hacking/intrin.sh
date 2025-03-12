#!/bin/bash
set -e
set -x
SOURCE=/home/jon/llvm-project
ROOT=/scratch/jon/llvm-build
DIR=/mnt/scratch/intrin

TARGET="nvptx64-nvidia-cuda -Xclang -target-feature -Xclang +ptx62" # 62 happy

rm -f $DIR/builtins_gpuintrin.*.ll  $DIR/reference_gpuintrin.*.ll

$ROOT/llvm/bin/clang -ffreestanding -isystem $SOURCE/clang/lib/Headers/ -O1 -DJC_USE_COMPILER_BUILTINS=0 --target=$TARGET -nogpulib -emit-llvm -S jc_gpuintrin.c -o $DIR/reference_gpuintrin.ptx.ll

$ROOT/llvm/bin/clang -ffreestanding -isystem $SOURCE/clang/lib/Headers/ -O1 -DJC_USE_COMPILER_BUILTINS=1 --target=$TARGET -nogpulib -emit-llvm -S jc_gpuintrin.c -o $DIR/builtins_gpuintrin.ptx.ll

TARGET=amdgcn-amd-amdhsa

$ROOT/llvm/bin/clang -ffreestanding -isystem $SOURCE/clang/lib/Headers/ -O1 -DJC_USE_COMPILER_BUILTINS=0 --target=$TARGET -nogpulib -emit-llvm -S jc_gpuintrin.c -o $DIR/reference_gpuintrin.gcn.ll

$ROOT/llvm/bin/clang -ffreestanding -isystem $SOURCE/clang/lib/Headers/ -O1 -DJC_USE_COMPILER_BUILTINS=1 --target=$TARGET -nogpulib -emit-llvm -S jc_gpuintrin.c -o $DIR/builtins_gpuintrin.gcn.ll



TARGET=spirv64--

$ROOT/llvm/bin/clang -ffreestanding -isystem $SOURCE/clang/lib/Headers/ -O1 -DJC_USE_COMPILER_BUILTINS=0 --target=$TARGET -nogpulib -emit-llvm -S jc_gpuintrin.c -o $DIR/reference_gpuintrin.spv.ll

$ROOT/llvm/bin/clang -ffreestanding -isystem $SOURCE/clang/lib/Headers/ -O1 -DJC_USE_COMPILER_BUILTINS=1 --target=$TARGET -nogpulib -emit-llvm -S jc_gpuintrin.c -o $DIR/builtins_gpuintrin.spv.ll




