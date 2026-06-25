#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build/ablations"
RESULTS="$ROOT/results/kernel_ablations.txt"
NVCC="${NVCC:-nvcc}"
COMMON=(-O3 -arch=sm_86 -std=c++17 -lineinfo -I"$ROOT")
SOURCE="$ROOT/tests/smoke_test.cu"

mkdir -p "$BUILD"

build_variant() {
    local name="$1"
    shift
    "$NVCC" "${COMMON[@]}" "$@" "$SOURCE" -o "$BUILD/$name"
}

build_variant precise_transpose -DPKV_FAST_EXP=0 -DPKV_USE_WMMA=0
build_variant fast_transpose -DPKV_FAST_EXP=1 -DPKV_USE_WMMA=0
build_variant precise_direct -DPKV_FAST_EXP=0 -DPKV_TRANSPOSE_K=0 \
    -DPKV_USE_WMMA=0
build_variant fast_direct -DPKV_FAST_EXP=1 -DPKV_TRANSPOSE_K=0 \
    -DPKV_USE_WMMA=0
build_variant fast_wmma_scalar_v -DPKV_FAST_EXP=1 -DPKV_USE_WMMA=1 \
    -DPKV_USE_WMMA_V=0
build_variant fast_wmma_value -DPKV_FAST_EXP=1 -DPKV_USE_WMMA=1 \
    -DPKV_USE_WMMA_V=1
build_variant fast_wmma_value_single_buffer \
    -DPKV_FAST_EXP=1 -DPKV_USE_WMMA=1 -DPKV_USE_WMMA_V=1 \
    -DPKV_DOUBLE_BUFFER_V=0

{
    echo "PersistentKV 8K ablations"
    echo "Shape: B=1 Hq=32 Hkv=8 G=4 N=8192 d=128 FP16"
    echo
    for name in \
        precise_transpose fast_transpose precise_direct fast_direct \
        fast_wmma_scalar_v fast_wmma_value \
        fast_wmma_value_single_buffer
    do
        echo "=== $name ==="
        "$BUILD/$name" --target-splits 16
    done
} | tee "$RESULTS"
