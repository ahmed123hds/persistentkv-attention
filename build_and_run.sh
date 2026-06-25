#!/usr/bin/env bash
# =============================================================================
# Build and run all micro-benchmarks + smoke test
# GPU: RTX 3060 (SM 8.6), CUDA 12.1
# =============================================================================

set -e
CUDA_ARCH=sm_86
NVCC=nvcc
PYTHON_BIN="${PYTHON_BIN:-python}"
NVCC_FLAGS="-O3 -arch=${CUDA_ARCH} -std=c++17 -lineinfo"
PTXAS_VERBOSE="-Xptxas -v,-dlcm=ca"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"
RESULTS="$ROOT/results"
mkdir -p "$BUILD" "$RESULTS"

echo "============================================================"
echo " PersistentKV-Attention Build + Run"
echo " CUDA arch: $CUDA_ARCH"
echo " Working dir: $ROOT"
echo "============================================================"
echo ""

# ─── Micro-benchmark 1: HBM bandwidth ────────────────────────────────────────
echo ">>> Building 01_hbm_bandwidth..."
$NVCC $NVCC_FLAGS $PTXAS_VERBOSE \
    "$ROOT/micro_benchmarks/01_hbm_bandwidth.cu" \
    -o "$BUILD/01_hbm_bandwidth" 2>&1 | tee "$RESULTS/01_ptxas.txt"
echo ">>> Running 01_hbm_bandwidth..."
"$BUILD/01_hbm_bandwidth" 2>&1 | tee "$RESULTS/01_hbm_bandwidth.txt"
echo ""

# ─── Micro-benchmark 2: GQA load patterns ────────────────────────────────────
echo ">>> Building 02_gqa_load_patterns..."
$NVCC $NVCC_FLAGS $PTXAS_VERBOSE \
    "$ROOT/micro_benchmarks/02_gqa_load_patterns.cu" \
    -o "$BUILD/02_gqa_load_patterns" 2>&1 | tee "$RESULTS/02_ptxas.txt"
echo ">>> Running 02_gqa_load_patterns..."
"$BUILD/02_gqa_load_patterns" 2>&1 | tee "$RESULTS/02_gqa_load_patterns.txt"
echo ""

# ─── Micro-benchmark 3: Register pressure ────────────────────────────────────
echo ">>> Building 03_register_pressure..."
$NVCC $NVCC_FLAGS $PTXAS_VERBOSE \
    "$ROOT/micro_benchmarks/03_register_pressure.cu" \
    -o "$BUILD/03_register_pressure" 2>&1 | tee "$RESULTS/03_ptxas.txt"
echo ">>> Running 03_register_pressure..."
"$BUILD/03_register_pressure" 2>&1 | tee "$RESULTS/03_register_pressure.txt"
echo ""

# ─── Micro-benchmark 4: Double buffering ─────────────────────────────────────
echo ">>> Building 04_double_buffering..."
$NVCC $NVCC_FLAGS $PTXAS_VERBOSE \
    "$ROOT/micro_benchmarks/04_double_buffering.cu" \
    -o "$BUILD/04_double_buffering" 2>&1 | tee "$RESULTS/04_ptxas.txt"
echo ">>> Running 04_double_buffering..."
"$BUILD/04_double_buffering" 2>&1 | tee "$RESULTS/04_double_buffering.txt"
echo ""

# ─── Micro-benchmark 5: L2 hit rate ──────────────────────────────────────────
echo ">>> Building 05_l2_hit_rate..."
$NVCC $NVCC_FLAGS $PTXAS_VERBOSE \
    "$ROOT/micro_benchmarks/05_l2_hit_rate.cu" \
    -o "$BUILD/05_l2_hit_rate" 2>&1 | tee "$RESULTS/05_ptxas.txt"
echo ">>> Running 05_l2_hit_rate..."
"$BUILD/05_l2_hit_rate" 2>&1 | tee "$RESULTS/05_l2_hit_rate.txt"
echo ""

# ─── Full smoke test ──────────────────────────────────────────────────────────
echo ">>> Building smoke_test..."
$NVCC $NVCC_FLAGS $PTXAS_VERBOSE \
    -I"$ROOT" \
    "$ROOT/tests/smoke_test.cu" \
    -o "$BUILD/smoke_test" 2>&1 | tee "$RESULTS/smoke_test_build.txt"
echo ">>> Running smoke_test..."
"$BUILD/smoke_test" 2>&1 | tee "$RESULTS/smoke_test.txt"
echo ""

echo "============================================================"
echo " All done! Results in: $RESULTS/"
echo "============================================================"
echo ""
echo "=== PTXAS register usage summary ==="
for f in "$RESULTS"/*_ptxas.txt; do
    echo "--- $(basename $f) ---"
    grep -E "registers|spills|smem" "$f" 2>/dev/null || echo "(no register info)"
done

echo ""
echo "=== Quick perf summary ==="
grep -E "time=|BW=|Speedup|efficiency" "$RESULTS/smoke_test.txt" 2>/dev/null | head -30

echo ""
echo "To run NSight profiling:"
echo "  ncu --kernel-name-base demangled \\"
echo "      --kernel-name 'regex:.*persistentkv_decode_kernel.*' \\"
echo "      --launch-count 1 \\"
echo "      --metrics dram__bytes_read.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active \\"
echo "      --export $RESULTS/ncu_persistentkv \\"
echo "      $BUILD/smoke_test --profile"
echo ""
echo "To run a targeted memory-safety check:"
echo "  compute-sanitizer --tool memcheck --error-exitcode 1 $BUILD/smoke_test --profile"
echo ""
echo "To run the PyTorch 2.5 SDPA GQA baseline:"
echo "  $PYTHON_BIN \\"
echo "      $ROOT/tests/benchmark_torch_sdpa.py | tee $RESULTS/pytorch_sdpa.txt"
echo ""
echo "To rebuild and run the focused 8K kernel ablations:"
echo "  $ROOT/run_kernel_ablations.sh"
echo ""
echo "To isolate stage costs and paged-layout sensitivity:"
echo "  nvcc $NVCC_FLAGS -I$ROOT $ROOT/tests/kernel_stage_bench.cu -o $BUILD/kernel_stage_bench"
echo "  $BUILD/kernel_stage_bench --stages"
echo "  $BUILD/kernel_stage_bench --paged"
echo "  $BUILD/kernel_stage_bench --paged-long"
echo "  $BUILD/kernel_stage_bench --canonical-paged 16"
echo ""
echo "To benchmark PyTorch materialization of paged KV before SDPA:"
echo "  $PYTHON_BIN \\"
echo "      $ROOT/tests/benchmark_torch_paged_sdpa.py | tee $RESULTS/pytorch_paged_sdpa.txt"
echo ""
echo "To benchmark staged native paged-attention baselines:"
echo "  PATH=\$(dirname \"$PYTHON_BIN\"):\$PATH \\"
echo "  FLASHINFER_SOURCE=/path/to/flashinfer/source \\"
echo "  FLASHINFER_WORKSPACE_BASE=$BUILD/baselines/flashinfer_cache \\"
echo "  TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 \\"
echo "  $PYTHON_BIN \\"
echo "      $ROOT/tests/benchmark_native_paged_baselines.py --page-size 16"
echo ""
echo "To run the first continuous-batching serving trace experiment:"
echo "  PATH=\$(dirname \"$PYTHON_BIN\"):\$PATH \\"
echo "  FLASHINFER_SOURCE=/path/to/flashinfer/source \\"
echo "  FLASHINFER_WORKSPACE_BASE=$BUILD/baselines/flashinfer_cache \\"
echo "  TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 \\"
echo "  $PYTHON_BIN \\"
echo "      $ROOT/tests/benchmark_serving_trace.py --trace bimodal --requests 24 --max-active 8 --steps 48"
echo "  # Try masked range bucketing:"
echo "  # ... benchmark_serving_trace.py --trace bimodal --requests 24 --max-active 8 --steps 48 --hole-fraction 0.0 --bucket-tokens 4096"
