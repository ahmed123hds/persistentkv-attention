# PersistentKV

PersistentKV is a CUDA/PyTorch research artifact for page-aware decode
scheduling in long-context LLM serving. The code studies when a native
block-table attention engine with sequence splitting and compact work queues can
improve end-to-end decode-step throughput over a strong FlashInfer native-paged
baseline.

Public repository target: <https://github.com/filliones/persistentkv-attention>

The current workshop draft is in
[`paper/persistentkv_mlsys_workshop.pdf`](paper/persistentkv_mlsys_workshop.pdf),
with source in
[`paper/persistentkv_mlsys_workshop.tex`](paper/persistentkv_mlsys_workshop.tex).

## What This Repo Contains

- `kernels/persistentkv_attention.cuh`: native block-table decode attention
  kernels and dispatch code.
- `tests/persistentkv_torch_extension.cu`: PyTorch extension bindings used by
  the serving benchmark.
- `tests/benchmark_serving_trace.py`: continuous decode trace harness comparing
  FlashInfer and PersistentKV routes.
- `tests/benchmark_native_paged_baselines.py`: isolated native-paged baseline
  harness for FlashInfer, vLLM, TensorRT-LLM, PersistentKV, and repack+SDPA
  when the corresponding baseline libraries are available.
- `micro_benchmarks/`: low-level CUDA experiments used during kernel
  development.
- `results/`: representative result logs from the RTX 3060 environment.

This is not a full LLM server. It is a kernel-scheduler artifact focused on
decode attention, page-table layout, and serving-step scheduling.

## Requirements

The reported numbers were measured on:

- NVIDIA RTX 3060, 12 GB VRAM
- CUDA 12.1
- PyTorch 2.5.1 + cu121
- FlashInfer 0.2.5
- Python 3.10+

The C++ smoke tests require `nvcc`. The PyTorch benchmarks require a CUDA-enabled
PyTorch install and FlashInfer. FlashInfer can be either installed as a Python
package or provided as a source checkout through `FLASHINFER_SOURCE`.

Example environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install FlashInfer 0.2.5 using the installation method appropriate for your
# CUDA/PyTorch build, or set FLASHINFER_SOURCE to a local checkout.
export FLASHINFER_SOURCE=/path/to/flashinfer_0_2_5_src
```

## Quick Smoke Test

```bash
export PYTHON_BIN="$(which python)"
export TORCH_CUDA_ARCH_LIST=8.6
./build_and_run.sh
```

This builds the CUDA microbenchmarks and `tests/smoke_test.cu`, then writes logs
under `results/`.

## Reproduce The Main Serving Rows

All commands below use the held-out seed from the paper. They rebuild the
PyTorch extension on first use and write result logs under `results/`.

```bash
export PYTHON_BIN="$(which python)"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
export FLASHINFER_SOURCE=/path/to/flashinfer_0_2_5_src
export FLASHINFER_WORKSPACE_BASE="$PWD/build/baselines/flashinfer_cache"
export TORCH_EXTENSIONS_DIR="$PWD/build/baselines/torch_extensions"
```

B1 bucketed long-context route:

```bash
"$PYTHON_BIN" tests/benchmark_serving_trace.py \
  --trace bucketed \
  --requests 16 \
  --max-active 1 \
  --steps 64 \
  --repeats 3 \
  --hole-fraction 0.0 \
  --seed 20260623 \
  --precompute-metadata \
  --adaptive-engine \
  --adaptive-bucket-splits 32 \
  --output results/reproduce_bucketed_b1.txt
```

B8 bimodal workqueue route:

```bash
"$PYTHON_BIN" tests/benchmark_serving_trace.py \
  --trace bimodal \
  --requests 24 \
  --max-active 8 \
  --steps 48 \
  --repeats 1 \
  --hole-fraction 0.0 \
  --seed 20260623 \
  --precompute-metadata \
  --adaptive-engine \
  --adaptive-workqueue-splits 20 \
  --output results/reproduce_bimodal_b8.txt
```

B8 uniform workqueue route:

```bash
"$PYTHON_BIN" tests/benchmark_serving_trace.py \
  --trace uniform \
  --requests 24 \
  --max-active 8 \
  --steps 48 \
  --repeats 1 \
  --hole-fraction 0.0 \
  --seed 20260623 \
  --precompute-metadata \
  --adaptive-engine \
  --adaptive-workqueue-splits 24 \
  --output results/reproduce_uniform_b8.txt
```

B8 Zipf-like workqueue route:

```bash
"$PYTHON_BIN" tests/benchmark_serving_trace.py \
  --trace zipf \
  --requests 24 \
  --max-active 8 \
  --steps 48 \
  --repeats 1 \
  --hole-fraction 0.0 \
  --seed 20260623 \
  --precompute-metadata \
  --adaptive-engine \
  --adaptive-workqueue-splits 28 \
  --output results/reproduce_zipf_b8.txt
```

B4 boundary case, which should route to FlashInfer:

```bash
"$PYTHON_BIN" tests/benchmark_serving_trace.py \
  --trace bimodal \
  --requests 16 \
  --max-active 4 \
  --steps 48 \
  --repeats 1 \
  --hole-fraction 0.0 \
  --seed 20260623 \
  --precompute-metadata \
  --adaptive-engine \
  --adaptive-workqueue-splits 20 \
  --output results/reproduce_bimodal_b4.txt
```

Expected behavior on a similar RTX 3060 setup: correctness should pass with max
absolute error well below `2e-3`. Throughput ratios will vary with driver,
clocks, background GPU load, and FlashInfer build details.

## Reviewer Ablations

The workshop paper reports one held-out seed and fixes `G=4` for the serving
experiments. To run the extra reviewer-facing checks, use:

```bash
export PYTHON_BIN="$(which python)"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
export FLASHINFER_SOURCE=/path/to/flashinfer_0_2_5_src
export FLASHINFER_WORKSPACE_BASE="$PWD/build/baselines/flashinfer_cache"
export TORCH_EXTENSIONS_DIR="$PWD/build/baselines/torch_extensions"

./scripts/run_review_ablations.sh
```

This emits logs under `results/review_ablations/` for:

- five held-out seeds for the main B1/B8 routes;
- split sensitivity around the calibrated split counts;
- a serving GQA/MQA sweep for `G=1,4,8`.

## Native-Paged Baseline Table

The isolated baseline harness expects optional local baseline artifacts:

- FlashInfer source or installed package
- vLLM extension at `VLLM_LIBRARY`, if measuring vLLM
- TensorRT-LLM source at `TRTLLM_SOURCE`, if measuring TensorRT-LLM

Run:

```bash
export FLASHINFER_SOURCE=/path/to/flashinfer_0_2_5_src
export VLLM_LIBRARY=/path/to/vllm/_C.abi3.so
export TRTLLM_SOURCE=/path/to/TensorRT-LLM

"$PYTHON_BIN" tests/benchmark_native_paged_baselines.py \
  --page-size 16 | tee results/reproduce_native_paged_baselines.txt
```

If vLLM or TensorRT-LLM artifacts are unavailable, use the serving-trace
benchmark above as the primary reproducible path.

## Rebuild The Paper

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error persistentkv_mlsys_workshop.tex
pdflatex -interaction=nonstopmode -halt-on-error persistentkv_mlsys_workshop.tex
```

The PDF is expected to compile to seven pages in the current format.

## Current Claim

PersistentKV is not presented as a universally faster attention kernel.
FlashInfer remains the strongest isolated single-request native-paged baseline
in this artifact. The paper claim is narrower: a calibrated page-aware policy
can route long-context serving steps to PersistentKV sequence splitting or
workqueue scheduling in regimes where that scheduling improves synchronized
decode-step throughput, while using FlashInfer for boundary cases where
PersistentKV overhead dominates.
