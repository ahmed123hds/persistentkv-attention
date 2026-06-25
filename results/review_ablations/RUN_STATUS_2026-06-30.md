# Review Ablation Run Status: 2026-06-30

The reviewer ablation suite was run after the NVIDIA driver mismatch was fixed
and the workspace drive was mounted.

## Environment Check

```text
NVIDIA GeForce RTX 3060, 560 MiB used, 12288 MiB total
```

Benchmark environment:

```text
torch=2.5.1+cu121
cuda=12.1
gpu=NVIDIA GeForce RTX 3060
```

## Command

```bash
cd /media/filliones/12280E43280E25F7/Arxiv/Kernel_Optmization
PATH="/home/filliones/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin:$PATH" \
PYTHON_BIN=/home/filliones/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python \
TORCH_CUDA_ARCH_LIST=8.6 \
MAX_JOBS=4 \
FLASHINFER_SOURCE="$PWD/build/baselines/flashinfer_0_2_5_src" \
FLASHINFER_WORKSPACE_BASE="$PWD/build/baselines/flashinfer_cache" \
TORCH_EXTENSIONS_DIR="$PWD/build/baselines/torch_extensions" \
./scripts/run_review_ablations.sh
```

The five-seed and split-sensitivity sections completed and wrote logs under
`results/review_ablations/`.

The full-size GQA serving sweep stopped at `G=1` because that configuration
uses `Hkv=32` and exceeded the 12 GB VRAM budget on the 24-request trace:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.96 GiB.
```

To still test the GQA/MQA serving behavior, the GQA sweep was rerun on a
smaller B8 trace:

```bash
for hkv in 32 8 4; do
  g=$((32 / hkv))
  "$PYTHON_BIN" tests/benchmark_serving_trace.py \
    --trace bimodal \
    --requests 8 \
    --max-active 8 \
    --steps 24 \
    --repeats 1 \
    --hole-fraction 0.0 \
    --seed 20260623 \
    --precompute-metadata \
    --adaptive-engine \
    --adaptive-workqueue-splits 20 \
    --hkv "$hkv" \
    --output "results/review_ablations/gqa_serving_small_bimodal_b8_g${g}.txt"
done
```

See `results/review_ablations/summary.md` for aggregate numbers.
