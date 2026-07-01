# Nsight Compute Profile Summary

Environment: RTX 3060 12 GB, driver 580.159.03, CUDA 12.1, PyTorch 2.5.1+cu121,
Nsight Compute 2023.1.0.

Profiling required admin access because the driver reported
`RmProfilingAdminOnly: 1`. Captures use a short B8 bimodal G=4 trace:

```bash
tests/benchmark_serving_trace.py \
  --trace bimodal \
  --requests 8 \
  --max-active 8 \
  --steps 8 \
  --repeats 1 \
  --warmup-steps 1 \
  --hole-fraction 0.0 \
  --seed 20260623 \
  --precompute-metadata \
  --adaptive-engine \
  --adaptive-workqueue-splits 20 \
  --hkv 8
```

The profiler replays instrumented kernels, so benchmark wall timings inside the
profiled runs are not used as performance claims. Use the normal unprofiled
ablation logs for speed ratios. The table below reports hardware counters from
filtered Nsight Compute captures.

| Kernel | Duration ms | SM throughput % | Memory throughput % | DRAM throughput % | DRAM GB/s | L1/TEX hit % | L2 hit % |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlashInfer decode | 2.271 | 9.55 | 62.72 | 62.72 | 219.2 | 1.52 | 0.12 |
| PersistentKV decode | 1.263 | 17.21 | 74.48 | 57.38 | 200.8 | 1.56 | 0.44 |
| PersistentKV merge | 0.014 | 32.72 | 62.02 | 62.02 | 223.5 | 39.68 | 8.98 |

Artifacts:

- `b8_g4_flashinfer_decode_ncu.ncu-rep`
- `b8_g4_pkv_decode_ncu.ncu-rep`
- `b8_g4_pkv_merge_ncu.ncu-rep`
- corresponding `*_details.csv` exports
- profiled-run text logs under `results/nsight/`
