# Review Ablation Summary

Environment: RTX 3060 12 GB, CUDA 12.1, PyTorch 2.5.1+cu121, FlashInfer 0.2.5
source checkout. All rows passed the FlashInfer-equivalence correctness
tolerance.

## Five Held-Out Seeds

Seeds: `20260623`, `20260624`, `20260625`, `20260626`, `20260627`.

| Trace / mode | CUDA tok/s ratio mean±std | Wall tok/s ratio mean±std | Worst max error | Worst mean error |
|---|---:|---:|---:|---:|
| Bucketed B1 | 1.471±0.037 | 1.403±0.065 | 6.104e-05 | 4.477e-06 |
| Bimodal B8 | 1.106±0.042 | 1.080±0.050 | 6.104e-05 | 4.201e-06 |
| Uniform B8 | 1.076±0.029 | 1.044±0.022 | 6.104e-05 | 3.053e-06 |
| Zipf B8 | 1.105±0.026 | 1.068±0.028 | 3.052e-05 | 2.671e-06 |

## Split Sensitivity

Seed: `20260623`.

| Trace | Split count | CUDA tok/s ratio | Wall tok/s ratio |
|---|---:|---:|---:|
| Bucketed B1 | 28 | 1.606 | 1.379 |
| Bucketed B1 | 32 | 1.477 | 1.516 |
| Bucketed B1 | 36 | 1.375 | 1.282 |
| Bimodal B8 | 16 | 1.182 | 1.128 |
| Bimodal B8 | 20 | 1.181 | 1.136 |
| Bimodal B8 | 24 | 1.129 | 1.168 |

## GQA/MQA Serving Sweep

The full 24-request G=1 trace exceeded 12 GB VRAM, so this sweep uses a smaller
B8 bimodal trace with 8 requests and 24 measured steps. Seed: `20260623`.

| G | Hkv | Selected route | CUDA tok/s ratio | Wall tok/s ratio | Worst error |
|---:|---:|---|---:|---:|---:|
| 1 | 32 | FlashInfer gate | 1.000 | 1.000 | 0.000e+00 |
| 4 | 8 | PersistentKV workqueue | 1.257 | 1.256 | 6.104e-05 |
| 8 | 4 | FlashInfer gate | 1.000 | 1.000 | 0.000e+00 |

This confirms that the current serving win is specific to the tested G=4
regime. The cost-model gate now routes G=1 and G=8 to FlashInfer rather than
executing the losing PersistentKV workqueue path.
