#!/usr/bin/env python3
"""Benchmark PyTorch SDPA GQA on the PersistentKV decode shapes."""

import argparse
import statistics

import torch
import torch.nn.functional as F


def benchmark(
    batch: int,
    hq: int,
    hkv: int,
    context: int,
    dim: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    q = torch.randn(
        batch, hq, 1, dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        batch, hkv, context, dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn_like(k)

    # For decode, the single query is the newest token and can see the entire
    # cache. PyTorch's non-square causal mask is upper-left aligned, so
    # is_causal=True would incorrectly expose only the first key.
    def run() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=True
        )

    with torch.inference_mode():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        samples = []
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(iterations):
            start.record()
            output = run()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))

    if output.shape != (batch, hq, 1, dim):
        raise RuntimeError(f"unexpected output shape: {output.shape}")

    median_ms = statistics.median(samples)
    mean_ms = statistics.fmean(samples)
    del q, k, v, output
    torch.cuda.empty_cache()
    return median_ms, mean_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch process")

    props = torch.cuda.get_device_properties(0)
    print("PyTorch SDPA GQA decode benchmark")
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={props.name} vram={props.total_memory / 2**30:.1f} GiB"
    )
    print("shape: B=1 Hq=32 Hkv=8 G=4 d=128 q_len=1")
    print("is_causal=False (the decode query sees the full KV cache)")
    print()
    print("| N | median ms | mean ms |")
    print("|---:|----------:|--------:|")

    for context in args.contexts:
        median_ms, mean_ms = benchmark(
            batch=1,
            hq=32,
            hkv=8,
            context=context,
            dim=128,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"| {context} | {median_ms:.3f} | {mean_ms:.3f} |")


if __name__ == "__main__":
    main()
