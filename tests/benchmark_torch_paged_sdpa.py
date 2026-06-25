#!/usr/bin/env python3
"""Benchmark the framework cost of repacking paged KV before PyTorch SDPA.

This is not a production paged-attention baseline. PyTorch SDPA does not accept
a block table, so the benchmark explicitly measures materialization plus SDPA.
"""

import argparse
import math
import random
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F


def time_cuda(
    function: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            result = function()
        torch.cuda.synchronize()

        samples = []
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(iterations):
            start.record()
            result = function()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))

    if isinstance(result, tuple):
        if any(tensor.numel() == 0 for tensor in result):
            raise RuntimeError("unexpected empty repack result")
    elif result.numel() == 0:
        raise RuntimeError("unexpected empty attention result")
    return statistics.median(samples), statistics.fmean(samples)


def make_block_table(
    logical_pages: int,
    physical_pages: int,
    random_order: bool,
) -> torch.Tensor:
    if random_order:
        pages = list(range(physical_pages))
        random.Random(1234).shuffle(pages)
        pages = pages[:logical_pages]
    elif logical_pages == 1:
        pages = [0]
    else:
        pages = [
            page * (physical_pages - 1) // (logical_pages - 1)
            for page in range(logical_pages)
        ]
    return torch.tensor(pages, device="cuda", dtype=torch.int64)


def benchmark_layout(
    context: int,
    page_size: int,
    fragmentation: float,
    random_order: bool,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    hq, hkv, dim = 32, 8, 128
    logical_pages = math.ceil(context / page_size)
    physical_pages = math.ceil(logical_pages / (1.0 - fragmentation))
    q = torch.randn(1, hq, 1, dim, device="cuda", dtype=torch.float16)
    k_pages = torch.randn(
        physical_pages,
        hkv,
        page_size,
        dim,
        device="cuda",
        dtype=torch.float16,
    )
    v_pages = torch.randn_like(k_pages)
    block_table = make_block_table(
        logical_pages, physical_pages, random_order
    )

    def repack() -> tuple[torch.Tensor, torch.Tensor]:
        k = (
            torch.index_select(k_pages, 0, block_table)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(1, hkv, logical_pages * page_size, dim)
        )
        v = (
            torch.index_select(v_pages, 0, block_table)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(1, hkv, logical_pages * page_size, dim)
        )
        return k[:, :, :context], v[:, :, :context]

    def repack_and_sdpa() -> torch.Tensor:
        k, v = repack()
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=True
        )

    repack_median, _ = time_cuda(repack, warmup, iterations)
    full_median, _ = time_cuda(repack_and_sdpa, warmup, iterations)
    del q, k_pages, v_pages, block_table
    torch.cuda.empty_cache()
    return repack_median, full_median


def benchmark_contiguous(
    context: int,
    warmup: int,
    iterations: int,
) -> float:
    q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, context, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    def run() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=True
        )

    median, _ = time_cuda(run, warmup, iterations)
    del q, k, v
    torch.cuda.empty_cache()
    return median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[8192, 32768]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch process")

    layouts = [
        (16, 0.0, False),
        (16, 0.5, True),
        (32, 0.0, False),
        (32, 0.5, True),
        (64, 0.5, True),
        (128, 0.5, True),
    ]
    print("PyTorch paged-KV materialization + SDPA")
    print("This is a framework composition baseline, not a native paged kernel.")
    print("Shape: B=1 Hq=32 Hkv=8 G=4 d=128 q_len=1")
    print()
    print("| N | Page | Frag. | Order | SDPA contiguous | Repack | "
          "Repack + SDPA |")
    print("|---:|---:|---:|---|---:|---:|---:|")
    for context in args.contexts:
        contiguous_ms = benchmark_contiguous(
            context, args.warmup, args.iterations
        )
        for page_size, fragmentation, random_order in layouts:
            repack_ms, full_ms = benchmark_layout(
                context,
                page_size,
                fragmentation,
                random_order,
                args.warmup,
                args.iterations,
            )
            print(
                f"| {context} | {page_size} | {fragmentation * 100:.0f}% | "
                f"{'random' if random_order else 'ordered'} | "
                f"{contiguous_ms:.3f} ms | {repack_ms:.3f} ms | "
                f"{full_ms:.3f} ms |"
            )


if __name__ == "__main__":
    main()
