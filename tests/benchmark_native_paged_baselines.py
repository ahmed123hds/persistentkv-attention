#!/usr/bin/env python3
"""Benchmark native paged-attention kernels on one shared block-table layout."""

import argparse
import gc
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
VLLM_LIBRARY = Path(
    os.environ.get("VLLM_LIBRARY", ROOT / "build/baselines/vllm_0_6_4/_C.abi3.so")
)
FLASHINFER_SOURCE = Path(
    os.environ.get("FLASHINFER_SOURCE", ROOT / "build/baselines/flashinfer_0_2_5_src")
)
TRTLLM_SOURCE = Path(
    os.environ.get("TRTLLM_SOURCE", ROOT / "build/baselines/trtllm_0_8_src")
)
TRTLLM_EXTENSION = ROOT / "tests/trtllm_mmha_extension.cu"

BATCH = 1
HQ = 32
HKV = 8
DIM = 128
PAGE_SIZE = 16
MAX_ERROR = 2e-3
MEAN_ERROR = 3e-4


def canonical_block_table(context: int) -> torch.Tensor:
    logical_pages = context // PAGE_SIZE
    physical_pages = logical_pages * 2
    logical = torch.arange(logical_pages, device="cuda", dtype=torch.int64)
    return ((405 * logical + 17) % physical_pages).to(torch.int32)


def make_inputs(
    context: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if context % PAGE_SIZE:
        raise ValueError("canonical contexts must be page aligned")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260615 + context)
    q = torch.randn(
        BATCH, HQ, DIM,
        generator=generator, device="cuda", dtype=torch.float16,
    )
    k = torch.randn(
        context, HKV, DIM,
        generator=generator, device="cuda", dtype=torch.float16,
    )
    v = torch.randn_like(k)
    table = canonical_block_table(context)
    return q, k, v, table


def pack_pages(
    logical: torch.Tensor,
    table: torch.Tensor,
) -> torch.Tensor:
    logical_pages = table.numel()
    physical = torch.zeros(
        logical_pages * 2,
        PAGE_SIZE,
        HKV,
        DIM,
        device="cuda",
        dtype=logical.dtype,
    )
    physical[table.long()] = logical.view(logical_pages, PAGE_SIZE, HKV, DIM)
    return physical


def reference_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.unsqueeze(2),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        is_causal=False,
        enable_gqa=True,
    ).squeeze(2)


def error(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    difference = (output.float() - reference.float()).abs()
    return difference.max().item(), difference.mean().item()


def time_cuda(
    function: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    repeats: int = 5,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            output = function()
        torch.cuda.synchronize()

        samples = []
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            start.record()
            for _ in range(iterations):
                output = function()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) / iterations)
    if output.numel() != BATCH * HQ * DIM:
        raise RuntimeError(f"unexpected output shape {tuple(output.shape)}")
    return statistics.median(samples)


def iterations_for(context: int) -> int:
    if context <= 8192:
        return 300
    if context <= 32768:
        return 150
    return 75


def benchmark_vllm(
    context: int,
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    table: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float, float]:
    if not VLLM_LIBRARY.exists():
        raise RuntimeError(f"missing vLLM extension: {VLLM_LIBRARY}")
    if not hasattr(torch.ops._C, "paged_attention_v2"):
        torch.ops.load_library(str(VLLM_LIBRARY))

    x = 16 // k_pages.element_size()
    physical_pages = k_pages.shape[0]
    key_cache = (
        k_pages.permute(0, 2, 3, 1)
        .contiguous()
        .view(physical_pages, HKV, DIM // x, x, PAGE_SIZE)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    value_cache = v_pages.permute(0, 2, 3, 1).contiguous()
    block_tables = table.view(1, -1).contiguous()
    seq_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
    partitions = math.ceil(context / 512)
    output = torch.empty_like(q)
    exp_sums = torch.empty(
        BATCH, HQ, partitions, device="cuda", dtype=torch.float32
    )
    max_logits = torch.empty_like(exp_sums)
    tmp_output = torch.empty(
        BATCH, HQ, partitions, DIM, device="cuda", dtype=q.dtype
    )

    def run() -> torch.Tensor:
        torch.ops._C.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            q,
            key_cache,
            value_cache,
            HKV,
            1.0 / math.sqrt(DIM),
            block_tables,
            seq_lens,
            PAGE_SIZE,
            context,
            None,
            "auto",
            1.0,
            1.0,
            0,
            0,
            0,
            64,
            0,
        )
        return output

    with torch.inference_mode():
        result = run()
        torch.cuda.synchronize()
    max_err, mean_err = error(result, reference)
    latency = time_cuda(run, 20, iterations_for(context))
    return latency, max_err, mean_err


def import_flashinfer():
    source = str(FLASHINFER_SOURCE)
    if FLASHINFER_SOURCE.exists() and source not in sys.path:
        sys.path.insert(0, source)
    try:
        import flashinfer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "FlashInfer is required. Install flashinfer-python or set "
            "FLASHINFER_SOURCE to a checkout/source directory containing the "
            "flashinfer Python package."
        ) from error

    return flashinfer


def benchmark_flashinfer(
    context: int,
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    table: torch.Tensor,
    reference: torch.Tensor,
    use_tensor_cores: bool,
) -> tuple[float, float, float]:
    flashinfer = import_flashinfer()
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=use_tensor_cores
    )
    logical_pages = table.numel()
    indptr = torch.tensor(
        [0, logical_pages], device="cuda", dtype=torch.int32
    )
    last_page_len = torch.tensor(
        [PAGE_SIZE], device="cuda", dtype=torch.int32
    )
    wrapper.plan(
        indptr,
        table,
        last_page_len,
        HQ,
        HKV,
        DIM,
        PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    output = torch.empty_like(q)

    def run() -> torch.Tensor:
        return wrapper.run(q, (k_pages, v_pages), out=output)

    with torch.inference_mode():
        result = run()
        torch.cuda.synchronize()
    max_err, mean_err = error(result, reference)
    latency = time_cuda(run, 20, iterations_for(context))
    return latency, max_err, mean_err


def benchmark_repack_sdpa(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    table: torch.Tensor,
) -> float:
    context = table.numel() * PAGE_SIZE

    def run() -> torch.Tensor:
        k = k_pages.index_select(0, table.long()).view(context, HKV, DIM)
        v = v_pages.index_select(0, table.long()).view(context, HKV, DIM)
        return F.scaled_dot_product_attention(
            q.unsqueeze(2),
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
            is_causal=False,
            enable_gqa=True,
        ).squeeze(2)

    return time_cuda(run, 10, max(20, iterations_for(context) // 3), repeats=3)


def import_trtllm_mmha():
    if not TRTLLM_SOURCE.exists():
        raise RuntimeError(f"missing TensorRT-LLM source: {TRTLLM_SOURCE}")
    from torch.utils.cpp_extension import load

    common = TRTLLM_SOURCE / "cpp/tensorrt_llm/common"
    return load(
        name="persistentkv_trtllm_mmha_v080",
        sources=[
            str(TRTLLM_EXTENSION),
            str(common / "envUtils.cpp"),
            str(common / "logger.cpp"),
            str(common / "stringUtils.cpp"),
            str(common / "tllmException.cpp"),
        ],
        extra_include_paths=[
            str(TRTLLM_SOURCE / "cpp"),
            str(TRTLLM_SOURCE / "cpp/include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-gencode=arch=compute_86,code=sm_86",
        ],
        verbose=False,
    )


def benchmark_trtllm(
    context: int,
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    table: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float, float]:
    module = import_trtllm_mmha()
    trt_k_pages = k_pages.permute(0, 2, 1, 3).contiguous()
    trt_v_pages = v_pages.permute(0, 2, 1, 3).contiguous()
    page_bytes = PAGE_SIZE * HKV * DIM * k_pages.element_size()
    logical = table.to(torch.int64)
    k_ptrs = logical * page_bytes + trt_k_pages.data_ptr()
    v_ptrs = logical * page_bytes + trt_v_pages.data_ptr()
    block_ptrs = torch.cat((k_ptrs, v_ptrs)).contiguous()
    seq_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
    last_page = table[-1].long()
    k_last = trt_k_pages[last_page, :, PAGE_SIZE - 1, :].contiguous()
    v_last = trt_v_pages[last_page, :, PAGE_SIZE - 1, :].contiguous()
    output = torch.empty_like(q)
    partial_out = torch.empty(
        8, BATCH, HQ, DIM, device="cuda", dtype=torch.float16
    )
    partial_sum = torch.empty(
        8, BATCH, HQ, device="cuda", dtype=torch.float32
    )
    partial_max = torch.empty_like(partial_sum)
    block_counter = torch.empty(BATCH, HQ, device="cuda", dtype=torch.int32)

    def run() -> torch.Tensor:
        return module.paged_mmha(
            q,
            k_last,
            v_last,
            block_ptrs,
            seq_lens,
            output,
            partial_out,
            partial_sum,
            partial_max,
            block_counter,
            context,
            PAGE_SIZE,
        )

    with torch.inference_mode():
        result = run()
        torch.cuda.synchronize()
    max_err, mean_err = error(result, reference)
    latency = time_cuda(run, 20, iterations_for(context))
    return latency, max_err, mean_err


def check_tolerance(max_err: float, mean_err: float) -> str:
    return (
        "PASS"
        if max_err < MAX_ERROR and mean_err < MEAN_ERROR
        else "FAIL"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["vllm", "flashinfer", "trtllm", "repack"],
        default=["vllm", "flashinfer", "trtllm", "repack"],
    )
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[8192, 32768, 65536]
    )
    parser.add_argument(
        "--page-size", type=int, choices=[16, 32, 64, 128], default=16
    )
    args = parser.parse_args()
    global PAGE_SIZE
    PAGE_SIZE = args.page_size

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    print("Native paged-attention baselines")
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)}"
    )
    print(
        f"B=1 Hq=32 Hkv=8 G=4 d=128 FP16 page={PAGE_SIZE}, 50% holes"
    )
    print("table[p]=(405*p+17) mod physical_pages")
    print("tolerance: max_abs<2e-3 and mean_abs<3e-4")
    print()
    print("| Method | N | Time (ms) | Max error | Mean error | Correct |")
    print("|---|---:|---:|---:|---:|---|")

    failures = 0
    for context in args.contexts:
        q, k, v, table = make_inputs(context)
        with torch.inference_mode():
            reference = reference_output(q, k, v)
            k_pages = pack_pages(k, table)
            v_pages = pack_pages(v, table)
        del k, v
        torch.cuda.empty_cache()

        if "vllm" in args.methods:
            latency, max_err, mean_err = benchmark_vllm(
                context, q, k_pages, v_pages, table, reference
            )
            status = check_tolerance(max_err, mean_err)
            failures += status == "FAIL"
            print(
                f"| vLLM 0.6.4.post1 | {context} | {latency:.4f} | "
                f"{max_err:.3e} | {mean_err:.3e} | {status} |"
            )

        if "flashinfer" in args.methods:
            candidates = []
            for tensor_cores in (False, True):
                latency, max_err, mean_err = benchmark_flashinfer(
                    context,
                    q,
                    k_pages,
                    v_pages,
                    table,
                    reference,
                    tensor_cores,
                )
                candidates.append(
                    (latency, max_err, mean_err, tensor_cores)
                )
            latency, max_err, mean_err, tensor_cores = min(candidates)
            status = check_tolerance(max_err, mean_err)
            failures += status == "FAIL"
            backend = "tensor-core" if tensor_cores else "cuda-core"
            print(
                f"| FlashInfer 0.2.5 ({backend}) | {context} | "
                f"{latency:.4f} | {max_err:.3e} | {mean_err:.3e} | "
                f"{status} |"
            )

        if "trtllm" in args.methods:
            latency, max_err, mean_err = benchmark_trtllm(
                context, q, k_pages, v_pages, table, reference
            )
            status = check_tolerance(max_err, mean_err)
            failures += status == "FAIL"
            print(
                f"| TensorRT-LLM 0.8 MMHA | {context} | {latency:.4f} | "
                f"{max_err:.3e} | {mean_err:.3e} | {status} |"
            )

        if "repack" in args.methods:
            latency = benchmark_repack_sdpa(q, k_pages, v_pages, table)
            print(
                f"| Repack + PyTorch SDPA | {context} | {latency:.4f} | "
                "- | - | reference path |"
            )

        del q, k_pages, v_pages, table, reference
        gc.collect()
        torch.cuda.empty_cache()

    if failures:
        raise SystemExit(f"{failures} native baseline correctness checks failed")


if __name__ == "__main__":
    main()
