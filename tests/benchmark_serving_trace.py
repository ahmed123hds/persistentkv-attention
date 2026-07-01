#!/usr/bin/env python3
"""Continuous-batching serving trace experiment for native paged decode.

This intentionally measures a dynamic serving loop, not only a single clean
attention call. FlashInfer runs one ragged paged batch per decode step.
PersistentKV groups active requests by current context length, then runs its
native block-table engine once per length bucket.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FLASHINFER_SOURCE = Path(
    os.environ.get("FLASHINFER_SOURCE", ROOT / "build/baselines/flashinfer_0_2_5_src")
)
PKV_EXTENSION = ROOT / "tests/persistentkv_torch_extension.cu"

HQ = 32
HKV = 8
DIM = 128
PAGE_SIZE = 16
MAX_ERROR = 2e-3
MEAN_ERROR = 3e-4
SM_COUNT = 28
ROOFLINE_READ_GBPS = 302.0
LAUNCH_OVERHEAD_US = 8.0
MIN_CTAS_PER_SM = 4.0
SUPPORTED_PERSISTENTKV_GQA = 4


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival: int
    prompt_len: int
    decode_len: int
    physical_pages: tuple[int, ...]


@dataclass(frozen=True)
class StepState:
    step: int
    active_ids: tuple[int, ...]
    lengths: tuple[int, ...]


@dataclass(frozen=True)
class FlashInferStepMetadata:
    indptr: torch.Tensor
    indices: torch.Tensor
    last_lens: torch.Tensor
    active_long: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class PersistentKVBucketMetadata:
    bucket_length: int
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    offsets_long: torch.Tensor
    q_indices: torch.Tensor
    out_indices: torch.Tensor
    splits: int


@dataclass(frozen=True)
class PersistentKVStepMetadata:
    buckets: tuple[PersistentKVBucketMetadata, ...]
    output: torch.Tensor
    work_items: torch.Tensor | None = None
    split_offsets: torch.Tensor | None = None
    merge_counters: torch.Tensor | None = None
    total_partial_slots: int = 0


@dataclass(frozen=True)
class RouteProfile:
    launches_per_step: float
    nonempty_cta_fraction: float
    cta_work_per_sm: float
    useful_kv_read_mb: float
    merge_state_mb: float
    merge_launches_per_step: float


@dataclass(frozen=True)
class RouteEstimate:
    route: str
    splits: int
    estimated_ms: float
    reason: str


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


def import_persistentkv():
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(ROOT / "build/baselines/torch_extensions"),
    )
    from torch.utils.cpp_extension import load

    return load(
        name="persistentkv_torch_ext",
        sources=[str(PKV_EXTENSION)],
        extra_include_paths=[str(ROOT)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_86,code=sm_86",
        ],
        verbose=False,
    )


def rounded_length(value: int) -> int:
    return max(PAGE_SIZE, ((value + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE)


def make_lengths(trace: str, requests: int, seed: int) -> list[tuple[int, int]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    result: list[tuple[int, int]] = []
    for index in range(requests):
        if trace == "uniform":
            prompt = int(torch.randint(4096, 32769, (1,), generator=gen).item())
            decode = int(torch.randint(16, 97, (1,), generator=gen).item())
        elif trace == "bimodal":
            long_request = (index % 4) == 0
            if long_request:
                prompt = int(torch.randint(24576, 65537, (1,), generator=gen).item())
                decode = int(torch.randint(32, 129, (1,), generator=gen).item())
            else:
                prompt = int(torch.randint(2048, 12289, (1,), generator=gen).item())
                decode = int(torch.randint(8, 65, (1,), generator=gen).item())
        elif trace == "zipf":
            rank = int(torch.randint(1, 33, (1,), generator=gen).item())
            prompt = int(2048 + 65536 / math.sqrt(rank))
            jitter = int(torch.randint(-512, 513, (1,), generator=gen).item())
            prompt = max(1024, prompt + jitter)
            decode = int(torch.randint(8, 129, (1,), generator=gen).item())
        elif trace == "bucketed":
            buckets = [8192, 16384, 32768, 65536]
            prompt = buckets[index % len(buckets)]
            decode = int(torch.randint(32, 97, (1,), generator=gen).item())
        elif trace == "homogeneous":
            prompt = 32768
            decode = int(torch.randint(32, 97, (1,), generator=gen).item())
        else:
            raise ValueError(f"unknown trace: {trace}")
        result.append((rounded_length(prompt), decode))
    return result


def build_requests(
    trace: str,
    requests: int,
    max_active: int,
    seed: int,
    hole_fraction: float,
    bucket_tokens: int,
) -> tuple[list[Request], int]:
    lengths = make_lengths(trace, requests, seed)
    logical_pages = []
    for prompt, decode in lengths:
        exact_length = prompt + decode
        if bucket_tokens > 0:
            reserved_length = rounded_length(
                ((exact_length + bucket_tokens - 1) // bucket_tokens)
                * bucket_tokens
            )
        else:
            reserved_length = rounded_length(exact_length)
        logical_pages.append(math.ceil(reserved_length / PAGE_SIZE))
    total_logical_pages = sum(logical_pages)
    physical_pages = math.ceil(total_logical_pages / max(1e-6, 1.0 - hole_fraction))
    physical_ids = torch.randperm(physical_pages, generator=torch.Generator().manual_seed(seed + 99))

    built: list[Request] = []
    cursor = 0
    for request_id, ((prompt, decode), pages) in enumerate(zip(lengths, logical_pages)):
        arrival = request_id // max_active
        assigned = tuple(int(x) for x in physical_ids[cursor: cursor + pages].tolist())
        cursor += pages
        built.append(Request(request_id, arrival, prompt, decode, assigned))
    return built, physical_pages


def build_steps(requests: list[Request], max_active: int) -> list[StepState]:
    steps: list[StepState] = []
    max_step = max(req.arrival + req.decode_len for req in requests)
    for step in range(max_step):
        candidates: list[tuple[int, int]] = []
        for req in requests:
            generated = step - req.arrival
            if 0 <= generated < req.decode_len:
                candidates.append((req.request_id, req.prompt_len + generated))
        candidates = candidates[:max_active]
        if candidates:
            active_ids, lengths = zip(*candidates)
            steps.append(StepState(step, tuple(active_ids), tuple(lengths)))
    return steps


def make_q_pool(steps: list[StepState], requests: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 2026)
    return torch.randn(
        len(steps), requests, HQ, DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )


def workspace_floats(batch: int, splits: int) -> int:
    return batch * HQ * splits * (2 + DIM)


def round_bucket_length(length: int, bucket_tokens: int, allocated_pages: int) -> int:
    if bucket_tokens <= 0:
        return length
    bucket = ((length + bucket_tokens - 1) // bucket_tokens) * bucket_tokens
    bucket = rounded_length(bucket)
    return min(bucket, allocated_pages * PAGE_SIZE)


def round_up4(value: int) -> int:
    return ((value + 3) // 4) * 4


def choose_splits(
    lengths: list[int],
    requested_splits: int,
    auto_splits: bool,
    single_bucket: bool,
) -> int:
    if not auto_splits:
        return requested_splits
    batch = len(lengths)
    tile_counts = [(length + 31) // 32 for length in lengths]
    t_max = max(tile_counts)
    t_min = min(tile_counts)
    if single_bucket:
        if batch >= 8:
            return max(4, min(40, t_min))
        if batch == 4:
            return max(4, min(16, t_min))
        if batch == 2:
            return max(4, min(24, t_min))
        if t_max >= 1024:
            return max(4, min(36, t_min))
        return max(4, min(24, t_min))

    if batch >= 8:
        return max(4, min(24, t_min))
    if batch == 4:
        return max(4, min(28, t_min))
    if batch == 2 and t_max >= 1024:
        return max(4, min(32, t_min))

    occupancy = math.ceil(320 / (HKV * batch))
    length_based = math.ceil(t_max / 32)
    splits = round_up4(max(occupancy, length_based))
    if t_max <= 384:
        splits = min(splits, 24)
    if batch >= 8:
        splits = min(splits, 40)
    if batch <= 2 and t_max >= 1024:
        splits = max(splits, 32)
    return max(4, min(splits, 48, t_min))


def choose_workqueue_row_splits(
    length: int,
    active_count: int,
    requested_splits: int,
    auto_splits: bool,
) -> int:
    if not auto_splits:
        return requested_splits

    tiles = max(1, (length + 31) // 32)
    occupancy_floor = round_up4(math.ceil(256 / (HKV * max(1, active_count))))
    length_target = round_up4(math.ceil(tiles / 64))
    splits = max(occupancy_floor, length_target)

    if active_count >= 8:
        splits = min(splits, 24)
    elif active_count >= 4:
        splits = min(splits, 32)
    elif active_count == 2:
        splits = min(splits, 40)
    else:
        splits = min(splits, 48)

    return max(4, min(splits, 48, tiles))


def model_bucket_splits(state: StepState) -> int:
    active = len(state.active_ids)
    max_tiles = max((length + 31) // 32 for length in state.lengths)
    min_tiles = min((length + 31) // 32 for length in state.lengths)
    occupancy = math.ceil(256 / (HKV * max(1, active)))
    length_based = math.ceil(max_tiles / 96)
    return max(4, min(round_up4(max(occupancy, length_based)), 48, min_tiles))


def model_workqueue_splits(state: StepState) -> int:
    row_splits = [
        choose_workqueue_row_splits(
            length,
            len(state.active_ids),
            requested_splits=1,
            auto_splits=True,
        )
        for length in state.lengths
    ]
    return max(row_splits)


def estimate_route_ms(state: StepState, route: str, splits: int) -> float:
    kv_mb = useful_kv_mb_for_state(state)
    kv_ms = kv_mb / ROOFLINE_READ_GBPS
    active = len(state.active_ids)
    unique_lengths = len(set(state.lengths))

    if route == "flashinfer":
        ctas_per_sm = active * HKV / SM_COUNT
        launch_count = 1
        merge_mb = 0.0
    elif route == "persistentkv_bucket":
        ctas_per_sm = active * HKV * splits / SM_COUNT
        launch_count = 2 * unique_lengths
        merge_mb = active * HQ * splits * (2 + DIM) * 4 / (1024.0 * 1024.0)
    elif route == "persistentkv_workqueue":
        ctas_per_sm = active * HKV * splits / SM_COUNT
        launch_count = 2
        merge_mb = active * HQ * splits * (2 + DIM) * 4 / (1024.0 * 1024.0)
    else:
        raise ValueError(f"unknown route: {route}")

    occupancy_penalty = max(1.0, MIN_CTAS_PER_SM / max(ctas_per_sm, 1e-6))
    launch_ms = launch_count * LAUNCH_OVERHEAD_US / 1000.0
    merge_ms = merge_mb / ROOFLINE_READ_GBPS
    return kv_ms * occupancy_penalty + merge_ms + launch_ms


def choose_adaptive_route_cost_model(state: StepState) -> RouteEstimate:
    gqa_ratio = HQ // HKV
    if gqa_ratio != SUPPORTED_PERSISTENTKV_GQA:
        return RouteEstimate(
            route="flashinfer",
            splits=1,
            estimated_ms=estimate_route_ms(state, "flashinfer", 1),
            reason=f"G={gqa_ratio} is outside the calibrated PersistentKV GQA route",
        )

    active = len(state.active_ids)
    max_len = max(state.lengths)
    if max_len < 16384:
        return RouteEstimate(
            route="flashinfer",
            splits=1,
            estimated_ms=estimate_route_ms(state, "flashinfer", 1),
            reason="short-context decode is launch/merge-overhead dominated",
        )

    flash_ms = estimate_route_ms(state, "flashinfer", 1)
    candidates = [RouteEstimate("flashinfer", 1, flash_ms, "baseline")]
    if active <= 1:
        splits = model_bucket_splits(state)
        candidates.append(
            RouteEstimate(
                "persistentkv_bucket",
                splits,
                estimate_route_ms(state, "persistentkv_bucket", splits),
                "B1 long-context sequence splitting exposes CTAs",
            )
        )
    if active >= 8:
        splits = model_workqueue_splits(state)
        candidates.append(
            RouteEstimate(
                "persistentkv_workqueue",
                splits,
                estimate_route_ms(state, "persistentkv_workqueue", splits),
                "B8 ragged long-context workqueue avoids length-bucket launch fan-out",
            )
        )

    return min(candidates, key=lambda item: item.estimated_ms)


def use_single_bucket_for_state(
    state: StepState,
    requested_single_bucket: bool,
    adaptive_route: bool,
) -> bool:
    if requested_single_bucket:
        return True
    if not adaptive_route:
        return False
    active = len(state.active_ids)
    unique_lengths = len(set(state.lengths))
    if active <= 1:
        return False
    return unique_lengths >= max(3, active - 1)


def choose_adaptive_engine(state: StepState) -> str:
    return choose_adaptive_route_cost_model(state).route


def useful_kv_mb_for_state(state: StepState) -> float:
    bytes_read = sum(state.lengths) * HKV * DIM * 2 * 2
    return bytes_read / (1024.0 * 1024.0)


def flashinfer_profile(steps: list[StepState]) -> RouteProfile:
    return RouteProfile(
        launches_per_step=1.0,
        nonempty_cta_fraction=1.0,
        cta_work_per_sm=0.0,
        useful_kv_read_mb=statistics.mean(useful_kv_mb_for_state(s) for s in steps),
        merge_state_mb=0.0,
        merge_launches_per_step=0.0,
    )


def bucket_profile(metadata: list[PersistentKVStepMetadata], steps: list[StepState]) -> RouteProfile:
    launches = []
    nonempty = []
    cta_work = []
    merge_bytes = []
    merge_launches = []
    for meta in metadata:
        step_launches = 0
        step_possible = 0
        step_nonempty = 0
        step_decode_ctas = 0
        step_merge_bytes = 0
        step_merge_launches = 0
        for bucket in meta.buckets:
            B = int(bucket.seq_lens.numel())
            splits = bucket.splits
            step_launches += 1 if splits == 1 else 2
            step_decode_ctas += B * HKV * splits
            step_possible += B * HKV * splits
            seq_lens = [int(x) for x in bucket.seq_lens.detach().cpu().tolist()]
            for length in seq_lens:
                tiles = (length + 31) // 32
                filled = sum(
                    1
                    for split in range(splits)
                    if (tiles * split) // splits < (tiles * (split + 1)) // splits
                )
                step_nonempty += filled * HKV
            if splits > 1:
                step_merge_launches += 1
                step_merge_bytes += B * HQ * splits * (2 + DIM) * 4
        launches.append(step_launches)
        nonempty.append(step_nonempty / max(1, step_possible))
        cta_work.append(step_decode_ctas / 28.0)
        merge_bytes.append(step_merge_bytes / (1024.0 * 1024.0))
        merge_launches.append(step_merge_launches)
    return RouteProfile(
        launches_per_step=statistics.mean(launches),
        nonempty_cta_fraction=statistics.mean(nonempty),
        cta_work_per_sm=statistics.mean(cta_work),
        useful_kv_read_mb=statistics.mean(useful_kv_mb_for_state(s) for s in steps),
        merge_state_mb=statistics.mean(merge_bytes),
        merge_launches_per_step=statistics.mean(merge_launches),
    )


def workqueue_profile(metadata: list[PersistentKVStepMetadata], steps: list[StepState], fused_merge: bool) -> RouteProfile:
    launches = []
    cta_work = []
    merge_bytes = []
    merge_launches = []
    for meta in metadata:
        work_items = 0 if meta.work_items is None else int(meta.work_items.size(0))
        launches.append(1 if fused_merge else 2)
        cta_work.append(work_items / 28.0)
        merge_bytes.append(meta.total_partial_slots * HQ * (2 + DIM) * 4 / (1024.0 * 1024.0))
        merge_launches.append(0 if fused_merge else 1)
    return RouteProfile(
        launches_per_step=statistics.mean(launches),
        nonempty_cta_fraction=1.0,
        cta_work_per_sm=statistics.mean(cta_work),
        useful_kv_read_mb=statistics.mean(useful_kv_mb_for_state(s) for s in steps),
        merge_state_mb=statistics.mean(merge_bytes),
        merge_launches_per_step=statistics.mean(merge_launches),
    )


def adaptive_profile(
    flash_profile: RouteProfile,
    bucket_prof: RouteProfile,
    workqueue_prof: RouteProfile,
    steps: list[StepState],
) -> RouteProfile:
    values = []
    for field in RouteProfile.__dataclass_fields__:
        samples = []
        for state in steps:
            route = choose_adaptive_engine(state)
            prof = (
                bucket_prof
                if route == "persistentkv_bucket"
                else workqueue_prof
                if route == "persistentkv_workqueue"
                else flash_profile
            )
            samples.append(getattr(prof, field))
        values.append(statistics.mean(samples))
    return RouteProfile(*values)


def step_metadata_flashinfer(
    requests: list[Request],
    state: StepState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indptr = [0]
    indices: list[int] = []
    last_lens: list[int] = []
    by_id = {req.request_id: req for req in requests}
    for request_id, length in zip(state.active_ids, state.lengths):
        req = by_id[request_id]
        page_count = math.ceil(length / PAGE_SIZE)
        indices.extend(req.physical_pages[:page_count])
        indptr.append(len(indices))
        last_lens.append((length - 1) % PAGE_SIZE + 1)
    return (
        torch.tensor(indptr, device="cuda", dtype=torch.int32),
        torch.tensor(indices, device="cuda", dtype=torch.int32),
        torch.tensor(last_lens, device="cuda", dtype=torch.int32),
    )


def compile_flashinfer_metadata(
    requests: list[Request],
    steps: list[StepState],
) -> list[FlashInferStepMetadata]:
    result: list[FlashInferStepMetadata] = []
    for state in steps:
        indptr, indices, last_lens = step_metadata_flashinfer(requests, state)
        active_long = torch.tensor(state.active_ids, device="cuda", dtype=torch.long)
        output = torch.empty(
            len(state.active_ids), HQ, DIM, device="cuda", dtype=torch.float16
        )
        result.append(
            FlashInferStepMetadata(indptr, indices, last_lens, active_long, output)
        )
    return result


def compile_persistentkv_metadata(
    requests: list[Request],
    steps: list[StepState],
    requested_splits: int,
    bucket_tokens: int,
    single_bucket: bool,
    auto_splits: bool,
    adaptive_route: bool,
    workqueue: bool,
) -> list[PersistentKVStepMetadata]:
    by_id = {req.request_id: req for req in requests}
    result: list[PersistentKVStepMetadata] = []
    for state in steps:
        if workqueue:
            bucket_length = max(state.lengths)
            page_count = math.ceil(bucket_length / PAGE_SIZE)
            rows = []
            seq_lens = []
            q_indices = []
            out_indices = []
            work_items: list[list[int]] = []
            split_offsets = [0]
            for offset, length in enumerate(state.lengths):
                req = by_id[state.active_ids[offset]]
                pages = req.physical_pages[:page_count]
                if len(pages) < page_count:
                    pages = pages + (req.physical_pages[-1],) * (
                        page_count - len(pages)
                    )
                rows.append(pages)
                seq_lens.append(length)
                q_indices.append(state.active_ids[offset])
                out_indices.append(offset)

                row_splits = choose_workqueue_row_splits(
                    length,
                    len(state.active_ids),
                    requested_splits,
                    auto_splits,
                )
                total_tiles = (length + 31) // 32
                for split in range(row_splits):
                    tile_begin = (total_tiles * split) // row_splits
                    tile_end = (total_tiles * (split + 1)) // row_splits
                    if tile_begin >= tile_end:
                        continue
                    for kvh in range(HKV):
                        work_items.append([offset, kvh, split, tile_begin, tile_end])
                split_offsets.append(split_offsets[-1] + row_splits)

            bucket = PersistentKVBucketMetadata(
                bucket_length=int(bucket_length),
                block_table=torch.tensor(rows, device="cuda", dtype=torch.int32),
                seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
                offsets_long=torch.tensor(out_indices, device="cuda", dtype=torch.long),
                q_indices=torch.tensor(q_indices, device="cuda", dtype=torch.int32),
                out_indices=torch.tensor(out_indices, device="cuda", dtype=torch.int32),
                splits=max(1, max(split_offsets[i + 1] - split_offsets[i] for i in range(len(state.active_ids)))),
            )
            output = torch.empty(
                len(state.active_ids), HQ, DIM, device="cuda", dtype=torch.float16
            )
            result.append(
                PersistentKVStepMetadata(
                    (bucket,),
                    output,
                    torch.tensor(work_items, device="cuda", dtype=torch.int32),
                    torch.tensor(split_offsets, device="cuda", dtype=torch.int32),
                    torch.zeros(
                        len(state.active_ids), HKV, device="cuda", dtype=torch.int32
                    ),
                    split_offsets[-1],
                )
            )
            continue

        buckets: dict[int, list[int]] = {}
        state_single_bucket = use_single_bucket_for_state(
            state, single_bucket, adaptive_route
        )
        if state_single_bucket:
            buckets[max(state.lengths)] = list(range(len(state.active_ids)))
        else:
            for offset, length in enumerate(state.lengths):
                req = by_id[state.active_ids[offset]]
                bucket_length = round_bucket_length(
                    length, bucket_tokens, len(req.physical_pages)
                )
                buckets.setdefault(bucket_length, []).append(offset)

        compiled_buckets: list[PersistentKVBucketMetadata] = []
        for bucket_length, offsets in buckets.items():
            page_count = math.ceil(bucket_length / PAGE_SIZE)
            rows = []
            seq_lens = []
            for offset in offsets:
                req = by_id[state.active_ids[offset]]
                pages = req.physical_pages[:page_count]
                if len(pages) < page_count:
                    pages = pages + (req.physical_pages[-1],) * (
                        page_count - len(pages)
                    )
                rows.append(pages)
                seq_lens.append(state.lengths[offset])
            bucket_splits = choose_splits(
                seq_lens, requested_splits, auto_splits, state_single_bucket
            )
            compiled_buckets.append(
                PersistentKVBucketMetadata(
                    bucket_length=int(bucket_length),
                    block_table=torch.tensor(rows, device="cuda", dtype=torch.int32),
                    seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
                    offsets_long=torch.tensor(offsets, device="cuda", dtype=torch.long),
                    q_indices=torch.tensor(
                        [state.active_ids[offset] for offset in offsets],
                        device="cuda",
                        dtype=torch.int32,
                    ),
                    out_indices=torch.tensor(offsets, device="cuda", dtype=torch.int32),
                    splits=bucket_splits,
                )
            )
        output = torch.empty(
            len(state.active_ids), HQ, DIM, device="cuda", dtype=torch.float16
        )
        result.append(PersistentKVStepMetadata(tuple(compiled_buckets), output))
    return result


def run_flashinfer_step(
    wrapper,
    q_pool: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    requests: list[Request],
    state_index: int,
    state: StepState,
    use_tensor_cores: bool,
) -> torch.Tensor:
    indptr, indices, last_lens = step_metadata_flashinfer(requests, state)
    wrapper.plan(
        indptr,
        indices,
        last_lens,
        HQ,
        HKV,
        DIM,
        PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    active = torch.tensor(state.active_ids, device="cuda", dtype=torch.long)
    q = q_pool[state_index].index_select(0, active).contiguous()
    output = torch.empty_like(q)
    return wrapper.run(q, (k_pages, v_pages), out=output)


def run_flashinfer_step_compiled(
    wrapper,
    q_pool: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    state_index: int,
    metadata: list[FlashInferStepMetadata],
) -> torch.Tensor:
    meta = metadata[state_index]
    wrapper.plan(
        meta.indptr,
        meta.indices,
        meta.last_lens,
        HQ,
        HKV,
        DIM,
        PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    q = q_pool[state_index].index_select(0, meta.active_long).contiguous()
    return wrapper.run(q, (k_pages, v_pages), out=meta.output)


def run_persistentkv_step(
    module,
    q_pool: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    requests: list[Request],
    state_index: int,
    state: StepState,
    workspace: torch.Tensor,
    splits: int,
    bucket_tokens: int,
    use_indexed: bool,
    single_bucket: bool,
    auto_splits: bool,
    adaptive_route: bool,
) -> torch.Tensor:
    by_id = {req.request_id: req for req in requests}
    buckets: dict[int, list[int]] = {}
    state_single_bucket = use_single_bucket_for_state(
        state, single_bucket, adaptive_route
    )
    if state_single_bucket:
        buckets[max(state.lengths)] = list(range(len(state.active_ids)))
    else:
        for offset, length in enumerate(state.lengths):
            req = by_id[state.active_ids[offset]]
            bucket_length = round_bucket_length(
                length, bucket_tokens, len(req.physical_pages)
            )
            buckets.setdefault(bucket_length, []).append(offset)

    full_output = torch.empty(
        len(state.active_ids), HQ, DIM, device="cuda", dtype=torch.float16
    )
    q_source = q_pool[state_index]
    if not use_indexed:
        active_ids = torch.tensor(state.active_ids, device="cuda", dtype=torch.long)
        q_active = q_source.index_select(0, active_ids).contiguous()

    for bucket_length, offsets in buckets.items():
        page_count = math.ceil(bucket_length / PAGE_SIZE)
        rows = []
        seq_lens = []
        for offset in offsets:
            req = by_id[state.active_ids[offset]]
            pages = req.physical_pages[:page_count]
            if len(pages) < page_count:
                pages = pages + (req.physical_pages[-1],) * (page_count - len(pages))
            rows.append(pages)
            seq_lens.append(state.lengths[offset])
        block_table = torch.tensor(rows, device="cuda", dtype=torch.int32)
        seq_lens_tensor = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
        offset_tensor = torch.tensor(offsets, device="cuda", dtype=torch.long)
        bucket_splits = choose_splits(
            seq_lens, splits, auto_splits, state_single_bucket
        )
        if use_indexed:
            q_indices = torch.tensor(
                [state.active_ids[offset] for offset in offsets],
                device="cuda",
                dtype=torch.int32,
            )
            out_indices = torch.tensor(offsets, device="cuda", dtype=torch.int32)
            module.paged_attention_indexed(
                q_source,
                k_pages,
                v_pages,
                block_table,
                seq_lens_tensor,
                q_indices,
                out_indices,
                full_output,
                workspace,
                int(bucket_length),
                PAGE_SIZE,
                bucket_splits,
            )
        elif bucket_tokens > 0:
            q_bucket = q_active.index_select(0, offset_tensor).contiguous()
            out_bucket = torch.empty_like(q_bucket)
            module.paged_attention_masked(
                q_bucket,
                k_pages,
                v_pages,
                block_table,
                seq_lens_tensor,
                out_bucket,
                workspace,
                int(bucket_length),
                PAGE_SIZE,
                bucket_splits,
            )
        else:
            q_bucket = q_active.index_select(0, offset_tensor).contiguous()
            out_bucket = torch.empty_like(q_bucket)
            module.paged_attention(
                q_bucket,
                k_pages,
                v_pages,
                block_table,
                out_bucket,
                workspace,
                int(bucket_length),
                PAGE_SIZE,
                bucket_splits,
            )
        if not use_indexed:
            full_output.index_copy_(0, offset_tensor, out_bucket)
    return full_output


def run_persistentkv_step_compiled(
    module,
    q_pool: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    state_index: int,
    workspace: torch.Tensor,
    metadata: list[PersistentKVStepMetadata],
    fused_merge: bool,
) -> torch.Tensor:
    meta = metadata[state_index]
    q_source = q_pool[state_index]
    if (
        meta.work_items is not None
        and meta.split_offsets is not None
        and meta.merge_counters is not None
    ):
        bucket = meta.buckets[0]
        module.paged_attention_workqueue(
            q_source,
            k_pages,
            v_pages,
            bucket.block_table,
            bucket.seq_lens,
            meta.work_items,
            meta.split_offsets,
            meta.merge_counters,
            bucket.q_indices,
            bucket.out_indices,
            meta.output,
            workspace,
            bucket.bucket_length,
            PAGE_SIZE,
            meta.total_partial_slots,
            fused_merge,
        )
        return meta.output
    for bucket in meta.buckets:
        module.paged_attention_indexed(
            q_source,
            k_pages,
            v_pages,
            bucket.block_table,
            bucket.seq_lens,
            bucket.q_indices,
            bucket.out_indices,
            meta.output,
            workspace,
            bucket.bucket_length,
            PAGE_SIZE,
            bucket.splits,
        )
    return meta.output


def synchronized_wall_ms(function, repeats: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), samples


def cuda_event_ms(function, repeats: int) -> tuple[float, list[float]]:
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        function()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples), samples


def summarize_step_latencies(samples: list[float]) -> tuple[float, float, float, float]:
    ordered = sorted(samples)
    if not ordered:
        return 0.0, 0.0, 0.0, 0.0
    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, int(math.ceil(p * len(ordered)) - 1))
        return ordered[idx]
    return statistics.mean(ordered), pct(0.50), pct(0.95), pct(0.99)


def benchmark_engine(
    name: str,
    step_fn,
    steps: list[StepState],
    warmup_steps: int,
    repeats: int,
) -> dict[str, float]:
    for index, state in enumerate(steps[:warmup_steps]):
        step_fn(index, state)
    torch.cuda.synchronize()

    event_samples = []
    wall_samples = []
    for _ in range(repeats):
        for index, state in enumerate(steps):
            event_ms, _ = cuda_event_ms(lambda i=index, s=state: step_fn(i, s), 1)
            wall_ms, _ = synchronized_wall_ms(lambda i=index, s=state: step_fn(i, s), 1)
            event_samples.append(event_ms)
            wall_samples.append(wall_ms)
    torch.cuda.synchronize()

    total_tokens = sum(len(state.active_ids) for state in steps) * repeats
    total_event_s = sum(event_samples) / 1000.0
    total_wall_s = sum(wall_samples) / 1000.0
    event_mean, event_p50, event_p95, event_p99 = summarize_step_latencies(event_samples)
    wall_mean, wall_p50, wall_p95, wall_p99 = summarize_step_latencies(wall_samples)
    return {
        "tokens": float(total_tokens),
        "event_total_ms": sum(event_samples),
        "wall_total_ms": sum(wall_samples),
        "event_tokens_s": total_tokens / total_event_s,
        "wall_tokens_s": total_tokens / total_wall_s,
        "event_mean_ms": event_mean,
        "event_p50_ms": event_p50,
        "event_p95_ms": event_p95,
        "event_p99_ms": event_p99,
        "wall_mean_ms": wall_mean,
        "wall_p50_ms": wall_p50,
        "wall_p95_ms": wall_p95,
        "wall_p99_ms": wall_p99,
    }


def correctness_check(flashinfer_step, persistentkv_step, steps: list[StepState]) -> tuple[float, float]:
    index = max(0, len(steps) // 2)
    reference = flashinfer_step(index, steps[index])
    candidate = persistentkv_step(index, steps[index])
    torch.cuda.synchronize()
    diff = (candidate.float() - reference.float()).abs()
    return diff.max().item(), diff.mean().item()


def format_row(name: str, stats: dict[str, float]) -> str:
    return (
        f"| {name} | {stats['event_tokens_s']:.1f} | "
        f"{stats['event_mean_ms']:.3f} | {stats['event_p95_ms']:.3f} | "
        f"{stats['event_p99_ms']:.3f} | {stats['wall_tokens_s']:.1f} | "
        f"{stats['wall_mean_ms']:.3f} | {stats['wall_p95_ms']:.3f} | "
        f"{stats['wall_p99_ms']:.3f} |"
    )


def format_profile_row(name: str, profile: RouteProfile, stats: dict[str, float]) -> str:
    python_ms = max(0.0, stats["wall_mean_ms"] - stats["event_mean_ms"])
    return (
        f"| {name} | {profile.launches_per_step:.2f} | "
        f"{profile.nonempty_cta_fraction:.3f} | "
        f"{profile.cta_work_per_sm:.1f} | "
        f"{profile.useful_kv_read_mb:.1f} | "
        f"{profile.merge_state_mb:.2f} | "
        f"{profile.merge_launches_per_step:.2f} | "
        f"{python_ms:.3f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        choices=["uniform", "bimodal", "zipf", "bucketed", "homogeneous"],
        default="bimodal",
    )
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--hole-fraction", type=float, default=0.50)
    parser.add_argument("--page-size", type=int, choices=[8, 16, 32, 64, 128], default=16)
    parser.add_argument("--splits", type=int, default=16)
    parser.add_argument(
        "--bucket-tokens",
        type=int,
        default=0,
        help="round PersistentKV route lengths up to this many tokens; 0 keeps exact lengths",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--output", type=Path, default=ROOT / "results/serving_trace_experiment.txt")
    parser.add_argument(
        "--hkv",
        type=int,
        default=8,
        choices=[4, 8, 16, 32],
        help="number of KV heads; Hq is fixed at 32, so G=32/hkv",
    )
    parser.add_argument("--flashinfer-cuda-core", action="store_true")
    parser.add_argument(
        "--pkv-indexed",
        action="store_true",
        help="let PersistentKV read Q and write output rows by index, avoiding per-bucket gather/scatter",
    )
    parser.add_argument(
        "--pkv-single-bucket",
        action="store_true",
        help="route all active PersistentKV requests through one ragged masked/indexed launch",
    )
    parser.add_argument(
        "--precompute-metadata",
        action="store_true",
        help="build per-step CUDA metadata and output buffers before timed decode steps",
    )
    parser.add_argument(
        "--auto-splits",
        action="store_true",
        help="choose PersistentKV split count independently for each route bucket",
    )
    parser.add_argument(
        "--pkv-adaptive-route",
        action="store_true",
        help="route ragged steps through one masked bucket and low-ragged steps through exact buckets",
    )
    parser.add_argument(
        "--pkv-workqueue",
        action="store_true",
        help="use compact non-empty PersistentKV work items instead of rectangular split grids",
    )
    parser.add_argument(
        "--pkv-cuda-graph",
        action="store_true",
        help="capture each precompiled PersistentKV decode step and replay it with CUDA Graphs",
    )
    parser.add_argument(
        "--pkv-fused-merge",
        action="store_true",
        help="use an experimental atomic completion counter to fuse workqueue decode and merge",
    )
    parser.add_argument(
        "--adaptive-engine",
        action="store_true",
        help="benchmark an adaptive route selector: FlashInfer for low-work steps, PersistentKV bucket for B1 long-context, PersistentKV workqueue for B8 long-context",
    )
    parser.add_argument(
        "--adaptive-bucket-splits",
        type=int,
        default=32,
        help="PersistentKV split count for the adaptive B1/length-bucket route",
    )
    parser.add_argument(
        "--adaptive-workqueue-splits",
        type=int,
        default=24,
        help="PersistentKV split count for the adaptive workqueue route",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    global PAGE_SIZE, HKV
    PAGE_SIZE = args.page_size
    HKV = args.hkv

    torch.manual_seed(args.seed)
    requests, physical_pages = build_requests(
        args.trace,
        args.requests,
        args.max_active,
        args.seed,
        args.hole_fraction,
        args.bucket_tokens,
    )
    steps = build_steps(requests, args.max_active)[:args.steps]
    if not steps:
        raise SystemExit("trace produced no active decode steps")
    adaptive_routes = [choose_adaptive_engine(state) for state in steps]
    adaptive_uses_persistentkv = (
        not args.adaptive_engine
        or any(route != "flashinfer" for route in adaptive_routes)
    )

    print("Loading kernels...")
    flashinfer = import_flashinfer()
    persistentkv = import_persistentkv() if adaptive_uses_persistentkv else None

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 7)
    if adaptive_uses_persistentkv:
        k_pages_pkv = torch.randn(
            physical_pages, HKV, PAGE_SIZE, DIM,
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        v_pages_pkv = torch.randn_like(k_pages_pkv)
        k_pages_flashinfer = k_pages_pkv.permute(0, 2, 1, 3).contiguous()
        v_pages_flashinfer = v_pages_pkv.permute(0, 2, 1, 3).contiguous()
    else:
        k_pages_flashinfer = torch.randn(
            physical_pages, PAGE_SIZE, HKV, DIM,
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        v_pages_flashinfer = torch.randn_like(k_pages_flashinfer)
        k_pages_pkv = torch.empty(0, device="cuda", dtype=torch.float16)
        v_pages_pkv = torch.empty(0, device="cuda", dtype=torch.float16)
    q_pool = make_q_pool(steps, args.requests, args.seed)
    workspace_splits = max(
        args.splits,
        args.adaptive_bucket_splits if args.adaptive_engine else args.splits,
        args.adaptive_workqueue_splits if args.adaptive_engine else args.splits,
        48 if args.auto_splits else args.splits,
    )
    workspace = torch.empty(
        workspace_floats(args.max_active, workspace_splits),
        device="cuda",
        dtype=torch.float32,
    )
    flashinfer_workspace = torch.empty(
        128 * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    flashinfer_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        flashinfer_workspace,
        "NHD",
        use_tensor_cores=not args.flashinfer_cuda_core,
    )
    flashinfer_metadata = None
    persistentkv_metadata = None
    persistentkv_bucket_metadata = None
    persistentkv_workqueue_metadata = None
    if (
        (args.precompute_metadata or args.pkv_workqueue)
        and adaptive_uses_persistentkv
    ) or (args.adaptive_engine and adaptive_uses_persistentkv):
        flashinfer_metadata = compile_flashinfer_metadata(requests, steps)
        persistentkv_metadata = compile_persistentkv_metadata(
            requests,
            steps,
            args.splits,
            args.bucket_tokens,
            args.pkv_single_bucket,
            args.auto_splits,
            args.pkv_adaptive_route,
            args.pkv_workqueue,
        )
        if args.adaptive_engine:
            persistentkv_bucket_metadata = compile_persistentkv_metadata(
                requests,
                steps,
                args.adaptive_bucket_splits,
                args.bucket_tokens,
                False,
                False,
                False,
                False,
            )
            persistentkv_workqueue_metadata = compile_persistentkv_metadata(
                requests,
                steps,
                args.adaptive_workqueue_splits,
                args.bucket_tokens,
                False,
                False,
                False,
                True,
            )
        torch.cuda.synchronize()
    elif args.precompute_metadata or args.adaptive_engine:
        flashinfer_metadata = compile_flashinfer_metadata(requests, steps)
        torch.cuda.synchronize()

    def flashinfer_step(index: int, state: StepState) -> torch.Tensor:
        if flashinfer_metadata is not None:
            return run_flashinfer_step_compiled(
                flashinfer_wrapper,
                q_pool,
                k_pages_flashinfer,
                v_pages_flashinfer,
                index,
                flashinfer_metadata,
            )
        return run_flashinfer_step(
            flashinfer_wrapper,
            q_pool,
            k_pages_flashinfer,
            v_pages_flashinfer,
            requests,
            index,
            state,
            use_tensor_cores=not args.flashinfer_cuda_core,
        )

    def persistentkv_step_uncaptured(index: int, state: StepState) -> torch.Tensor:
        if persistentkv is None:
            raise RuntimeError("PersistentKV route was not compiled")
        if persistentkv_metadata is not None:
            return run_persistentkv_step_compiled(
                persistentkv,
                q_pool,
                k_pages_pkv,
                v_pages_pkv,
                index,
                workspace,
                persistentkv_metadata,
                args.pkv_fused_merge,
            )
        return run_persistentkv_step(
            persistentkv,
            q_pool,
            k_pages_pkv,
            v_pages_pkv,
            requests,
            index,
            state,
            workspace,
            args.splits,
            args.bucket_tokens,
            args.pkv_indexed,
            args.pkv_single_bucket,
            args.auto_splits,
            args.pkv_adaptive_route,
        )

    def persistentkv_bucket_step(index: int, state: StepState) -> torch.Tensor:
        if persistentkv is None:
            raise RuntimeError("PersistentKV bucket route was not compiled")
        if persistentkv_bucket_metadata is None:
            raise RuntimeError("bucket metadata was not compiled")
        return run_persistentkv_step_compiled(
            persistentkv,
            q_pool,
            k_pages_pkv,
            v_pages_pkv,
            index,
            workspace,
            persistentkv_bucket_metadata,
            False,
        )

    def persistentkv_workqueue_step(index: int, state: StepState) -> torch.Tensor:
        if persistentkv is None:
            raise RuntimeError("PersistentKV workqueue route was not compiled")
        if persistentkv_workqueue_metadata is None:
            raise RuntimeError("workqueue metadata was not compiled")
        return run_persistentkv_step_compiled(
            persistentkv,
            q_pool,
            k_pages_pkv,
            v_pages_pkv,
            index,
            workspace,
            persistentkv_workqueue_metadata,
            args.pkv_fused_merge,
        )

    def adaptive_step(index: int, state: StepState) -> torch.Tensor:
        route = adaptive_routes[index]
        if route == "persistentkv_bucket":
            return persistentkv_bucket_step(index, state)
        if route == "persistentkv_workqueue":
            return persistentkv_workqueue_step(index, state)
        return flashinfer_step(index, state)

    persistentkv_graphs: list[torch.cuda.CUDAGraph] | None = None
    persistentkv_graph_outputs: list[torch.Tensor] | None = None

    def persistentkv_step(index: int, state: StepState) -> torch.Tensor:
        if persistentkv_graphs is not None and persistentkv_graph_outputs is not None:
            persistentkv_graphs[index].replay()
            return persistentkv_graph_outputs[index]
        return persistentkv_step_uncaptured(index, state)

    candidate_step = adaptive_step if args.adaptive_engine else persistentkv_step

    max_err, mean_err = correctness_check(flashinfer_step, candidate_step, steps)
    correctness = "PASS" if max_err < MAX_ERROR and mean_err < MEAN_ERROR else "FAIL"
    if correctness == "FAIL":
        raise SystemExit(
            f"correctness failed: max={max_err:.3e} mean={mean_err:.3e}"
        )

    if args.pkv_cuda_graph:
        if persistentkv_metadata is None:
            raise SystemExit("--pkv-cuda-graph requires precomputed PersistentKV metadata")
        persistentkv_graphs = []
        persistentkv_graph_outputs = []
        torch.cuda.synchronize()
        for index, state in enumerate(steps):
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                persistentkv_step_uncaptured(index, state)
            persistentkv_graphs.append(graph)
            persistentkv_graph_outputs.append(persistentkv_metadata[index].output)
        torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()

    flashinfer_stats = benchmark_engine(
        "FlashInfer", flashinfer_step, steps, args.warmup_steps, args.repeats
    )
    if args.adaptive_engine:
        if all(route == "flashinfer" for route in adaptive_routes):
            bucket_stats = None
            workqueue_stats = None
            persistentkv_stats = flashinfer_stats
        else:
            bucket_stats = benchmark_engine(
                "PersistentKV bucket",
                persistentkv_bucket_step,
                steps,
                args.warmup_steps,
                args.repeats,
            )
            workqueue_stats = benchmark_engine(
                "PersistentKV workqueue",
                persistentkv_workqueue_step,
                steps,
                args.warmup_steps,
                args.repeats,
            )
            if all(route == "persistentkv_bucket" for route in adaptive_routes):
                persistentkv_stats = bucket_stats
            elif all(route == "persistentkv_workqueue" for route in adaptive_routes):
                persistentkv_stats = workqueue_stats
            else:
                persistentkv_stats = benchmark_engine(
                    "Adaptive routed engine",
                    adaptive_step,
                    steps,
                    args.warmup_steps,
                    args.repeats,
                )
    else:
        bucket_stats = None
        workqueue_stats = None
        persistentkv_stats = benchmark_engine(
            "PersistentKV", persistentkv_step, steps, args.warmup_steps, args.repeats
        )

    active_counts = [len(step.active_ids) for step in steps]
    unique_lengths = [len(set(step.lengths)) for step in steps]
    by_id = {req.request_id: req for req in requests}
    route_bucket_counts = [
        1
        if args.pkv_workqueue or use_single_bucket_for_state(
            step, args.pkv_single_bucket, args.pkv_adaptive_route
        )
        else len({
            round_bucket_length(
                length,
                args.bucket_tokens,
                len(by_id[request_id].physical_pages),
            )
            for request_id, length in zip(step.active_ids, step.lengths)
        })
        for step in steps
    ]
    total_decode_tokens = sum(active_counts) * args.repeats
    flash_profile = flashinfer_profile(steps)
    if args.adaptive_engine:
        if adaptive_uses_persistentkv:
            assert persistentkv_bucket_metadata is not None
            assert persistentkv_workqueue_metadata is not None
            bucket_prof = bucket_profile(persistentkv_bucket_metadata, steps)
            workqueue_prof = workqueue_profile(
                persistentkv_workqueue_metadata, steps, args.pkv_fused_merge
            )
            adaptive_prof = adaptive_profile(
                flash_profile, bucket_prof, workqueue_prof, steps
            )
        else:
            bucket_prof = None
            workqueue_prof = None
            adaptive_prof = flash_profile
        route_counts = {
            "flashinfer": sum(1 for route in adaptive_routes if route == "flashinfer"),
            "persistentkv_bucket": sum(1 for route in adaptive_routes if route == "persistentkv_bucket"),
            "persistentkv_workqueue": sum(1 for route in adaptive_routes if route == "persistentkv_workqueue"),
        }
        route_estimates = [choose_adaptive_route_cost_model(state) for state in steps]
        route_split_summary = {
            route: sorted({
                estimate.splits
                for estimate in route_estimates
                if estimate.route == route
            })
            for route in route_counts
        }
    else:
        route_counts = None
        route_split_summary = None
        bucket_prof = None
        workqueue_prof = None
        adaptive_prof = None

    lines = [
        "Serving trace experiment",
        f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}",
        f"trace={args.trace} requests={args.requests} max_active={args.max_active} measured_steps={len(steps)} repeats={args.repeats}",
        f"B active mean={statistics.mean(active_counts):.2f} max={max(active_counts)} exact_length_buckets mean={statistics.mean(unique_lengths):.2f} max={max(unique_lengths)}",
        f"PersistentKV bucket_tokens={args.bucket_tokens} route_buckets mean={statistics.mean(route_bucket_counts):.2f} max={max(route_bucket_counts)}",
        f"PersistentKV indexed_qo={args.pkv_indexed}",
        f"PersistentKV single_bucket={args.pkv_single_bucket}",
        f"PersistentKV adaptive_route={args.pkv_adaptive_route}",
        f"PersistentKV workqueue={args.pkv_workqueue}",
        f"PersistentKV cuda_graph={args.pkv_cuda_graph}",
        f"PersistentKV fused_merge={args.pkv_fused_merge}",
        f"PersistentKV precompute_metadata={args.precompute_metadata or args.pkv_workqueue or args.adaptive_engine}",
        f"PersistentKV auto_splits={args.auto_splits}",
        f"Hq={HQ} Hkv={HKV} G={HQ // HKV} d={DIM} page={PAGE_SIZE} hole_fraction={args.hole_fraction:.2f} physical_pages={physical_pages}",
        "dynamic lengths grow by one token per active request per measured step",
        f"correctness vs FlashInfer: max_abs={max_err:.3e} mean_abs={mean_err:.3e} {correctness}",
        "",
        "| Engine | GPU tokens/s | GPU mean step ms | GPU p95 ms | GPU p99 ms | Wall tokens/s | Wall mean step ms | Wall p95 ms | Wall p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        format_row("FlashInfer ragged paged batch", flashinfer_stats),
    ]
    if args.adaptive_engine:
        assert route_counts is not None
        lines.extend([
            *(
                [format_row("PersistentKV length buckets", bucket_stats)]
                if bucket_stats is not None
                else []
            ),
            *(
                [format_row("PersistentKV compact workqueue", workqueue_stats)]
                if workqueue_stats is not None
                else []
            ),
            format_row("Adaptive routed engine", persistentkv_stats),
            "",
            f"GPU speed ratio Adaptive/FlashInfer tokens/s: {persistentkv_stats['event_tokens_s'] / flashinfer_stats['event_tokens_s']:.3f}x",
            f"Wall speed ratio Adaptive/FlashInfer tokens/s: {persistentkv_stats['wall_tokens_s'] / flashinfer_stats['wall_tokens_s']:.3f}x",
            f"Adaptive route counts: flashinfer={route_counts['flashinfer']} persistentkv_bucket={route_counts['persistentkv_bucket']} persistentkv_workqueue={route_counts['persistentkv_workqueue']}",
            f"Adaptive cost-model candidate splits: flashinfer={route_split_summary['flashinfer']} persistentkv_bucket={route_split_summary['persistentkv_bucket']} persistentkv_workqueue={route_split_summary['persistentkv_workqueue']}",
        ])
    else:
        lines.extend([
            format_row("PersistentKV page-aware length buckets", persistentkv_stats),
            "",
            f"GPU speed ratio PersistentKV/FlashInfer tokens/s: {persistentkv_stats['event_tokens_s'] / flashinfer_stats['event_tokens_s']:.3f}x",
            f"Wall speed ratio PersistentKV/FlashInfer tokens/s: {persistentkv_stats['wall_tokens_s'] / flashinfer_stats['wall_tokens_s']:.3f}x",
        ])
    lines.extend([
        f"total measured decode tokens={total_decode_tokens}",
    ])
    if args.adaptive_engine:
        assert adaptive_prof is not None
        lines.extend([
            "",
            "Causal profile counters (occupancy and traffic are structural proxies, not Nsight hardware counters)",
            "| Engine | CUDA launches/step | Non-empty CTA fraction | Decode CTA work/SM | Useful KV read MB/step | Merge state MB/step | Merge launches/step | Python+scheduling ms/step |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            format_profile_row("FlashInfer ragged paged batch", flash_profile, flashinfer_stats),
            *(
                [format_profile_row("PersistentKV length buckets", bucket_prof, bucket_stats)]
                if bucket_prof is not None and bucket_stats is not None
                else []
            ),
            *(
                [format_profile_row("PersistentKV compact workqueue", workqueue_prof, workqueue_stats)]
                if workqueue_prof is not None and workqueue_stats is not None
                else []
            ),
            format_profile_row("Adaptive routed engine", adaptive_prof, persistentkv_stats),
        ])
    lines.extend([
        "",
        "Notes:",
        "- FlashInfer uses one ragged native-paged batch per decode step.",
        "- PersistentKV normally groups active requests by current context length and launches once per route bucket.",
        "- PersistentKV compact workqueue emits only non-empty row/KV-head/split tasks and then segmented-merges partial softmax states.",
        "- Adaptive routing uses a lightweight calibrated cost model with a GQA gate: unsupported GQA ratios route to FlashInfer.",
        "- With --pkv-single-bucket, PersistentKV uses one masked ragged launch and row-local sequence bounds.",
        "- With --bucket-tokens > 0, PersistentKV rounds route lengths up; masked kernels use true per-row lengths.",
        "- FlashInfer plan() is included in synchronized wall time; CUDA-event time captures queued GPU work.",
        "- This is a first serving-loop experiment, not a full vLLM/TensorRT serving stack.",
    ])
    text = "\n".join(lines)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
