// micro_benchmark 04: Shared-memory double-buffering pipeline idle time
// Measures the overlap between async HBM->smem copies and compute.
// Uses PTX-level cp.async for maximum reliability on SM 8.6.
// Compile: nvcc -O3 -arch=sm_86 -std=c++17 04_double_buffering.cu -o 04_double_buffering

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>

#define CHECK(x) do { \
    cudaError_t err__ = (x); \
    if(err__ != cudaSuccess){ \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static constexpr int TILE    = 32;
static constexpr int D       = 128;
static constexpr int THREADS = 128;

// ─── Inline PTX cp.async wrappers ─────────────────────────────────────────────
// Copy 16 bytes (8 x FP16) asynchronously — cp.async.cg requires exactly 16 bytes.
__device__ __forceinline__ void cp_async_16(void* __restrict__ dst,
                                             const void* __restrict__ src) {
    unsigned int dst_smem = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_smem), "l"((const char*)src)
        : "memory");
}

// Commit the current group of cp.async operations.
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until at most N groups are still pending.
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// =====================================================================
// VARIANT A: Single-buffered (sync load, then compute) — baseline
// =====================================================================
__global__ void single_buffer_decode(
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const float*  __restrict__ Q,
    float* __restrict__ out,
    int N)
{
    // Static shared memory: one K tile + one V tile
    __shared__ __half smem[2][TILE * D];   // [0]=K, [1]=V
    int tid = threadIdx.x;

    float m = -1e20f, lv = 0.f, acc = 0.f;
    float q = (tid < D) ? Q[tid] : 0.f;

    for (int t0 = 0; t0 < N; t0 += TILE) {
        int tsz = min(TILE, N - t0);
        // --- Synchronous load ---
        __syncthreads();
        for (int elem = tid; elem < tsz * D; elem += THREADS) {
            smem[0][elem] = K[(t0 + elem/D)*D + elem%D];
            smem[1][elem] = V[(t0 + elem/D)*D + elem%D];
        }
        __syncthreads();

        for (int t = 0; t < tsz; t++) {
            float s = (tid < D) ? q * __half2float(smem[0][t*D + tid]) / 11.31f : 0.f;
            for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
            s = __shfl_sync(0xffffffff, s, 0);
            float nm = max(m, s), corr = expf(m - nm), ew = expf(s - nm);
            lv = corr * lv + ew; m = nm;
            if (tid < D) acc = corr * acc + ew * __half2float(smem[1][t*D + tid]);
        }
    }
    if (tid < D) out[tid] = (lv > 1e-8f) ? acc / lv : 0.f;
}

// =====================================================================
// VARIANT B: Double-buffered with PTX cp.async — hides memory latency
// Two ping-pong slots: while computing on slot 0, prefetch slot 1.
// Each slot holds K+V for one tile.
// smem layout: [slot0_K | slot0_V | slot1_K | slot1_V]
// =====================================================================
__global__ void double_buffer_decode(
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const float*  __restrict__ Q,
    float* __restrict__ out,
    int N)
{
    // 4 half-precision buffers: 4 * 32 * 128 * 2 = 32 KB < 48 KB limit
    __shared__ __half smem_k[2][TILE * D];
    __shared__ __half smem_v[2][TILE * D];
    int tid = threadIdx.x;
    int num_tiles = (N + TILE - 1) / TILE;

    float m = -1e20f, lv = 0.f, acc = 0.f;
    float q = (tid < D) ? Q[tid] : 0.f;

    // ─── Prefetch tile 0 into slot 0 ────────────────────────────────────────
    {
        int tsz = min(TILE, N);
        // Copy 16 bytes (8 FP16) per iteration — cp.async.cg requires 16B
        int total_chunks = (tsz * D) / 8;   // 8 halves = 16 bytes per chunk
        for (int chunk = tid; chunk < total_chunks; chunk += THREADS) {
            int elem = chunk * 8;  // starting half index
            cp_async_16(&smem_k[0][elem], &K[elem]);   // k starts at t0=0, so offset=elem
            cp_async_16(&smem_v[0][elem], &V[elem]);
        }
        cp_async_commit();
    }

    for (int tile = 0; tile < num_tiles; tile++) {
        int cur = tile & 1;
        int nxt = 1 - cur;
        int t0  = tile * TILE;

        // ─── Prefetch NEXT tile into slot nxt ─────────────────────────────
        if (tile + 1 < num_tiles) {
            int nt0  = (tile + 1) * TILE;
            int ntsz = min(TILE, N - nt0);
            int total_chunks = (ntsz * D) / 8;
            for (int chunk = tid; chunk < total_chunks; chunk += THREADS) {
                int local_elem = chunk * 8;
                int global_elem = nt0 * D + local_elem;  // flat offset into [N,D]
                cp_async_16(&smem_k[nxt][local_elem], &K[global_elem]);
                cp_async_16(&smem_v[nxt][local_elem], &V[global_elem]);
            }
            cp_async_commit();
        }

        // ─── Wait for CURRENT tile (allow 1 group still in flight) ────────
        // If no next tile, wait for all groups.
        if (tile + 1 < num_tiles) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        int tsz = min(TILE, N - t0);
        for (int t = 0; t < tsz; t++) {
            float s = (tid < D) ? q * __half2float(smem_k[cur][t*D + tid]) / 11.31f : 0.f;
            for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
            s = __shfl_sync(0xffffffff, s, 0);
            float nm = max(m, s), corr = expf(m - nm), ew = expf(s - nm);
            lv = corr * lv + ew; m = nm;
            if (tid < D) acc = corr * acc + ew * __half2float(smem_v[cur][t*D + tid]);
        }
    }
    if (tid < D) out[tid] = (lv > 1e-8f) ? acc / lv : 0.f;
}

int main() {
    printf("=== Double-Buffering Pipeline Idle Time (Ampere SM8.6) ===\n");
    printf("TILE=%d, D=%d, THREADS=%d\n", TILE, D, THREADS);
    printf("Single buf smem: 16 KB  |  Double buf smem: 32 KB\n");
    printf("Using PTX cp.async.cg (cache-global, bypasses L1)\n\n");

    const int ITERS = 200;
    float *d_Q, *d_O;
    CHECK(cudaMalloc(&d_Q, D * sizeof(float)));
    CHECK(cudaMalloc(&d_O, D * sizeof(float)));
    CHECK(cudaMemset(d_Q, 0, D * sizeof(float)));

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    for (int N : {4096, 8192, 16384, 32768}) {
        __half *d_K, *d_V;
        size_t kv_bytes = (size_t)N * D * sizeof(__half);
        CHECK(cudaMalloc(&d_K, kv_bytes));
        CHECK(cudaMalloc(&d_V, kv_bytes));
        CHECK(cudaMemset(d_K, 0, kv_bytes));
        CHECK(cudaMemset(d_V, 0, kv_bytes));

        // Warm-up
        for (int i = 0; i < 3; i++) {
            single_buffer_decode<<<1, THREADS>>>(d_K, d_V, d_Q, d_O, N);
            double_buffer_decode<<<1, THREADS>>>(d_K, d_V, d_Q, d_O, N);
        }
        CHECK(cudaDeviceSynchronize());

        // Time single
        CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < ITERS; i++)
            single_buffer_decode<<<1, THREADS>>>(d_K, d_V, d_Q, d_O, N);
        CHECK(cudaEventRecord(ev1));
        float ms_single;
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaEventElapsedTime(&ms_single, ev0, ev1));
        ms_single /= ITERS;

        // Time double
        CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < ITERS; i++)
            double_buffer_decode<<<1, THREADS>>>(d_K, d_V, d_Q, d_O, N);
        CHECK(cudaEventRecord(ev1));
        float ms_double;
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaEventElapsedTime(&ms_double, ev0, ev1));
        ms_double /= ITERS;

        double kv_bytes_d = 2.0 * N * D * sizeof(__half);
        double gbps_single = kv_bytes_d / (ms_single * 1e6);
        double gbps_double = kv_bytes_d / (ms_double * 1e6);

        printf("N=%6d | single=%6.3f ms (%5.1f GB/s) | double=%6.3f ms (%5.1f GB/s) | gain=%.2fx\n",
               N, ms_single, gbps_single, ms_double, gbps_double, ms_single / ms_double);

        cudaFree(d_K); cudaFree(d_V);
    }

    printf("\n>>> If gain > 1.1x: async copy hides memory latency meaningfully.\n");
    printf(">>> Profile with:\n");
    printf("    ncu --metrics smsp__warp_issue_stalled_long_sb_per_warp_active.pct,"
           "l1tex__t_sector_hit_rate.pct ./04_double_buffering\n");

    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
    cudaFree(d_Q); cudaFree(d_O);
    return 0;
}
