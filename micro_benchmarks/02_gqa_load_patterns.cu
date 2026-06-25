// micro_benchmark 02: GQA-group load patterns — baseline (per-query-head KV load)
//                     vs PersistentKV (load once, reuse across G query heads)
// Shows the DRAM traffic ratio analytically + measures shared-memory reuse cost.
// Compile: nvcc -O3 -arch=sm_86 02_gqa_load_patterns.cu -o 02_gqa_load_patterns
// Run:     ./02_gqa_load_patterns

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if(e != cudaSuccess){ \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// =====================================================================
// BASELINE: query-head-centric decode attention (naive GQA schedule)
// Each warp handles one query head; each warp independently loads K,V.
// This is the "bad" pattern we want to beat.
// =====================================================================
__global__ void gqa_baseline_decode(
    const __half* __restrict__ Q,   // [Hq, d]
    const __half* __restrict__ K,   // [Hkv, N, d]
    const __half* __restrict__ V,   // [Hkv, N, d]
    float* __restrict__ O,          // [Hq, d]
    int Hq, int Hkv, int N, int d)
{
    // One block = one query head
    int qh = blockIdx.x;
    int kvh = qh / (Hq / Hkv);  // GQA group mapping
    int tid = threadIdx.x;

    extern __shared__ float smem[];   // [2 * TILE * d/2] for K and V tiles
    // simplified: each thread accumulates one output element
    const int TILE = 32;  // tokens per tile
    float m = -1e20f, l = 0.f;
    float acc[128 / 32] = {};  // d=128, threads=128 -> each thread holds 1 element

    for (int tile_start = 0; tile_start < N; tile_start += TILE) {
        int tile_end = min(tile_start + TILE, N);
        // Load K tile into shared memory  — BASELINE loads per query head (redundant for GQA)
        __syncthreads();
        for (int t = tile_start + tid; t < tile_end; t += blockDim.x) {
            for (int dd = 0; dd < d; dd++) {
                // NOTE: in real baseline these loads happen once per query head
                // even when multiple query heads share the same KV head
                smem[((t - tile_start) * d + dd)] = __half2float(K[kvh * N * d + t * d + dd]);
            }
        }
        __syncthreads();
        // Compute Q * K^T for this query head (simplified for demo)
        float score = 0.f;
        if (tid < d) {
            float q = __half2float(Q[qh * d + tid]);
            for (int t = tile_start; t < tile_end; t++) {
                score += q * smem[(t - tile_start) * d + tid];
            }
        }
        // Online softmax update (simplified)
        float new_m = max(m, score);
        l = expf(m - new_m) * l + expf(score - new_m);
        m = new_m;
    }
    if (tid < d) O[qh * d + tid] = m;  // placeholder output
}

// =====================================================================
// PERSISTENTKV: KV-head-group-centric decode attention
// One block = one KV head; all G query heads share the same K,V tiles.
// K and V are loaded ONCE into shared memory, reused across G query heads.
// =====================================================================
__global__ void gqa_persistentkv_decode(
    const __half* __restrict__ Q,   // [Hq, d]
    const __half* __restrict__ K,   // [Hkv, N, d]
    const __half* __restrict__ V,   // [Hkv, N, d]
    float* __restrict__ O,          // [Hq, d]
    int Hq, int Hkv, int N, int d, int G)
{
    // One block = one KV head group (covers G query heads)
    int kvh = blockIdx.x;
    int tid = threadIdx.x;
    // Identify G query heads belonging to this KV head
    int qh_base = kvh * G;

    extern __shared__ float smem[];   // [2 * TILE * d] for K tile + V tile
    float* k_smem = smem;
    float* v_smem = smem + 32 * d;   // TILE=32

    const int TILE = 32;
    // Per-query-head online softmax state (G sets), keep in registers when G is small
    float m[8], l_acc[8];   // max G=8
    float acc[8][4];         // simplified: each thread tracks 4 output dims per query head
    for (int g = 0; g < G; g++) { m[g] = -1e20f; l_acc[g] = 0.f; }

    for (int tile_start = 0; tile_start < N; tile_start += TILE) {
        int tile_end = min(tile_start + TILE, N);
        // *** Load K,V tile ONCE for the whole KV group ***
        __syncthreads();
        for (int elem = tid; elem < (tile_end - tile_start) * d; elem += blockDim.x) {
            int t = elem / d;
            int dd = elem % d;
            int global_t = tile_start + t;
            k_smem[t * d + dd] = __half2float(K[kvh * N * d + global_t * d + dd]);
            v_smem[t * d + dd] = __half2float(V[kvh * N * d + global_t * d + dd]);
        }
        __syncthreads();
        // *** Reuse K,V tile for ALL G query heads ***
        for (int g = 0; g < G; g++) {
            int qh = qh_base + g;
            float score = 0.f;
            if (tid < d) {
                float q = __half2float(Q[qh * d + tid]);
                for (int t = 0; t < tile_end - tile_start; t++) {
                    score += q * k_smem[t * d + tid];
                }
            }
            // Online softmax
            float new_m = max(m[g], score);
            l_acc[g] = expf(m[g] - new_m) * l_acc[g] + expf(score - new_m);
            m[g] = new_m;
        }
    }
    // Write outputs
    for (int g = 0; g < G; g++) {
        if (tid == 0) O[(qh_base + g) * d] = m[g];
    }
}

// =====================================================================
// Benchmark: time both kernels, compare effective DRAM loads
// =====================================================================
double time_kernel(cudaEvent_t ev_start, cudaEvent_t ev_stop, int iters) {
    float ms;
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    return (double)ms / iters;
}

int main() {
    printf("=== GQA Load Pattern Micro-Benchmark ===\n\n");

    // Hardware
    const double PEAK_BW_GBS = 360.0;

    // GQA config mimicking Llama-style
    const int Hq = 32, Hkv = 8, G = Hq / Hkv;  // G=4
    const int d = 128;
    const int ITERS = 50;

    printf("GQA config: Hq=%d, Hkv=%d, G=%d, d=%d\n\n", Hq, Hkv, G, d);

    for (int N : {4096, 8192, 16384, 32768}) {
        size_t q_bytes = (size_t)Hq * d * sizeof(__half);
        size_t kv_bytes = (size_t)Hkv * N * d * sizeof(__half);

        __half *d_Q, *d_K, *d_V;
        float *d_O;
        CHECK(cudaMalloc(&d_Q, q_bytes));
        CHECK(cudaMalloc(&d_K, kv_bytes));
        CHECK(cudaMalloc(&d_V, kv_bytes));
        CHECK(cudaMalloc(&d_O, (size_t)Hq * d * sizeof(float)));
        CHECK(cudaMemset(d_Q, 0, q_bytes));
        CHECK(cudaMemset(d_K, 0, kv_bytes));
        CHECK(cudaMemset(d_V, 0, kv_bytes));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        size_t smem_bytes = 2 * 32 * d * sizeof(float);

        // Warm up
        gqa_baseline_decode<<<Hq, 128, smem_bytes>>>(d_Q, d_K, d_V, d_O, Hq, Hkv, N, d);
        gqa_persistentkv_decode<<<Hkv, 128, smem_bytes>>>(d_Q, d_K, d_V, d_O, Hq, Hkv, N, d, G);
        CHECK(cudaDeviceSynchronize());

        // Time baseline
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            gqa_baseline_decode<<<Hq, 128, smem_bytes>>>(d_Q, d_K, d_V, d_O, Hq, Hkv, N, d);
        CHECK(cudaEventRecord(stop));
        double ms_base = time_kernel(start, stop, ITERS);

        // Time PersistentKV
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            gqa_persistentkv_decode<<<Hkv, 128, smem_bytes>>>(d_Q, d_K, d_V, d_O, Hq, Hkv, N, d, G);
        CHECK(cudaEventRecord(stop));
        double ms_pkv = time_kernel(start, stop, ITERS);

        // Theoretical DRAM traffic
        double base_dram = 2.0 * Hq * N * d * 2;  // K+V, BF16(2bytes), per query head
        double pkv_dram  = 2.0 * Hkv * N * d * 2; // K+V loaded once per KV group
        double base_bw = base_dram / (ms_base * 1e6);
        double pkv_bw  = pkv_dram  / (ms_pkv  * 1e6);

        printf("N=%6d:\n", N);
        printf("  Baseline  : %7.3f ms | %6.1f GB/s effective | theoretical DRAM=%.1f MB\n",
               ms_base, base_bw, base_dram/1e6);
        printf("  PersistKV : %7.3f ms | %6.1f GB/s effective | theoretical DRAM=%.1f MB\n",
               ms_pkv, pkv_bw, pkv_dram/1e6);
        printf("  Speedup   : %.2fx  |  DRAM reduction target: %.1fx (G=%d)\n",
               ms_base / ms_pkv, (double)Hq / Hkv, G);
        printf("  Peak BW utilization: baseline=%.1f%%, PersistKV=%.1f%%\n\n",
               100.0 * base_bw / PEAK_BW_GBS, 100.0 * pkv_bw / PEAK_BW_GBS);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }
    return 0;
}
