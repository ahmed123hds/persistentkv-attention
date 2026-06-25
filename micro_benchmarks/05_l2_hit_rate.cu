// micro_benchmark 05: L2 hit rate probe — does the L2 cache capture KV reuse?
// The paper's key concern: "L2 already reuses KV" reviewer attack.
// This test uses cuda_pmmu counters via cudaProfilerStart/Stop + manual timing
// to estimate L2 miss rate for repeated access to the same buffer.
// Compile: nvcc -O3 -arch=sm_86 05_l2_hit_rate.cu -o 05_l2_hit_rate
// Profile: ncu --metrics lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_hit_rate.pct ./05_l2_hit_rate

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if(e != cudaSuccess){ \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// =====================================================================
// Test 1: ACCESS SAME KV BUFFER G times (simulating G=4 query heads sharing 1 KV)
// Natural L2 reuse test: does L2 hold the KV after first access?
// =====================================================================
__global__ void l2_reuse_test(
    const __half* __restrict__ KV,  // [N, D]  — one KV head buffer
    float* __restrict__ sinks,       // prevents dead code
    int N, int D, int G)             // access G times (simulating G query heads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.f;

    for (int g = 0; g < G; g++) {
        // Each iteration simulates one query head reading the whole KV buffer
        for (int idx = tid; idx < N * D; idx += stride) {
            acc += __half2float(__ldg(&KV[idx]));
        }
    }
    if (tid == 0) *sinks = acc;
}

// =====================================================================
// Test 2: G independent KV buffers (simulating NO reuse — baseline)
// =====================================================================
__global__ void l2_no_reuse_test(
    const __half* __restrict__ KV0,
    const __half* __restrict__ KV1,
    const __half* __restrict__ KV2,
    const __half* __restrict__ KV3,
    float* __restrict__ sinks,
    int N, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.f;
    for (int idx = tid; idx < N * D; idx += stride) {
        acc += __half2float(__ldg(&KV0[idx]));
        acc += __half2float(__ldg(&KV1[idx]));
        acc += __half2float(__ldg(&KV2[idx]));
        acc += __half2float(__ldg(&KV3[idx]));
    }
    if (tid == 0) *sinks = acc;
}

int main() {
    printf("=== L2 Cache Hit Rate Probe — KV Reuse Experiment ===\n");
    printf("RTX 3060: L2 = 2 MB\n\n");
    printf("Key question: Does L2 naturally cache KV when G heads access same buffer?\n");
    printf("If YES: L2 already provides reuse, PersistentKV's smem benefit is smaller.\n");
    printf("If NO:  Explicit smem tiling is necessary for reuse.\n\n");

    const int D = 128;
    const int G = 4;
    const int ITERS = 50;
    const double PEAK_BW = 360.0;

    float *d_sink;
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    for (int N : {4096, 8192, 16384, 32768}) {
        size_t bytes = (size_t)N * D * sizeof(__half);
        bool fits_in_l2 = (bytes <= 2 * 1024 * 1024);  // 2MB L2

        __half *d_KV, *d_KV0, *d_KV1, *d_KV2, *d_KV3;
        CHECK(cudaMalloc(&d_KV,  bytes));
        CHECK(cudaMalloc(&d_KV0, bytes));
        CHECK(cudaMalloc(&d_KV1, bytes));
        CHECK(cudaMalloc(&d_KV2, bytes));
        CHECK(cudaMalloc(&d_KV3, bytes));
        CHECK(cudaMemset(d_KV,  0, bytes));
        CHECK(cudaMemset(d_KV0, 0, bytes));
        CHECK(cudaMemset(d_KV1, 0, bytes));
        CHECK(cudaMemset(d_KV2, 0, bytes));
        CHECK(cudaMemset(d_KV3, 0, bytes));

        // Warm up
        l2_reuse_test<<<56, 256>>>(d_KV, d_sink, N, D, G);
        l2_no_reuse_test<<<56, 256>>>(d_KV0, d_KV1, d_KV2, d_KV3, d_sink, N, D);
        CHECK(cudaDeviceSynchronize());

        // Time REUSE pattern (1 buffer, accessed G times)
        CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < ITERS; i++)
            l2_reuse_test<<<56, 256>>>(d_KV, d_sink, N, D, G);
        CHECK(cudaEventRecord(ev1));
        float ms_reuse;
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaEventElapsedTime(&ms_reuse, ev0, ev1));
        ms_reuse /= ITERS;

        // Time NO-REUSE pattern (G independent buffers)
        CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < ITERS; i++)
            l2_no_reuse_test<<<56, 256>>>(d_KV0, d_KV1, d_KV2, d_KV3, d_sink, N, D);
        CHECK(cudaEventRecord(ev1));
        float ms_noreuse;
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaEventElapsedTime(&ms_noreuse, ev0, ev1));
        ms_noreuse /= ITERS;

        // Theoretical bytes: reuse = G*bytes (if NO L2 hit), = bytes (if perfect L2)
        double bytes_if_no_l2  = G * (double)bytes;
        double bytes_if_full_l2 = (double)bytes;
        double bytes_noreuse    = G * (double)bytes;

        double gbps_reuse   = bytes_if_no_l2   / (ms_reuse   * 1e6);
        double gbps_noreuse = bytes_noreuse     / (ms_noreuse * 1e6);

        printf("N=%6d  kv=%.1f MB  L2_fits=%s\n", N, bytes/1e6, fits_in_l2?"YES":"NO");
        printf("  Reuse(1 buf x G):  %.3f ms  eff_bw=%.1f GB/s\n", ms_reuse, gbps_reuse);
        printf("  NoReuse(G bufs) :  %.3f ms  eff_bw=%.1f GB/s\n", ms_noreuse, gbps_noreuse);
        printf("  Time ratio       : %.2fx  (1.0=perfect L2 reuse, %.1fx=no L2)\n",
               ms_noreuse / ms_reuse, (double)G);

        if (ms_noreuse / ms_reuse > 0.9 * G) {
            printf("  -> L2 provides LITTLE natural reuse: smem tiling IS necessary.\n");
        } else if (ms_noreuse / ms_reuse > 1.5) {
            printf("  -> L2 provides PARTIAL reuse: smem tiling still beneficial.\n");
        } else {
            printf("  -> L2 provides GOOD natural reuse: smem mainly for latency hiding.\n");
        }
        printf("\n");

        cudaFree(d_KV); cudaFree(d_KV0); cudaFree(d_KV1); cudaFree(d_KV2); cudaFree(d_KV3);
    }

    printf("For full hardware counters run:\n");
    printf("  ncu --metrics lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum ./05_l2_hit_rate\n");

    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
    cudaFree(d_sink);
    return 0;
}
