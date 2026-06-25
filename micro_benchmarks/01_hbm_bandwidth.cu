// micro_benchmark 01: Raw HBM read bandwidth (the ceiling for KV-streaming kernels)
// RTX 3060 Ampere SM 8.6, theoretical peak ~360 GB/s
// Compile: nvcc -O3 -arch=sm_86 01_hbm_bandwidth.cu -o 01_hbm_bandwidth
// Run:     ./01_hbm_bandwidth

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if(e != cudaSuccess){ \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Read-only streaming: every thread reads 128 bits (float4) sequentially
__global__ void hbm_read_kernel(const float4* __restrict__ src, float4* __restrict__ sink,
                                 int64_t n_float4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int64_t i = idx; i < n_float4; i += stride) {
        float4 v = __ldg(&src[i]);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    // Write to prevent dead-code elimination
    if (idx == 0) *sink = acc;
}

// Write bandwidth
__global__ void hbm_write_kernel(float4* __restrict__ dst, int64_t n_float4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    for (int64_t i = idx; i < n_float4; i += stride) {
        dst[i] = v;
    }
}

// Read+Write (copy): models KV-load + output-write path
__global__ void hbm_copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst,
                                 int64_t n_float4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n_float4; i += stride) {
        dst[i] = __ldg(&src[i]);
    }
}

static double bench_kernel(void (*fn)(float4*, float4*, int64_t), // for copy variant
                            const float4* src, float4* dst, int64_t n_float4,
                            int variant, // 0=read, 1=write, 2=copy
                            int repeats) {
    // Warm-up
    for (int r = 0; r < 2; r++) {
        if (variant == 0) hbm_read_kernel<<<256, 512>>>( (const float4*)src, dst, n_float4);
        else if (variant == 1) hbm_write_kernel<<<256, 512>>>(dst, n_float4);
        else hbm_copy_kernel<<<256, 512>>>((const float4*)src, dst, n_float4);
    }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; r++) {
        if (variant == 0) hbm_read_kernel<<<256, 512>>>((const float4*)src, dst, n_float4);
        else if (variant == 1) hbm_write_kernel<<<256, 512>>>(dst, n_float4);
        else hbm_copy_kernel<<<256, 512>>>((const float4*)src, dst, n_float4);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return (double)ms / repeats;
}

int main() {
    printf("=== HBM Bandwidth Micro-Benchmark (RTX 3060, SM 8.6) ===\n\n");

    // 512 MB buffer (floats)
    const int64_t N_BYTES = 512LL * 1024 * 1024;
    const int64_t N_FLOAT4 = N_BYTES / sizeof(float4);

    float4 *d_src, *d_dst, *d_sink;
    CHECK(cudaMalloc(&d_src, N_BYTES));
    CHECK(cudaMalloc(&d_dst, N_BYTES));
    CHECK(cudaMalloc(&d_sink, sizeof(float4)));
    CHECK(cudaMemset(d_src, 0, N_BYTES));

    const int REPEATS = 20;

    // READ
    double ms_read = bench_kernel(nullptr, d_src, d_sink, N_FLOAT4, 0, REPEATS);
    double gbps_read = (double)N_BYTES / (ms_read * 1e6);
    printf("[READ ]  %6.2f ms  ->  %6.1f GB/s  (bytes=%lld)\n",
           ms_read, gbps_read, (long long)N_BYTES);

    // WRITE
    double ms_write = bench_kernel(nullptr, nullptr, d_dst, N_FLOAT4, 1, REPEATS);
    double gbps_write = (double)N_BYTES / (ms_write * 1e6);
    printf("[WRITE]  %6.2f ms  ->  %6.1f GB/s\n", ms_write, gbps_write);

    // COPY (read+write)
    double ms_copy = bench_kernel(nullptr, d_src, d_dst, N_FLOAT4, 2, REPEATS);
    double gbps_copy = 2.0 * N_BYTES / (ms_copy * 1e6);   // both directions
    printf("[COPY ]  %6.2f ms  ->  %6.1f GB/s  (read+write)\n", ms_copy, gbps_copy);

    // Theoretical peak (RTX 3060: 192-bit GDDR6 at 15 Gbps = 360 GB/s)
    const double PEAK_BW = 360.0;
    printf("\nPeak theoretical BW (RTX 3060): %.0f GB/s\n", PEAK_BW);
    printf("Achieved READ efficiency:  %.1f%%\n", 100.0 * gbps_read / PEAK_BW);
    printf("Achieved COPY efficiency:  %.1f%%\n", 100.0 * gbps_copy / (2.0 * PEAK_BW));

    printf("\n--- KV-Cache Streaming Budget ---\n");
    // Typical GQA shape: Hq=32, Hkv=8, G=4, d=128, N tokens, BF16
    // Per token generated: we must stream N*Hkv*2*d*2 bytes  (K+V, BF16)
    for (int N : {4096, 8192, 16384, 32768}) {
        int Hkv = 8, d = 128;
        double kv_bytes = (double)N * Hkv * 2 * d * 2;  // BF16
        double kv_gb = kv_bytes / 1e9;
        double t_ms = kv_bytes / (gbps_read * 1e9) * 1e3;
        printf("  N=%6d: KV=%.2f MB -> %.3f ms floor (memory-bound, GQA-centric)\n",
               N, kv_bytes/1e6, t_ms);
    }

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_sink);
    return 0;
}
