// =============================================================================
// smoke_test.cu — Correctness + Performance verification for PersistentKV
// =============================================================================
// Tests:
//   1. Numerical correctness against a CPU FP32 reference
//   2. Direct and sequence-split PersistentKV paths
//   3. Contiguous and paged KV layouts
//   4. Exact query-head baseline comparison
//   5. Shape sweep and effective bandwidth reporting
// Compile: nvcc -O3 -arch=sm_86 smoke_test.cu -o smoke_test
// Run:     ./smoke_test

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <assert.h>

// Include the full PersistentKV kernel
#include "../kernels/persistentkv_attention.cuh"

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if(e != cudaSuccess){ \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ─── CPU Reference (exact FP32) ──────────────────────────────────────────────
// O[b, qh, d] = softmax(Q[b,qh,:] @ K[b,kvh,:,:].T / sqrt(d)) @ V[b,kvh,:,:]
void cpu_gqa_attention(
    const float* Q,   // [B, Hq, d]
    const float* K,   // [B, Hkv, N, d]
    const float* V,   // [B, Hkv, N, d]
    float* O,         // [B, Hq, d]
    int B, int Hq, int Hkv, int N, int d)
{
    int G = Hq / Hkv;
    float scale = 1.0f / sqrtf((float)d);

    for (int b = 0; b < B; b++) {
        for (int qh = 0; qh < Hq; qh++) {
            int kvh = qh / G;
            const float* q = Q + b * Hq * d + qh * d;
            const float* Kh = K + b * Hkv * N * d + kvh * N * d;
            const float* Vh = V + b * Hkv * N * d + kvh * N * d;
            float* o = O + b * Hq * d + qh * d;

            // Compute scores
            float m = -FLT_MAX, l = 0.f;
            float* scores = (float*)malloc(N * sizeof(float));
            for (int t = 0; t < N; t++) {
                float s = 0.f;
                for (int dd = 0; dd < d; dd++) s += q[dd] * Kh[t * d + dd];
                s *= scale;
                scores[t] = s;
                m = fmaxf(m, s);
            }
            // Softmax + weighted sum
            memset(o, 0, d * sizeof(float));
            for (int t = 0; t < N; t++) {
                float w = expf(scores[t] - m);
                l += w;
                for (int dd = 0; dd < d; dd++) o[dd] += w * Vh[t * d + dd];
            }
            for (int dd = 0; dd < d; dd++) o[dd] /= (l + 1e-8f);
            free(scores);
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
float rand_float() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

void fill_random(float* arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = rand_float();
}

void fp32_to_fp16(const float* src, __half* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

void fp16_to_fp32(const __half* src, float* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = __half2float(src[i]);
}

double max_abs_error(const float* ref, const float* got, int n) {
    double mx = 0.0;
    for (int i = 0; i < n; i++) mx = fmax(mx, fabs((double)ref[i] - got[i]));
    return mx;
}

double mean_abs_error(const float* ref, const float* got, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += fabs((double)ref[i] - got[i]);
    return s / n;
}

// ─── Correctness Test ─────────────────────────────────────────────────────────
bool test_correctness(int B, int Hq, int Hkv, int N, int d, int num_splits,
                       bool check_baseline=false,
                       double tol_max=2e-3, double tol_mean=3e-4) {
    int G = Hq / Hkv;
    if (d != PKV_D) { printf("  [SKIP] d=%d != PKV_D=%d\n", d, PKV_D); return true; }
    if (G > PKV_MAX_G) { printf("  [SKIP] G=%d > PKV_MAX_G=%d\n", G, PKV_MAX_G); return true; }

    int q_n = B * Hq * d;
    int kv_n = B * Hkv * N * d;
    int o_n = B * Hq * d;

    float *h_Q = new float[q_n];
    float *h_K = new float[kv_n];
    float *h_V = new float[kv_n];
    float *h_O_ref = new float[o_n]();
    float *h_O_got = new float[o_n];

    fill_random(h_Q, q_n);
    fill_random(h_K, kv_n);
    fill_random(h_V, kv_n);

    // Quantize first, then run the reference on exactly the values seen by CUDA.
    __half *d_Q, *d_K, *d_V, *d_O;
    __half *h_Q16 = new __half[q_n];
    __half *h_K16 = new __half[kv_n];
    __half *h_V16 = new __half[kv_n];
    __half *h_O16 = new __half[o_n];

    fp32_to_fp16(h_Q, h_Q16, q_n);
    fp32_to_fp16(h_K, h_K16, kv_n);
    fp32_to_fp16(h_V, h_V16, kv_n);
    fp16_to_fp32(h_Q16, h_Q, q_n);
    fp16_to_fp32(h_K16, h_K, kv_n);
    fp16_to_fp32(h_V16, h_V, kv_n);
    cpu_gqa_attention(h_Q, h_K, h_V, h_O_ref, B, Hq, Hkv, N, d);

    CHECK(cudaMalloc(&d_Q, q_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_K, kv_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_V, kv_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_O, o_n * sizeof(__half)));

    CHECK(cudaMemcpy(d_Q, h_Q16, q_n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K, h_K16, kv_n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, h_V16, kv_n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_O, 0, o_n * sizeof(__half)));

    void* d_workspace = nullptr;
    size_t workspace_bytes =
        persistentkv_workspace_bytes(B, Hq, num_splits);
    if (workspace_bytes > 0)
        CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    if (num_splits == 1) {
        persistentkv_dispatch(d_Q, d_K, d_V, d_O, B, Hq, Hkv, N, d);
    } else {
        persistentkv_split_dispatch(
            d_Q, d_K, d_V, d_O, d_workspace,
            B, Hq, Hkv, N, d, num_splits);
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_O16, d_O, o_n * sizeof(__half), cudaMemcpyDeviceToHost));
    fp16_to_fp32(h_O16, h_O_got, o_n);

    double max_err  = max_abs_error(h_O_ref, h_O_got, o_n);
    double mean_err = mean_abs_error(h_O_ref, h_O_got, o_n);
    bool pass = (max_err < tol_max) && (mean_err < tol_mean);

    printf("  B=%d Hq=%d Hkv=%d G=%d N=%d splits=%d  "
           "max_err=%.2e mean_err=%.2e  [%s]\n",
           B, Hq, Hkv, G, N, num_splits,
           max_err, mean_err, pass ? "PASS" : "FAIL");

    if (check_baseline) {
        CHECK(cudaMemset(d_O, 0, o_n * sizeof(__half)));
        persistentkv_baseline_dispatch(
            d_Q, d_K, d_V, d_O, B, Hq, Hkv, N, d);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(
            h_O16, d_O, o_n * sizeof(__half), cudaMemcpyDeviceToHost));
        fp16_to_fp32(h_O16, h_O_got, o_n);
        double baseline_max = max_abs_error(h_O_ref, h_O_got, o_n);
        double baseline_mean = mean_abs_error(h_O_ref, h_O_got, o_n);
        bool baseline_pass =
            baseline_max < tol_max && baseline_mean < tol_mean;
        printf("    query-head baseline             "
               "max_err=%.2e mean_err=%.2e  [%s]\n",
               baseline_max, baseline_mean,
               baseline_pass ? "PASS" : "FAIL");
        pass &= baseline_pass;
    }

    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O_ref; delete[] h_O_got;
    delete[] h_Q16; delete[] h_K16; delete[] h_V16; delete[] h_O16;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    if (d_workspace) cudaFree(d_workspace);
    return pass;
}

bool test_paged_correctness(
    int B, int Hq, int Hkv, int N, int d,
    int page_size, int num_splits,
    double tol_max=2e-3, double tol_mean=3e-4)
{
    int G = Hq / Hkv;
    int q_n = B * Hq * d;
    int kv_n = B * Hkv * N * d;
    int o_n = B * Hq * d;
    int pages_per_batch = (N + page_size - 1) / page_size;
    int total_pages = B * pages_per_batch;
    size_t paged_n = (size_t)total_pages * Hkv * page_size * d;

    float* h_Q = new float[q_n];
    float* h_K = new float[kv_n];
    float* h_V = new float[kv_n];
    float* h_O_ref = new float[o_n]();
    float* h_O_got = new float[o_n];
    __half* h_Q16 = new __half[q_n];
    __half* h_K16 = new __half[kv_n];
    __half* h_V16 = new __half[kv_n];
    __half* h_K_pages = new __half[paged_n]();
    __half* h_V_pages = new __half[paged_n]();
    __half* h_O16 = new __half[o_n];
    int* h_table = new int[B * pages_per_batch];

    fill_random(h_Q, q_n);
    fill_random(h_K, kv_n);
    fill_random(h_V, kv_n);
    fp32_to_fp16(h_Q, h_Q16, q_n);
    fp32_to_fp16(h_K, h_K16, kv_n);
    fp32_to_fp16(h_V, h_V16, kv_n);
    fp16_to_fp32(h_Q16, h_Q, q_n);
    fp16_to_fp32(h_K16, h_K, kv_n);
    fp16_to_fp32(h_V16, h_V, kv_n);
    cpu_gqa_attention(h_Q, h_K, h_V, h_O_ref, B, Hq, Hkv, N, d);

    // Reverse each batch's physical page order to exercise block-table lookup.
    for (int b = 0; b < B; ++b) {
        for (int logical = 0; logical < pages_per_batch; ++logical) {
            int physical =
                b * pages_per_batch + (pages_per_batch - 1 - logical);
            h_table[b * pages_per_batch + logical] = physical;
            for (int offset = 0; offset < page_size; ++offset) {
                int token = logical * page_size + offset;
                if (token >= N) continue;
                for (int kvh = 0; kvh < Hkv; ++kvh) {
                    size_t src =
                        ((size_t)b * Hkv * N + (size_t)kvh * N + token) * d;
                    size_t dst =
                        (((size_t)physical * Hkv + kvh) * page_size + offset) * d;
                    memcpy(h_K_pages + dst, h_K16 + src, d * sizeof(__half));
                    memcpy(h_V_pages + dst, h_V16 + src, d * sizeof(__half));
                }
            }
        }
    }

    __half *d_Q, *d_K_pages, *d_V_pages, *d_O;
    int* d_table;
    void* d_workspace = nullptr;
    CHECK(cudaMalloc(&d_Q, q_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_K_pages, paged_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_V_pages, paged_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_O, o_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_table, B * pages_per_batch * sizeof(int)));
    size_t workspace_bytes =
        persistentkv_workspace_bytes(B, Hq, num_splits);
    if (workspace_bytes > 0)
        CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    CHECK(cudaMemcpy(
        d_Q, h_Q16, q_n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_K_pages, h_K_pages, paged_n * sizeof(__half),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_V_pages, h_V_pages, paged_n * sizeof(__half),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_table, h_table, B * pages_per_batch * sizeof(int),
        cudaMemcpyHostToDevice));

    persistentkv_paged_dispatch(
        d_Q, d_K_pages, d_V_pages, d_table, d_O, d_workspace,
        B, Hq, Hkv, N, d, page_size, pages_per_batch, num_splits);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(
        h_O16, d_O, o_n * sizeof(__half), cudaMemcpyDeviceToHost));
    fp16_to_fp32(h_O16, h_O_got, o_n);

    double max_err = max_abs_error(h_O_ref, h_O_got, o_n);
    double mean_err = mean_abs_error(h_O_ref, h_O_got, o_n);
    bool pass = max_err < tol_max && mean_err < tol_mean;
    printf("  PAGED B=%d Hq=%d Hkv=%d G=%d N=%d page=%d splits=%d  "
           "max_err=%.2e mean_err=%.2e  [%s]\n",
           B, Hq, Hkv, G, N, page_size, num_splits,
           max_err, mean_err, pass ? "PASS" : "FAIL");

    delete[] h_Q; delete[] h_K; delete[] h_V;
    delete[] h_O_ref; delete[] h_O_got; delete[] h_Q16;
    delete[] h_K16; delete[] h_V16; delete[] h_K_pages;
    delete[] h_V_pages; delete[] h_O16; delete[] h_table;
    cudaFree(d_Q); cudaFree(d_K_pages); cudaFree(d_V_pages);
    cudaFree(d_O); cudaFree(d_table);
    if (d_workspace) cudaFree(d_workspace);
    return pass;
}

// ─── Performance Benchmark ────────────────────────────────────────────────────
void bench_perf(
    int B, int Hq, int Hkv, int N, int d,
    int num_splits, bool compare_baseline=false, int iters=100)
{
    int G = Hq / Hkv;
    if (d != PKV_D || G > PKV_MAX_G) return;

    int q_n = B * Hq * d;
    int kv_n = B * Hkv * N * d;
    int o_n = B * Hq * d;

    __half *d_Q, *d_K, *d_V, *d_O;
    void* d_workspace = nullptr;
    CHECK(cudaMalloc(&d_Q, q_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_K, kv_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_V, kv_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_O, o_n * sizeof(__half)));
    CHECK(cudaMemset(d_Q, 0, q_n * sizeof(__half)));
    CHECK(cudaMemset(d_K, 0, kv_n * sizeof(__half)));
    CHECK(cudaMemset(d_V, 0, kv_n * sizeof(__half)));
    size_t workspace_bytes =
        persistentkv_workspace_bytes(B, Hq, num_splits);
    if (workspace_bytes > 0)
        CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    // Warm up
    for (int i = 0; i < 5; i++) {
        persistentkv_split_dispatch(
            d_Q, d_K, d_V, d_O, d_workspace,
            B, Hq, Hkv, N, d, num_splits);
        if (compare_baseline)
            persistentkv_baseline_dispatch(
                d_Q, d_K, d_V, d_O, B, Hq, Hkv, N, d);
    }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++) {
        persistentkv_split_dispatch(
            d_Q, d_K, d_V, d_O, d_workspace,
            B, Hq, Hkv, N, d, num_splits);
    }
    CHECK(cudaEventRecord(ev1));
    float pkv_ms;
    CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&pkv_ms, ev0, ev1));
    pkv_ms /= iters;

    float baseline_ms = 0.0f;
    if (compare_baseline) {
        CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; i++) {
            persistentkv_baseline_dispatch(
                d_Q, d_K, d_V, d_O, B, Hq, Hkv, N, d);
        }
        CHECK(cudaEventRecord(ev1));
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaEventElapsedTime(&baseline_ms, ev0, ev1));
        baseline_ms /= iters;
    }

    // KV bandwidth (primary bottleneck)
    double kv_bytes = 2.0 * B * Hkv * N * d * 2;  // K+V, BF16
    double gbps = kv_bytes / (pkv_ms * 1e6);

    // Baseline: if each query head loaded KV independently
    double kv_bytes_baseline = 2.0 * B * Hq * N * d * 2;  // G times more
    printf("  B=%2d Hq=%2d Hkv=%d G=%d N=%6d splits=%2d | "
           "time=%6.3f ms | BW=%5.1f GB/s | efficiency=%.0f%% | "
           "baseline_dram=%.1fMB pkv_dram=%.1fMB",
           B, Hq, Hkv, G, N, num_splits, pkv_ms, gbps,
           gbps / 360.0 * 100.0,
           kv_bytes_baseline/1e6, kv_bytes/1e6);
    if (compare_baseline) {
        printf(" | query_head=%6.3f ms speedup=%.2fx",
               baseline_ms, baseline_ms / pkv_ms);
    }
    printf("\n");

    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    if (d_workspace) cudaFree(d_workspace);
}

int run_profile_target() {
    const int B = 1;
    const int Hq = 32;
    const int Hkv = 8;
    const int N = 8192;
    const int d = 128;
    const int num_splits = 16;
    size_t q_n = (size_t)B * Hq * d;
    size_t kv_n = (size_t)B * Hkv * N * d;
    size_t o_n = (size_t)B * Hq * d;

    __half *d_Q, *d_K, *d_V, *d_O;
    void* d_workspace;
    CHECK(cudaMalloc(&d_Q, q_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_K, kv_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_V, kv_n * sizeof(__half)));
    CHECK(cudaMalloc(&d_O, o_n * sizeof(__half)));
    CHECK(cudaMalloc(
        &d_workspace,
        persistentkv_workspace_bytes(B, Hq, num_splits)));
    CHECK(cudaMemset(d_Q, 0, q_n * sizeof(__half)));
    CHECK(cudaMemset(d_K, 0, kv_n * sizeof(__half)));
    CHECK(cudaMemset(d_V, 0, kv_n * sizeof(__half)));

    persistentkv_split_dispatch(
        d_Q, d_K, d_V, d_O, d_workspace,
        B, Hq, Hkv, N, d, num_splits);
    CHECK(cudaDeviceSynchronize());
    printf("Profile target complete: B=1 Hq=32 Hkv=8 N=8192 splits=16\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_workspace);
    return 0;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    srand(42);
    if (argc == 2 && strcmp(argv[1], "--profile") == 0)
        return run_profile_target();
    if (argc == 2 && strcmp(argv[1], "--split-sweep") == 0) {
        for (int splits : {1, 2, 4, 8, 16, 32, 64})
            bench_perf(1, 32, 8, 8192, 128, splits);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--target") == 0) {
        printf("PKV_FAST_EXP=%d PKV_TRANSPOSE_K=%d PKV_USE_WMMA=%d "
               "PKV_USE_WMMA_V=%d PKV_DOUBLE_BUFFER_V=%d\n",
               PKV_FAST_EXP, PKV_TRANSPOSE_K, PKV_USE_WMMA,
               PKV_USE_WMMA_V, PKV_DOUBLE_BUFFER_V);
        bench_perf(1, 32, 8, 8192, 128, 32);
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "--target-splits") == 0) {
        int splits = atoi(argv[2]);
        printf("PKV_FAST_EXP=%d PKV_TRANSPOSE_K=%d PKV_USE_WMMA=%d "
               "PKV_USE_WMMA_V=%d PKV_DOUBLE_BUFFER_V=%d\n",
               PKV_FAST_EXP, PKV_TRANSPOSE_K, PKV_USE_WMMA,
               PKV_USE_WMMA_V, PKV_DOUBLE_BUFFER_V);
        bench_perf(1, 32, 8, 8192, 128, splits);
        return 0;
    }
    if ((argc == 2 || argc == 3) &&
        strcmp(argv[1], "--target-long") == 0) {
        int splits = argc == 3 ? atoi(argv[2]) : 16;
        printf("PKV_FAST_EXP=%d PKV_TRANSPOSE_K=%d PKV_USE_WMMA=%d "
               "PKV_USE_WMMA_V=%d PKV_DOUBLE_BUFFER_V=%d\n",
               PKV_FAST_EXP, PKV_TRANSPOSE_K, PKV_USE_WMMA,
               PKV_USE_WMMA_V, PKV_DOUBLE_BUFFER_V);
        bench_perf(1, 32, 8, 8192, 128, splits, false, 1000);
        return 0;
    }

    printf("================================================================\n");
    printf(" PersistentKV-Attention Smoke Test\n");
    printf(" RTX 3060 (SM 8.6), CUDA 12.1\n");
    printf(" PKV_D=%d PKV_TILE=%d PKV_THREADS=%d PKV_SMEM=%.1fKB\n",
           PKV_D, PKV_TILE, PKV_THREADS, PKV_SMEM_BYTES / 1024.0f);
    printf("================================================================\n\n");

    // ── Section 1: Correctness ─────────────────────────────────────────────
    printf("=== SECTION 1: Correctness (vs CPU FP32 reference) ===\n");
    bool all_pass = true;

    // Small shapes (fast CPU reference)
    all_pass &= test_correctness(1,  4,  1, 257, 128, 1, true);
    all_pass &= test_correctness(1,  8,  2, 257, 128, 4);
    all_pass &= test_correctness(1, 32,  8, 512, 128, 4);
    all_pass &= test_correctness(2, 32,  8, 259, 128, 3);
    all_pass &= test_correctness(1, 32,  4, 511, 128, 4);
    all_pass &= test_correctness(1,  8,  8, 256, 128, 2);
    all_pass &= test_paged_correctness(1, 8, 2, 257, 128, 16, 4);
    all_pass &= test_paged_correctness(2, 8, 2, 259, 128, 32, 3);

    printf("\nCorrectness: %s\n\n", all_pass ? "ALL PASSED" : "SOME FAILED");

    // ── Section 2: Performance sweep ──────────────────────────────────────
    printf("=== SECTION 2: Performance Sweep (latency + effective BW) ===\n");
    printf("Peak HBM bandwidth (RTX 3060): ~360 GB/s\n\n");

    printf("--- GQA: Hq=32, Hkv=8, G=4, d=128 ---\n");
    bench_perf(1, 32, 8,  4096, 128,  16, true);
    bench_perf(1, 32, 8,  8192, 128,  16, true);
    bench_perf(1, 32, 8, 16384, 128,  16, true);
    bench_perf(1, 32, 8, 32768, 128,  16, true);

    printf("\n--- MQA-like: Hq=8, Hkv=1, G=8, d=128 ---\n");
    bench_perf(1, 8, 1, 4096, 128, 128, true);
    bench_perf(1, 8, 1, 8192, 128, 128, true);

    printf("\n--- Batch sweep: Hq=32, Hkv=8, N=8192 ---\n");
    bench_perf(1, 32, 8, 8192, 128, 16);
    bench_perf(2, 32, 8, 8192, 128,  8);
    bench_perf(4, 32, 8, 8192, 128,  4);
    bench_perf(8, 32, 8, 8192, 128,  2);

    printf("\n--- G sweep: N=8192 ---\n");
    bench_perf(1, 4,  4, 8192, 128, 32);
    bench_perf(1, 8,  4, 8192, 128, 32);
    bench_perf(1, 16, 4, 8192, 128, 32);
    bench_perf(1, 32, 4, 8192, 128, 32);

    printf("\n================================================================\n");
    printf("Done. For full hardware profiling run:\n");
    printf("  ncu --kernel-name-base demangled "
           "--kernel-name 'regex:.*persistentkv_decode_kernel.*' \\\n");
    printf("      --launch-count 1 --metrics dram__bytes_read.sum,"
           "lts__t_sector_hit_rate.pct,\\\n");
    printf("      sm__warps_active.avg.pct_of_peak_sustained_active "
           "./smoke_test --profile\n");
    printf("================================================================\n");
    return all_pass ? 0 : 1;
}
