// micro_benchmark 03: Online softmax state register pressure vs occupancy
// Tests how many query-head states (m, l, acc[d]) we can keep in registers
// before register spills destroy occupancy.  RTX 3060: 65536 regs per SM.
// Compile: nvcc -O3 -arch=sm_86 -Xptxas -v 03_register_pressure.cu -o 03_register_pressure
// Run:     ./03_register_pressure
// Also run: cuobjdump --dump-sass 03_register_pressure | grep "register"

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

// Template on G (group size) so the compiler sees the register count statically
// d is fixed at 128.  Each query head needs: m(1) + l(1) + acc(d_per_thread) registers.
// Threads=128, d=128 -> each thread covers 1 element of output.
// So per thread: G*(2 + 1) + misc = G*3 registers for the state alone.

template<int G>
__global__ void softmax_state_kernel(
    const __half* __restrict__ Q,   // [G, 128]
    const __half* __restrict__ K,   // [N, 128]
    const __half* __restrict__ V,   // [N, 128]
    float* __restrict__ O,          // [G, 128]
    int N, int d)
{
    int tid = threadIdx.x;  // tid < 128

    // Per-query-head online softmax state IN REGISTERS
    float m_arr[G];
    float l_arr[G];
    float acc_arr[G];     // each thread holds 1 output dim per head

    #pragma unroll
    for (int g = 0; g < G; g++) {
        m_arr[g] = -1e20f;
        l_arr[g] = 0.f;
        acc_arr[g] = 0.f;
    }

    // Cache query vectors in registers (each thread: 1 element per query head)
    float q_reg[G];
    if (tid < d) {
        #pragma unroll
        for (int g = 0; g < G; g++) {
            q_reg[g] = (tid < d) ? __half2float(Q[g * d + tid]) : 0.f;
        }
    }

    const int TILE = 32;
    extern __shared__ float smem[];   // [2 * TILE * d]
    float* k_smem = smem;
    float* v_smem = smem + TILE * d;

    for (int t0 = 0; t0 < N; t0 += TILE) {
        int t_end = min(t0 + TILE, N);
        int tile_sz = t_end - t0;

        // Load K tile
        __syncthreads();
        for (int elem = tid; elem < tile_sz * d; elem += blockDim.x) {
            k_smem[elem] = __half2float(K[(t0 + elem / d) * d + elem % d]);
            v_smem[elem] = __half2float(V[(t0 + elem / d) * d + elem % d]);
        }
        __syncthreads();

        // For each K token in tile, compute dot with each query head
        for (int t = 0; t < tile_sz; t++) {
            // Dot product for each group head (thread contributes its dimension)
            #pragma unroll
            for (int g = 0; g < G; g++) {
                float s = (tid < d) ? (q_reg[g] * k_smem[t * d + tid] / 11.31f) : 0.f;
                // Warp-level reduction for the dot product
                for (int offset = 16; offset > 0; offset >>= 1)
                    s += __shfl_down_sync(0xffffffff, s, offset);
                // Thread 0 of each warp does the softmax update
                if (tid % 32 == 0) {
                    float new_m = max(m_arr[g], s);
                    l_arr[g] = expf(m_arr[g] - new_m) * l_arr[g] + expf(s - new_m);
                    m_arr[g] = new_m;
                }
            }
            // Value accumulation (simplified per-thread)
            #pragma unroll
            for (int g = 0; g < G; g++) {
                if (tid < d) {
                    acc_arr[g] += v_smem[t * d + tid];  // simplified (no weight)
                }
            }
        }
    }

    // Normalize and write
    #pragma unroll
    for (int g = 0; g < G; g++) {
        if (tid < d) {
            O[g * d + tid] = (l_arr[g] > 0.f) ? acc_arr[g] / l_arr[g] : 0.f;
        }
    }
}

// Specialize for G=1,2,4,8 to compare register usage
template __global__ void softmax_state_kernel<1>(const __half*, const __half*, const __half*, float*, int, int);
template __global__ void softmax_state_kernel<2>(const __half*, const __half*, const __half*, float*, int, int);
template __global__ void softmax_state_kernel<4>(const __half*, const __half*, const __half*, float*, int, int);
template __global__ void softmax_state_kernel<8>(const __half*, const __half*, const __half*, float*, int, int);

int main() {
    printf("=== Register Pressure vs Occupancy (G=1,2,4,8) ===\n");
    printf("RTX 3060: 28 SMs, 65536 regs/SM, 1536 threads/SM max\n\n");

    const int d = 128;
    const int N = 8192;
    const int THREADS = 128;
    const int ITERS = 100;

    // allocate for max G=8
    __half *d_Q, *d_K, *d_V;
    float *d_O;
    CHECK(cudaMalloc(&d_Q, 8 * d * sizeof(__half)));
    CHECK(cudaMalloc(&d_K, (size_t)N * d * sizeof(__half)));
    CHECK(cudaMalloc(&d_V, (size_t)N * d * sizeof(__half)));
    CHECK(cudaMalloc(&d_O, 8 * d * sizeof(float)));
    CHECK(cudaMemset(d_Q, 0, 8 * d * sizeof(__half)));
    CHECK(cudaMemset(d_K, 0, (size_t)N * d * sizeof(__half)));
    CHECK(cudaMemset(d_V, 0, (size_t)N * d * sizeof(__half)));

    size_t smem = 2 * 32 * d * sizeof(float);  // 32KB

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    auto bench = [&](auto fn, int G_val) {
        // warm up
        fn<<<1, THREADS, smem>>>(d_Q, d_K, d_V, d_O, N, d);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            fn<<<1, THREADS, smem>>>(d_Q, d_K, d_V, d_O, N, d);
        CHECK(cudaEventRecord(stop));
        float ms;
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        double t = ms / ITERS;

        // Query occupancy
        int max_active;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active,
            (void*)fn, THREADS, smem);
        double occ = (double)(max_active * THREADS) / 1536.0 * 100.0;

        printf("G=%d: %.3f ms  | max_active_blocks/SM=%d  | occupancy=%.0f%%\n",
               G_val, t, max_active, occ);
    };

    bench(softmax_state_kernel<1>, 1);
    bench(softmax_state_kernel<2>, 2);
    bench(softmax_state_kernel<4>, 4);
    bench(softmax_state_kernel<8>, 8);

    printf("\nNote: compile with -Xptxas -v to see register counts per kernel variant.\n");
    printf("Register spills -> local memory -> HBM traffic: bad for bandwidth-bound kernels.\n");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
