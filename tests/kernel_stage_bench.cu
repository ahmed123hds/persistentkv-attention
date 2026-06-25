// Focused bottleneck and paged-layout benchmarks for PersistentKV.
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++17 -lineinfo -I. \
//       tests/kernel_stage_bench.cu -o build/kernel_stage_bench

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../kernels/persistentkv_attention.cuh"

#define CHECK(x) do { \
    cudaError_t error__ = (x); \
    if (error__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(error__), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

static constexpr int B = 1;
static constexpr int HQ = 32;
static constexpr int HKV = 8;
static constexpr int G = 4;
static constexpr int D = 128;
static constexpr int N = 8192;
static constexpr int SPLITS = 16;

__global__ void score_only_wmma_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    float* __restrict__ sinks,
    int context)
{
    using namespace nvcuda;
    int group_split = blockIdx.x;
    int split = group_split % SPLITS;
    int kvh = group_split / SPLITS;
    int tid = threadIdx.x;
    int warp = tid >> 5;

    __shared__ __half smem_k[PKV_TILE * D];
    __shared__ __half smem_q[16 * D];
    __shared__ float smem_scores[16 * PKV_TILE];

    for (int elem = tid; elem < 16 * D; elem += blockDim.x) {
        int row = elem / D;
        int dim = elem - row * D;
        smem_q[elem] = row < G
            ? Q[(kvh * G + row) * D + dim]
            : __float2half(0.0f);
    }
    __syncthreads();

    int total_tiles = (context + PKV_TILE - 1) / PKV_TILE;
    int tile_begin = total_tiles * split / SPLITS;
    int tile_end = total_tiles * (split + 1) / SPLITS;
    float checksum = 0.0f;
    const __half* Kh = K + (size_t)kvh * context * D;

    for (int tile = tile_begin; tile < tile_end; ++tile) {
        int t0 = tile * PKV_TILE;
        int tile_size = min(PKV_TILE, context - t0);
        int chunks = tile_size * D / 8;
        for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
            int elem = chunk * 8;
            pkv_cp_async16(
                &smem_k[elem],
                &Kh[(size_t)t0 * D + elem]);
        }
        pkv_cp_commit();
        pkv_cp_wait<0>();
        __syncthreads();

        if (warp < 2) {
            wmma::fragment<
                wmma::accumulator, 16, 16, 16, float> score_frag;
            wmma::fill_fragment(score_frag, 0.0f);
            #pragma unroll
            for (int k0 = 0; k0 < D; k0 += 16) {
                wmma::fragment<
                    wmma::matrix_a, 16, 16, 16,
                    __half, wmma::row_major> q_frag;
                wmma::fragment<
                    wmma::matrix_b, 16, 16, 16,
                    __half, wmma::col_major> k_frag;
                wmma::load_matrix_sync(q_frag, &smem_q[k0], D);
                wmma::load_matrix_sync(
                    k_frag, &smem_k[warp * 16 * D + k0], D);
                wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
            }
            wmma::store_matrix_sync(
                &smem_scores[warp * 16], score_frag,
                PKV_TILE, wmma::mem_row_major);
        }
        __syncthreads();
        if (tid < G * PKV_TILE)
            checksum += smem_scores[tid];
        __syncthreads();
    }
    if (tid < G * PKV_TILE)
        sinks[(size_t)blockIdx.x * G * PKV_TILE + tid] = checksum;
}

__global__ void softmax_only_kernel(
    const float* __restrict__ scores,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int context)
{
    int qh_split = blockIdx.x;
    int split = qh_split % SPLITS;
    int qh = qh_split / SPLITS;
    int lane = threadIdx.x;
    int total_tiles = (context + PKV_TILE - 1) / PKV_TILE;
    int tile_begin = total_tiles * split / SPLITS;
    int tile_end = total_tiles * (split + 1) / SPLITS;
    float m = -FLT_MAX;
    float l = 0.0f;

    for (int tile = tile_begin; tile < tile_end; ++tile) {
        int token = tile * PKV_TILE + lane;
        float score = token < context
            ? scores[(size_t)qh * context + token]
            : -FLT_MAX;
        float tile_m = pkv_warp_max(score);
        float new_m = fmaxf(m, tile_m);
        float correction = pkv_exp(m - new_m);
        float weight = token < context ? pkv_exp(score - new_m) : 0.0f;
        l = correction * l + pkv_warp_sum(weight);
        m = new_m;
    }

    if (lane == 0) {
        partial_m[qh_split] = m;
        partial_l[qh_split] = l;
    }
}

__global__ void value_only_kernel(
    const __half* __restrict__ weights,
    const __half* __restrict__ V,
    float* __restrict__ partial_acc,
    int context)
{
    int group_split = blockIdx.x;
    int split = group_split % SPLITS;
    int kvh = group_split / SPLITS;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int qh = kvh * G + warp;

    __shared__ __half smem_v[2][PKV_TILE * D];
    float acc[PKV_D_PER_LANE] = {};
    int total_tiles = (context + PKV_TILE - 1) / PKV_TILE;
    int tile_begin = total_tiles * split / SPLITS;
    int tile_end = total_tiles * (split + 1) / SPLITS;
    int local_tiles = tile_end - tile_begin;
    const __half* Vh = V + (size_t)kvh * context * D;

#define PREFETCH_V(tile_, slot_) do { \
    int t0__ = (tile_) * PKV_TILE; \
    int size__ = min(PKV_TILE, context - t0__); \
    int chunks__ = size__ * D / 8; \
    for (int chunk__ = tid; chunk__ < chunks__; chunk__ += blockDim.x) { \
        int elem__ = chunk__ * 8; \
        pkv_cp_async16(&smem_v[(slot_)][elem__], \
                       &Vh[(size_t)t0__ * D + elem__]); \
    } \
    pkv_cp_commit(); \
} while (0)

    if (local_tiles > 0)
        PREFETCH_V(tile_begin, 0);

    for (int local = 0; local < local_tiles; ++local) {
        int cur = local & 1;
        int next = 1 - cur;
        int tile = tile_begin + local;
        if (local + 1 < local_tiles)
            PREFETCH_V(tile + 1, next);
        if (local + 1 < local_tiles)
            pkv_cp_wait<1>();
        else
            pkv_cp_wait<0>();
        __syncthreads();

        int t0 = tile * PKV_TILE;
        int tile_size = min(PKV_TILE, context - t0);
        float weight = lane < tile_size
            ? __half2float(weights[(size_t)qh * context + t0 + lane])
            : 0.0f;
        #pragma unroll
        for (int j = 0; j < PKV_D_PER_LANE; ++j) {
            int dim = lane + j * 32;
            float tile_acc = 0.0f;
            #pragma unroll
            for (int token = 0; token < PKV_TILE; ++token) {
                float token_weight =
                    __shfl_sync(0xffffffff, weight, token);
                if (token < tile_size) {
                    tile_acc = fmaf(
                        token_weight,
                        __half2float(smem_v[cur][token * D + dim]),
                        tile_acc);
                }
            }
            acc[j] += tile_acc;
        }
        __syncthreads();
    }
#undef PREFETCH_V

    size_t state = (size_t)qh * SPLITS + split;
    #pragma unroll
    for (int j = 0; j < PKV_D_PER_LANE; ++j) {
        int dim = lane + j * 32;
        partial_acc[state * D + dim] = acc[j];
    }
}

__global__ void value_only_wmma_kernel(
    const __half* __restrict__ weights,
    const __half* __restrict__ V,
    float* __restrict__ partial_acc,
    int context)
{
    using namespace nvcuda;
    int group_split = blockIdx.x;
    int split = group_split % SPLITS;
    int kvh = group_split / SPLITS;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    __shared__ __half smem_v[2][PKV_TILE * D];
    __shared__ __half smem_weights[16 * PKV_TILE];
    __shared__ float smem_acc[16 * D];
    int total_tiles = (context + PKV_TILE - 1) / PKV_TILE;
    int tile_begin = total_tiles * split / SPLITS;
    int tile_end = total_tiles * (split + 1) / SPLITS;
    int local_tiles = tile_end - tile_begin;
    const __half* Vh = V + (size_t)kvh * context * D;
    float acc[PKV_D_PER_LANE] = {};

#define PREFETCH_WMMA_V(tile_, slot_) do { \
    int t0__ = (tile_) * PKV_TILE; \
    int size__ = min(PKV_TILE, context - t0__); \
    int chunks__ = size__ * D / 8; \
    for (int chunk__ = tid; chunk__ < chunks__; chunk__ += blockDim.x) { \
        int elem__ = chunk__ * 8; \
        pkv_cp_async16(&smem_v[(slot_)][elem__], \
                       &Vh[(size_t)t0__ * D + elem__]); \
    } \
    pkv_cp_commit(); \
} while (0)

    if (local_tiles > 0)
        PREFETCH_WMMA_V(tile_begin, 0);

    for (int local = 0; local < local_tiles; ++local) {
        int cur = local & 1;
        int next = 1 - cur;
        int tile = tile_begin + local;
        if (local + 1 < local_tiles)
            PREFETCH_WMMA_V(tile + 1, next);
        if (local + 1 < local_tiles)
            pkv_cp_wait<1>();
        else
            pkv_cp_wait<0>();
        __syncthreads();

        int t0 = tile * PKV_TILE;
        int tile_size = min(PKV_TILE, context - t0);
        for (int elem = tid; elem < 16 * PKV_TILE;
             elem += blockDim.x) {
            int row = elem / PKV_TILE;
            int token = elem - row * PKV_TILE;
            smem_weights[elem] =
                row < G && token < tile_size
                ? weights[(size_t)(kvh * G + row) * context + t0 + token]
                : __float2half(0.0f);
        }
        for (int elem = tid + tile_size * D;
             elem < PKV_TILE * D; elem += blockDim.x) {
            smem_v[cur][elem] = __float2half(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int output_block = 0; output_block < 2; ++output_block) {
            int n0 = (warp * 2 + output_block) * 16;
            wmma::fragment<
                wmma::accumulator, 16, 16, 16, float> value_frag;
            wmma::fill_fragment(value_frag, 0.0f);
            #pragma unroll
            for (int k0 = 0; k0 < PKV_TILE; k0 += 16) {
                wmma::fragment<
                    wmma::matrix_a, 16, 16, 16,
                    __half, wmma::row_major> weight_frag;
                wmma::fragment<
                    wmma::matrix_b, 16, 16, 16,
                    __half, wmma::row_major> value_frag_input;
                wmma::load_matrix_sync(
                    weight_frag, &smem_weights[k0], PKV_TILE);
                wmma::load_matrix_sync(
                    value_frag_input, &smem_v[cur][k0 * D + n0], D);
                wmma::mma_sync(
                    value_frag, weight_frag, value_frag_input, value_frag);
            }
            wmma::store_matrix_sync(
                &smem_acc[n0], value_frag, D, wmma::mem_row_major);
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < PKV_D_PER_LANE; ++j) {
            int dim = lane + j * 32;
            acc[j] += smem_acc[warp * D + dim];
        }
        __syncthreads();
    }
#undef PREFETCH_WMMA_V

    size_t state = (size_t)(kvh * G + warp) * SPLITS + split;
    #pragma unroll
    for (int j = 0; j < PKV_D_PER_LANE; ++j) {
        int dim = lane + j * 32;
        partial_acc[state * D + dim] = acc[j];
    }
}

template<typename Accessor>
__global__ void accessor_only_kernel(
    Accessor kv,
    unsigned int* __restrict__ sinks,
    int context,
    int num_splits)
{
    int group_split = blockIdx.x;
    int split = group_split % num_splits;
    int kvh = group_split / num_splits;
    int tid = threadIdx.x;
    int total_tiles = (context + PKV_TILE - 1) / PKV_TILE;
    int tile_begin = total_tiles * split / num_splits;
    int tile_end = total_tiles * (split + 1) / num_splits;
    unsigned int checksum = 0;

    for (int tile = tile_begin; tile < tile_end; ++tile) {
        int t0 = tile * PKV_TILE;
        int tile_size = min(PKV_TILE, context - t0);
        int chunks = tile_size * D / 8;
        for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
            int elem = chunk * 8;
            int token = t0 + elem / D;
            int dim = elem % D;
            uint4 k = *reinterpret_cast<const uint4*>(
                kv.k_ptr(0, kvh, token, dim));
            uint4 v = *reinterpret_cast<const uint4*>(
                kv.v_ptr(0, kvh, token, dim));
            checksum ^= k.x ^ k.y ^ k.z ^ k.w;
            checksum ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    sinks[(size_t)blockIdx.x * blockDim.x + tid] = checksum;
}

template<typename Launch>
float time_launch(Launch launch, int warmup=10, int iterations=1000) {
    for (int i = 0; i < warmup; ++i)
        launch();
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i)
        launch();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed;
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return elapsed / iterations;
}

void run_stage_breakdown() {
    size_t q_count = (size_t)HQ * D;
    size_t kv_count = (size_t)HKV * N * D;
    size_t score_count = (size_t)HQ * N;
    size_t states = (size_t)HQ * SPLITS;
    __half *q, *k, *v, *o, *weights;
    float *scores, *partial_m, *partial_l, *partial_acc, *score_sinks;
    void* workspace;

    CHECK(cudaMalloc(&q, q_count * sizeof(__half)));
    CHECK(cudaMalloc(&k, kv_count * sizeof(__half)));
    CHECK(cudaMalloc(&v, kv_count * sizeof(__half)));
    CHECK(cudaMalloc(&o, q_count * sizeof(__half)));
    CHECK(cudaMalloc(&weights, score_count * sizeof(__half)));
    CHECK(cudaMalloc(&scores, score_count * sizeof(float)));
    CHECK(cudaMalloc(&partial_m, states * sizeof(float)));
    CHECK(cudaMalloc(&partial_l, states * sizeof(float)));
    CHECK(cudaMalloc(&partial_acc, states * D * sizeof(float)));
    CHECK(cudaMalloc(
        &score_sinks,
        (size_t)HKV * SPLITS * G * PKV_TILE * sizeof(float)));
    CHECK(cudaMalloc(
        &workspace, persistentkv_workspace_bytes(B, HQ, SPLITS)));
    CHECK(cudaMemset(q, 0, q_count * sizeof(__half)));
    CHECK(cudaMemset(k, 0, kv_count * sizeof(__half)));
    CHECK(cudaMemset(v, 0, kv_count * sizeof(__half)));
    CHECK(cudaMemset(weights, 0, score_count * sizeof(__half)));
    CHECK(cudaMemset(scores, 0, score_count * sizeof(float)));
    CHECK(cudaMemset(partial_m, 0, states * sizeof(float)));
    CHECK(cudaMemset(partial_l, 0, states * sizeof(float)));
    CHECK(cudaMemset(partial_acc, 0, states * D * sizeof(float)));

    float score_ms = time_launch([&] {
        score_only_wmma_kernel<<<HKV * SPLITS, PKV_THREADS>>>(
            q, k, score_sinks, N);
    });
    float softmax_ms = time_launch([&] {
        softmax_only_kernel<<<HQ * SPLITS, 32>>>(
            scores, partial_m, partial_l, N);
    });
    float value_ms = time_launch([&] {
        value_only_kernel<<<HKV * SPLITS, PKV_THREADS>>>(
            weights, v, partial_acc, N);
    });
    float value_wmma_ms = time_launch([&] {
        value_only_wmma_kernel<<<HKV * SPLITS, PKV_THREADS>>>(
            weights, v, partial_acc, N);
    });
    float merge_ms = time_launch([&] {
        persistentkv_merge_splits_kernel<<<HQ, PKV_THREADS>>>(
            partial_m, partial_l, partial_acc, o, nullptr, HQ, SPLITS);
    });
    float full_ms = time_launch([&] {
        persistentkv_split_dispatch(
            q, k, v, o, workspace,
            B, HQ, HKV, N, D, SPLITS);
    });

    printf("PersistentKV stage breakdown\n");
    printf("Shape: B=1 Hq=32 Hkv=8 G=4 N=8192 d=128 splits=16\n\n");
    printf("| Stage | Time (ms) | Fraction of full |\n");
    printf("|---|---:|---:|\n");
    printf("| WMMA score path | %.4f | %.1f%% |\n",
           score_ms, 100.0f * score_ms / full_ms);
    printf("| Softmax only | %.4f | %.1f%% |\n",
           softmax_ms, 100.0f * softmax_ms / full_ms);
    printf("| V accumulation only | %.4f | %.1f%% |\n",
           value_ms, 100.0f * value_ms / full_ms);
    printf("| WMMA V accumulation only | %.4f | %.1f%% |\n",
           value_wmma_ms, 100.0f * value_wmma_ms / full_ms);
    printf("| Split merge only | %.4f | %.1f%% |\n",
           merge_ms, 100.0f * merge_ms / full_ms);
    printf("| Full PersistentKV | %.4f | 100.0%% |\n", full_ms);
    printf("\nIsolated stages do not sum to the full kernel because the full path "
           "overlaps copies and compute.\n");

    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
    cudaFree(weights); cudaFree(scores); cudaFree(partial_m);
    cudaFree(partial_l); cudaFree(partial_acc); cudaFree(score_sinks);
    cudaFree(workspace);
}

std::vector<int> make_block_table(
    int logical_pages,
    int physical_pages,
    bool random_order,
    bool canonical_fragmented = false)
{
    if (canonical_fragmented) {
        std::vector<int> table(logical_pages);
        for (int page = 0; page < logical_pages; ++page)
            table[page] = (405 * page + 17) % physical_pages;
        return table;
    }

    std::vector<int> available(physical_pages);
    std::iota(available.begin(), available.end(), 0);
    if (random_order) {
        std::mt19937 generator(1234);
        std::shuffle(available.begin(), available.end(), generator);
        available.resize(logical_pages);
        return available;
    }

    std::vector<int> table(logical_pages);
    for (int page = 0; page < logical_pages; ++page) {
        table[page] = logical_pages == 1
            ? 0
            : (int)((long long)page * (physical_pages - 1) /
                    (logical_pages - 1));
    }
    return table;
}

struct LayoutTiming {
    float contiguous_ms;
    float paged_ms;
    float contiguous_accessor_ms;
    float paged_accessor_ms;
};

float median_of_three(float a, float b, float c) {
    float values[3] = {a, b, c};
    std::sort(values, values + 3);
    return values[1];
}

LayoutTiming bench_paged_layout(
    const __half* q,
    const __half* k,
    const __half* v,
    __half* o,
    void* workspace,
    int context,
    int page_size,
    float fragmentation,
    bool random_order,
    bool canonical_fragmented = false)
{
    int logical_pages = (context + page_size - 1) / page_size;
    int physical_pages =
        (int)ceilf(logical_pages / (1.0f - fragmentation));
    size_t page_elements =
        (size_t)physical_pages * HKV * page_size * D;
    __half *k_pages, *v_pages;
    int* block_table;
    unsigned int* sinks;
    CHECK(cudaMalloc(&k_pages, page_elements * sizeof(__half)));
    CHECK(cudaMalloc(&v_pages, page_elements * sizeof(__half)));
    CHECK(cudaMalloc(&block_table, logical_pages * sizeof(int)));
    CHECK(cudaMalloc(
        &sinks,
        (size_t)HKV * SPLITS * PKV_THREADS * sizeof(unsigned int)));
    CHECK(cudaMemset(k_pages, 0, page_elements * sizeof(__half)));
    CHECK(cudaMemset(v_pages, 0, page_elements * sizeof(__half)));

    std::vector<int> table = make_block_table(
        logical_pages, physical_pages, random_order, canonical_fragmented);
    CHECK(cudaMemcpy(
        block_table, table.data(), logical_pages * sizeof(int),
        cudaMemcpyHostToDevice));

    auto launch_contiguous = [&] {
        persistentkv_split_dispatch(
            q, k, v, o, workspace,
            B, HQ, HKV, context, D, SPLITS);
    };
    auto launch_paged = [&] {
        persistentkv_paged_dispatch(
            q, k_pages, v_pages, block_table, o, workspace,
            B, HQ, HKV, context, D, page_size, logical_pages, SPLITS);
    };
    PkvContiguousAccessor contiguous_accessor{k, v, HKV, context};
    auto launch_contiguous_accessor = [&] {
        accessor_only_kernel<<<HKV * SPLITS, PKV_THREADS>>>(
            contiguous_accessor, sinks, context, SPLITS);
    };
    PkvPagedAccessor paged_accessor{
        k_pages, v_pages, block_table, nullptr,
        logical_pages, HKV, page_size};
    auto launch_paged_accessor = [&] {
        accessor_only_kernel<<<HKV * SPLITS, PKV_THREADS>>>(
            paged_accessor, sinks, context, SPLITS);
    };

    float contiguous_samples[3];
    float paged_samples[3];
    float contiguous_accessor_samples[3];
    float paged_accessor_samples[3];
    for (int repetition = 0; repetition < 3; ++repetition) {
        if ((repetition & 1) == 0) {
            contiguous_samples[repetition] =
                time_launch(launch_contiguous, 3, 150);
            paged_samples[repetition] =
                time_launch(launch_paged, 3, 150);
            contiguous_accessor_samples[repetition] =
                time_launch(launch_contiguous_accessor, 3, 150);
            paged_accessor_samples[repetition] =
                time_launch(launch_paged_accessor, 3, 150);
        } else {
            paged_samples[repetition] =
                time_launch(launch_paged, 3, 150);
            contiguous_samples[repetition] =
                time_launch(launch_contiguous, 3, 150);
            paged_accessor_samples[repetition] =
                time_launch(launch_paged_accessor, 3, 150);
            contiguous_accessor_samples[repetition] =
                time_launch(launch_contiguous_accessor, 3, 150);
        }
    }

    cudaFree(k_pages);
    cudaFree(v_pages);
    cudaFree(block_table);
    cudaFree(sinks);
    return {
        median_of_three(
            contiguous_samples[0], contiguous_samples[1],
            contiguous_samples[2]),
        median_of_three(
            paged_samples[0], paged_samples[1], paged_samples[2]),
        median_of_three(
            contiguous_accessor_samples[0],
            contiguous_accessor_samples[1],
            contiguous_accessor_samples[2]),
        median_of_three(
            paged_accessor_samples[0], paged_accessor_samples[1],
            paged_accessor_samples[2])
    };
}

void run_paged_sweep(int context) {
    size_t q_count = (size_t)HQ * D;
    size_t kv_count = (size_t)HKV * context * D;
    __half *q, *k, *v, *o;
    void* workspace;
    CHECK(cudaMalloc(&q, q_count * sizeof(__half)));
    CHECK(cudaMalloc(&k, kv_count * sizeof(__half)));
    CHECK(cudaMalloc(&v, kv_count * sizeof(__half)));
    CHECK(cudaMalloc(&o, q_count * sizeof(__half)));
    CHECK(cudaMalloc(
        &workspace, persistentkv_workspace_bytes(B, HQ, SPLITS)));
    CHECK(cudaMemset(q, 0, q_count * sizeof(__half)));
    CHECK(cudaMemset(k, 0, kv_count * sizeof(__half)));
    CHECK(cudaMemset(v, 0, kv_count * sizeof(__half)));

    printf("PersistentKV paged-layout sweep\n");
    printf(
        "Shape: B=1 Hq=32 Hkv=8 G=4 N=%d d=128 splits=16\n", context);
    printf(
        "Each row is the median of three paired, alternating-order runs.\n\n");
    printf("| Page | Frag. | Order | Contig. ms | Paged ms | "
           "Full P/C | Accessor P/C |\n");
    printf("|---:|---:|---|---:|---:|---:|---:|\n");
    for (int page_size : {16, 32, 64, 128}) {
        for (float fragmentation : {0.0f, 0.25f, 0.50f}) {
            for (bool random_order : {false, true}) {
                LayoutTiming timing = bench_paged_layout(
                    q, k, v, o, workspace, context, page_size,
                    fragmentation, random_order);
                printf(
                       "| %d | %.0f%% | %s | %.4f | %.4f | %.3fx | "
                       "%.3fx |\n",
                       page_size, fragmentation * 100.0f,
                       random_order ? "random" : "ordered",
                       timing.contiguous_ms, timing.paged_ms,
                       timing.paged_ms / timing.contiguous_ms,
                       timing.paged_accessor_ms /
                           timing.contiguous_accessor_ms);
            }
        }
    }

    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
    cudaFree(workspace);
}

void run_canonical_paged(int page_size) {
    printf("PersistentKV canonical native-paged benchmark\n");
    printf("Shape: B=1 Hq=32 Hkv=8 G=4 d=128, page=%d, 50%% holes\n",
           page_size);
    printf("Table: physical_page=(405*logical_page+17) mod physical_pages\n\n");
    printf("| N | Contiguous ms | Native paged ms | Paged overhead |\n");
    printf("|---:|---:|---:|---:|\n");

    for (int context : {8192, 32768, 65536}) {
        size_t q_count = (size_t)HQ * D;
        size_t kv_count = (size_t)HKV * context * D;
        __half *q, *k, *v, *o;
        void* workspace;
        CHECK(cudaMalloc(&q, q_count * sizeof(__half)));
        CHECK(cudaMalloc(&k, kv_count * sizeof(__half)));
        CHECK(cudaMalloc(&v, kv_count * sizeof(__half)));
        CHECK(cudaMalloc(&o, q_count * sizeof(__half)));
        CHECK(cudaMalloc(
            &workspace, persistentkv_workspace_bytes(B, HQ, SPLITS)));
        CHECK(cudaMemset(q, 0, q_count * sizeof(__half)));
        CHECK(cudaMemset(k, 0, kv_count * sizeof(__half)));
        CHECK(cudaMemset(v, 0, kv_count * sizeof(__half)));

        LayoutTiming timing = bench_paged_layout(
            q, k, v, o, workspace, context, page_size, 0.50f, true, true);
        printf("| %d | %.4f | %.4f | %.3fx |\n",
               context, timing.contiguous_ms, timing.paged_ms,
               timing.paged_ms / timing.contiguous_ms);

        cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
        cudaFree(workspace);
    }
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(
            stderr,
            "usage: %s --stages | --paged | --paged-long | "
            "--canonical-paged [page-size]\n", argv[0]);
        return 2;
    }
    std::string mode = argv[1];
    if (mode == "--stages") {
        run_stage_breakdown();
        return 0;
    }
    if (mode == "--paged") {
        run_paged_sweep(8192);
        return 0;
    }
    if (mode == "--paged-long") {
        run_paged_sweep(32768);
        return 0;
    }
    if (mode == "--canonical-paged") {
        int page_size = argc == 3 ? atoi(argv[2]) : 16;
        if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
            fprintf(stderr, "page-size must be a positive power of two\n");
            return 2;
        }
        run_canonical_paged(page_size);
        return 0;
    }
    fprintf(stderr, "unknown mode: %s\n", argv[1]);
    return 2;
}
