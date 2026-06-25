// =============================================================================
// PersistentKV-Attention: exact GQA/MQA decode attention for Ampere (SM 8.x)
// =============================================================================
// Design:
//   1. A CTA is assigned to one KV head and one sequence split.
//   2. Four warps process up to four grouped query heads concurrently.
//   3. Each warp computes 32 token scores concurrently; each lane owns one
//      token score and four output dimensions.
//   4. K,V tiles are loaded once per KV group with a cp.async pipeline.
//   5. Optional sequence splitting increases occupancy for small B*Hkv grids;
//      a second kernel merges partial online-softmax states exactly.
//   6. Contiguous and paged KV layouts share the same compute kernel.
// =============================================================================

#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mma.h>

#define PKV_D        128
#define PKV_MAX_G    8
#define PKV_TILE     32
#define PKV_THREADS  128
#define PKV_WARPS    (PKV_THREADS / 32)
#define PKV_D_PER_LANE (PKV_D / 32)
#define PKV_KT_STRIDE (PKV_TILE + 1)
#ifndef PKV_FAST_EXP
#define PKV_FAST_EXP 1
#endif
#ifndef PKV_TRANSPOSE_K
#define PKV_TRANSPOSE_K 1
#endif
#ifndef PKV_USE_WMMA
#define PKV_USE_WMMA 1
#endif
#ifndef PKV_DOUBLE_BUFFER_V
#define PKV_DOUBLE_BUFFER_V 1
#endif
#ifndef PKV_USE_WMMA_V
#define PKV_USE_WMMA_V 1
#endif
#if PKV_USE_WMMA
#define PKV_SMEM_BYTES \
    (((2 + PKV_DOUBLE_BUFFER_V) * PKV_TILE * PKV_D + 16 * PKV_D) * \
     (int)sizeof(__half) + \
     16 * PKV_TILE * (int)sizeof(float) + \
     (PKV_USE_WMMA_V ? \
      (16 * PKV_TILE * (int)sizeof(__half) + \
       16 * PKV_D * (int)sizeof(float)) : 0))
#elif PKV_TRANSPOSE_K
#define PKV_SMEM_BYTES \
    (((2 + PKV_DOUBLE_BUFFER_V) * PKV_TILE * PKV_D + \
      PKV_D * PKV_KT_STRIDE + \
      PKV_MAX_G * PKV_D) * (int)sizeof(__half))
#else
#define PKV_SMEM_BYTES \
    ((3 * PKV_TILE * PKV_D + PKV_MAX_G * PKV_D) * (int)sizeof(__half))
#endif

struct PkvWorkItem {
    int row;
    int kvh;
    int split;
    int tile_begin;
    int tile_end;
};

inline bool pkv_is_supported_g(int G) {
    return G == 1 || G == 2 || G == 4 || G == 8;
}

#define PKV_CHECK(x) do { \
    cudaError_t pkv_err__ = (x); \
    if (pkv_err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(pkv_err__), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__device__ __forceinline__
void pkv_cp_async16(void* __restrict__ dst, const void* __restrict__ src) {
    unsigned int dst_smem =
        static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_smem), "l"((const char*)src) : "memory");
}

__device__ __forceinline__ void pkv_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N_PENDING>
__device__ __forceinline__ void pkv_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_PENDING) : "memory");
}

__device__ __forceinline__ float pkv_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

__device__ __forceinline__ float pkv_warp_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(
            value, __shfl_xor_sync(0xffffffff, value, offset));
    return value;
}

__device__ __forceinline__ float pkv_exp(float value) {
#if PKV_FAST_EXP
    return __expf(value);
#else
    return expf(value);
#endif
}

struct PkvContiguousAccessor {
    const __half* K;
    const __half* V;
    int Hkv;
    int N;

    __device__ __forceinline__
    const __half* k_ptr(int b, int kvh, int token, int dim) const {
        size_t index =
            ((size_t)b * Hkv * N + (size_t)kvh * N + token) * PKV_D + dim;
        return K + index;
    }

    __device__ __forceinline__
    const __half* v_ptr(int b, int kvh, int token, int dim) const {
        size_t index =
            ((size_t)b * Hkv * N + (size_t)kvh * N + token) * PKV_D + dim;
        return V + index;
    }

    __device__ __forceinline__
    bool valid_token(int, int token) const {
        return token < N;
    }

    __device__ __forceinline__
    int sequence_length(int, int fallback) const {
        return fallback;
    }

    __device__ __forceinline__
    void prefetch_tile(
        int b,
        int kvh,
        int t0,
        int tile_size,
        int tid,
        __half* __restrict__ smem_k,
        __half* __restrict__ smem_v_slot) const
    {
        int chunks = (tile_size * PKV_D) / 8;
        for (int chunk = tid; chunk < chunks; chunk += PKV_THREADS) {
            int elem = chunk * 8;
            int token = t0 + elem / PKV_D;
            int dim = elem % PKV_D;
            pkv_cp_async16(&smem_k[elem], k_ptr(b, kvh, token, dim));
            pkv_cp_async16(&smem_v_slot[elem], v_ptr(b, kvh, token, dim));
        }
        pkv_cp_commit();
    }
};

// Paged layout: [physical_page, Hkv, page_size, D].
// block_table[b, logical_page] contains the physical page number.
struct PkvPagedAccessor {
    const __half* K;
    const __half* V;
    const int* block_table;
    const int* seq_lens;
    int max_pages;
    int Hkv;
    int page_size;

    __device__ __forceinline__
    size_t index(int b, int kvh, int token, int dim) const {
        int logical_page = token / page_size;
        int page_offset = token - logical_page * page_size;
        int physical_page =
            block_table[(size_t)b * max_pages + logical_page];
        return (((size_t)physical_page * Hkv + kvh) * page_size +
                page_offset) * PKV_D + dim;
    }

    __device__ __forceinline__
    const __half* k_ptr(int b, int kvh, int token, int dim) const {
        return K + index(b, kvh, token, dim);
    }

    __device__ __forceinline__
    const __half* v_ptr(int b, int kvh, int token, int dim) const {
        return V + index(b, kvh, token, dim);
    }

    __device__ __forceinline__
    bool valid_token(int b, int token) const {
        return seq_lens == nullptr || token < seq_lens[b];
    }

    __device__ __forceinline__
    int sequence_length(int b, int fallback) const {
        return seq_lens == nullptr ? fallback : seq_lens[b];
    }

    __device__ __forceinline__
    void prefetch_tile(
        int b,
        int kvh,
        int t0,
        int tile_size,
        int tid,
        __half* __restrict__ smem_k,
        __half* __restrict__ smem_v_slot) const
    {
        if (page_size == 16) {
            int logical0 = t0 >> 4;
            int pages = (tile_size + 15) >> 4;
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                if (p < pages) {
                    int page_tokens = min(16, tile_size - p * 16);
                    int physical_page = block_table[
                        (size_t)b * max_pages + logical0 + p];
                    const __half* k_base =
                        K + (((size_t)physical_page * Hkv + kvh) * 16) * PKV_D;
                    const __half* v_base =
                        V + (((size_t)physical_page * Hkv + kvh) * 16) * PKV_D;
                    int chunks = (page_tokens * PKV_D) / 8;
                    int dst_base = p * 16 * PKV_D;
                    for (int chunk = tid; chunk < chunks; chunk += PKV_THREADS) {
                        int elem = chunk * 8;
                        pkv_cp_async16(&smem_k[dst_base + elem], k_base + elem);
                        pkv_cp_async16(
                            &smem_v_slot[dst_base + elem], v_base + elem);
                    }
                }
            }
            pkv_cp_commit();
            return;
        }

        int chunks = (tile_size * PKV_D) / 8;
        for (int chunk = tid; chunk < chunks; chunk += PKV_THREADS) {
            int elem = chunk * 8;
            int token = t0 + elem / PKV_D;
            int dim = elem % PKV_D;
            pkv_cp_async16(&smem_k[elem], k_ptr(b, kvh, token, dim));
            pkv_cp_async16(&smem_v_slot[elem], v_ptr(b, kvh, token, dim));
        }
        pkv_cp_commit();
    }
};

template<int G, typename Accessor, bool WRITE_PARTIAL>
__global__ __launch_bounds__(PKV_THREADS, 2)
void persistentkv_decode_kernel(
    const __half* __restrict__ Q,
    Accessor kv,
    __half* __restrict__ O,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    const int* __restrict__ q_indices,
    const int* __restrict__ out_indices,
    const PkvWorkItem* __restrict__ work_items,
    const int* __restrict__ split_offsets,
    int* __restrict__ merge_counters,
    int Hq,
    int Hkv,
    int N,
    int num_splits,
    float scale)
{
    static_assert(G >= 1 && G <= PKV_MAX_G, "G out of range");
    static_assert(PKV_D == PKV_THREADS, "D and CTA size must match");
    static_assert(PKV_D_PER_LANE * 32 == PKV_D, "D must be warp divisible");

    int bid = blockIdx.y;
    int group_split = blockIdx.x;
    int split = group_split % num_splits;
    int kvh = group_split / num_splits;
    int explicit_tile_begin = -1;
    int explicit_tile_end = -1;
    if (work_items != nullptr) {
        PkvWorkItem item = work_items[blockIdx.x];
        bid = item.row;
        kvh = item.kvh;
        split = item.split;
        explicit_tile_begin = item.tile_begin;
        explicit_tile_end = item.tile_end;
    }
    const int q_bid = q_indices == nullptr ? bid : q_indices[bid];
    const int qh0 = kvh * G;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int HEADS_PER_WARP = (G + PKV_WARPS - 1) / PKV_WARPS;
    constexpr bool USE_WMMA = PKV_USE_WMMA && G == 4;
    constexpr bool USE_WMMA_V = USE_WMMA && PKV_USE_WMMA_V;
    constexpr int Q_ROWS = USE_WMMA ? 16 : G;
    constexpr int V_SLOTS = PKV_DOUBLE_BUFFER_V ? 2 : 1;

    __shared__ __half smem_k[PKV_TILE * PKV_D];
    __shared__ __half smem_v[V_SLOTS][PKV_TILE * PKV_D];
    __shared__ __half smem_kt[
        USE_WMMA ? 1 : PKV_D * PKV_KT_STRIDE];
    __shared__ __half smem_q[Q_ROWS * PKV_D];
    __shared__ float smem_scores[USE_WMMA ? 16 * PKV_TILE : 1];
    __shared__ __half smem_weights[
        USE_WMMA_V ? 16 * PKV_TILE : 1];
    __shared__ float smem_value_acc[
        USE_WMMA_V ? 16 * PKV_D : 1];

    const int effective_N = kv.sequence_length(bid, N);
    const int total_tiles = (effective_N + PKV_TILE - 1) / PKV_TILE;
    const int tile_begin = explicit_tile_begin >= 0
        ? explicit_tile_begin
        : (total_tiles * split) / num_splits;
    const int tile_end = explicit_tile_end >= 0
        ? explicit_tile_end
        : (total_tiles * (split + 1)) / num_splits;
    const int local_tiles = tile_end - tile_begin;

    if (local_tiles == 0) {
        if constexpr (WRITE_PARTIAL) {
            #pragma unroll
            for (int slot = 0; slot < HEADS_PER_WARP; ++slot) {
                int g = warp + slot * PKV_WARPS;
                if (g < G) {
                    int qh = qh0 + g;
                    size_t state_index = split_offsets == nullptr
                        ? ((size_t)bid * Hq + qh) * num_splits + split
                        : ((size_t)split_offsets[bid] + split) * Hq + qh;
                    if (lane == 0) {
                        partial_m[state_index] = -FLT_MAX;
                        partial_l[state_index] = 0.0f;
                    }
                    #pragma unroll
                    for (int j = 0; j < PKV_D_PER_LANE; ++j) {
                        int dim = lane + j * 32;
                        partial_acc[state_index * PKV_D + dim] = 0.0f;
                    }
                }
            }
        }
        return;
    }

    float m_h[HEADS_PER_WARP];
    float l_h[HEADS_PER_WARP];
    float acc[HEADS_PER_WARP][PKV_D_PER_LANE];

    for (int elem = tid; elem < Q_ROWS * PKV_D; elem += PKV_THREADS) {
        int g = elem / PKV_D;
        int dim = elem - g * PKV_D;
        smem_q[elem] = g < G
            ? Q[(size_t)(q_bid * Hq + qh0 + g) * PKV_D + dim]
            : __float2half(0.0f);
    }

    #pragma unroll
    for (int slot = 0; slot < HEADS_PER_WARP; ++slot) {
        m_h[slot] = -FLT_MAX;
        l_h[slot] = 0.0f;
        #pragma unroll
        for (int j = 0; j < PKV_D_PER_LANE; ++j) {
            acc[slot][j] = 0.0f;
        }
    }
    __syncthreads();

#define PKV_PREFETCH_ABS(abs_tile_, slot_) do { \
    int _t0 = (abs_tile_) * PKV_TILE; \
    int _tsz = min(PKV_TILE, effective_N - _t0); \
    kv.prefetch_tile( \
        bid, kvh, _t0, _tsz, tid, smem_k, smem_v[(slot_) % V_SLOTS]); \
} while (0)

    if (local_tiles > 0)
        PKV_PREFETCH_ABS(tile_begin, 0);

    for (int local_tile = 0; local_tile < local_tiles; ++local_tile) {
        int cur = local_tile & 1;
        int next = 1 - cur;
        int abs_tile = tile_begin + local_tile;

        pkv_cp_wait<0>();
        __syncthreads();

        int t0 = abs_tile * PKV_TILE;
        int tile_size = min(PKV_TILE, effective_N - t0);

        if constexpr (!USE_WMMA && PKV_TRANSPOSE_K) {
        // Transpose K from [token, dim] to padded [dim, token]. A warp can
        // then compute 32 token scores at once without shared-memory conflicts.
        for (int elem = tid; elem < tile_size * PKV_D;
             elem += PKV_THREADS) {
            int token = elem / PKV_D;
            int dim = elem - token * PKV_D;
            smem_kt[dim * PKV_KT_STRIDE + token] =
                smem_k[elem];
        }
        __syncthreads();
        }

        if constexpr (USE_WMMA) {
            using namespace nvcuda;
            if (warp < 2) {
                wmma::fragment<
                    wmma::accumulator, 16, 16, 16, float> score_frag;
                wmma::fill_fragment(score_frag, 0.0f);
                #pragma unroll
                for (int k0 = 0; k0 < PKV_D; k0 += 16) {
                    wmma::fragment<
                        wmma::matrix_a, 16, 16, 16,
                        __half, wmma::row_major> q_frag;
                    wmma::fragment<
                        wmma::matrix_b, 16, 16, 16,
                        __half, wmma::col_major> k_frag;
                    wmma::load_matrix_sync(
                        q_frag, &smem_q[k0], PKV_D);
                    wmma::load_matrix_sync(
                        k_frag,
                        &smem_k[(warp * 16) * PKV_D + k0],
                        PKV_D);
                    wmma::mma_sync(
                        score_frag, q_frag, k_frag, score_frag);
                }
                wmma::store_matrix_sync(
                    &smem_scores[warp * 16], score_frag,
                    PKV_TILE, wmma::mem_row_major);
            }
            __syncthreads();

            if (PKV_DOUBLE_BUFFER_V && local_tile + 1 < local_tiles)
                PKV_PREFETCH_ABS(abs_tile + 1, next);

            int g = warp;
            bool valid_lane = lane < tile_size;
            float score = valid_lane
                ? smem_scores[g * PKV_TILE + lane] * scale
                : -FLT_MAX;
            float tile_m = pkv_warp_max(score);
            float new_m = fmaxf(m_h[0], tile_m);
            float correction = pkv_exp(m_h[0] - new_m);
            float weight = valid_lane ? pkv_exp(score - new_m) : 0.0f;
            float tile_l = pkv_warp_sum(weight);
            l_h[0] = correction * l_h[0] + tile_l;

            if constexpr (USE_WMMA_V) {
                for (int elem = tid; elem < 16 * PKV_TILE;
                     elem += PKV_THREADS) {
                    smem_weights[elem] = __float2half(0.0f);
                }
                if (valid_lane)
                    smem_weights[g * PKV_TILE + lane] =
                        __float2half(weight);
                for (int elem = tid + tile_size * PKV_D;
                     elem < PKV_TILE * PKV_D;
                     elem += PKV_THREADS) {
                    smem_v[cur % V_SLOTS][elem] = __float2half(0.0f);
                }
                __syncthreads();

                #pragma unroll
                for (int output_block = 0; output_block < 2;
                     ++output_block) {
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
                            __half, wmma::row_major> value_input_frag;
                        wmma::load_matrix_sync(
                            weight_frag, &smem_weights[k0], PKV_TILE);
                        wmma::load_matrix_sync(
                            value_input_frag,
                            &smem_v[cur % V_SLOTS][k0 * PKV_D + n0],
                            PKV_D);
                        wmma::mma_sync(
                            value_frag, weight_frag,
                            value_input_frag, value_frag);
                    }
                    wmma::store_matrix_sync(
                        &smem_value_acc[n0], value_frag,
                        PKV_D, wmma::mem_row_major);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < PKV_D_PER_LANE; ++j) {
                    int dim = lane + j * 32;
                    acc[0][j] =
                        correction * acc[0][j] +
                        smem_value_acc[g * PKV_D + dim];
                }
            } else {
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
                                __half2float(
                                    smem_v[cur % V_SLOTS][
                                        token * PKV_D + dim]),
                                tile_acc);
                        }
                    }
                    acc[0][j] =
                        correction * acc[0][j] + tile_acc;
                }
            }
            m_h[0] = new_m;
        } else {
            // With transposition, K staging is free during compute. The direct
            // ablation intentionally waits because its score path still reads K.
            if constexpr (PKV_TRANSPOSE_K) {
                if (PKV_DOUBLE_BUFFER_V &&
                    local_tile + 1 < local_tiles)
                    PKV_PREFETCH_ABS(abs_tile + 1, next);
            }

            #pragma unroll
            for (int slot = 0; slot < HEADS_PER_WARP; ++slot) {
                int g = warp + slot * PKV_WARPS;
                if (g < G) {
                    float score = -FLT_MAX;
                    bool valid_lane = lane < tile_size;
                    if (valid_lane) {
                        score = 0.0f;
                        #pragma unroll 4
                        for (int dim = 0; dim < PKV_D; ++dim) {
                            score = fmaf(
                                __half2float(smem_q[g * PKV_D + dim]),
                                __half2float(PKV_TRANSPOSE_K
                                    ? smem_kt[
                                        dim * PKV_KT_STRIDE + lane]
                                    : smem_k[lane * PKV_D + dim]),
                                score);
                        }
                        score *= scale;
                    }

                    float tile_m = pkv_warp_max(score);
                    float new_m = fmaxf(m_h[slot], tile_m);
                    float correction = pkv_exp(m_h[slot] - new_m);
                    float weight = valid_lane ? pkv_exp(score - new_m) : 0.0f;
                    float tile_l = pkv_warp_sum(weight);
                    l_h[slot] = correction * l_h[slot] + tile_l;

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
                                    __half2float(
                                        smem_v[cur % V_SLOTS][
                                            token * PKV_D + dim]),
                                    tile_acc);
                            }
                        }
                        acc[slot][j] =
                            correction * acc[slot][j] + tile_acc;
                    }
                    m_h[slot] = new_m;
                }
            }
        }
        __syncthreads();
        if constexpr (!PKV_DOUBLE_BUFFER_V) {
            if (local_tile + 1 < local_tiles)
                PKV_PREFETCH_ABS(abs_tile + 1, next);
        } else if constexpr (!USE_WMMA && !PKV_TRANSPOSE_K) {
            if (local_tile + 1 < local_tiles)
                PKV_PREFETCH_ABS(abs_tile + 1, next);
        }
    }

#undef PKV_PREFETCH_ABS

    #pragma unroll
    for (int slot = 0; slot < HEADS_PER_WARP; ++slot) {
        int g = warp + slot * PKV_WARPS;
        if (g < G) {
            int qh = qh0 + g;
            if constexpr (WRITE_PARTIAL) {
                size_t state_index = split_offsets == nullptr
                    ? ((size_t)bid * Hq + qh) * num_splits + split
                    : ((size_t)split_offsets[bid] + split) * Hq + qh;
                if (lane == 0) {
                    partial_m[state_index] = m_h[slot];
                    partial_l[state_index] = l_h[slot];
                }
                #pragma unroll
                for (int j = 0; j < PKV_D_PER_LANE; ++j) {
                    int dim = lane + j * 32;
                    partial_acc[state_index * PKV_D + dim] = acc[slot][j];
                }
            } else {
                int out_bid =
                    out_indices == nullptr ? bid : out_indices[bid];
                size_t q_index = (size_t)out_bid * Hq + qh;
                float inv_l = l_h[slot] > 0.0f ? 1.0f / l_h[slot] : 0.0f;
                #pragma unroll
                for (int j = 0; j < PKV_D_PER_LANE; ++j) {
                    int dim = lane + j * 32;
                    O[q_index * PKV_D + dim] =
                        __float2half(acc[slot][j] * inv_l);
                }
            }
        }
    }

    if constexpr (WRITE_PARTIAL) {
        if (split_offsets != nullptr && merge_counters != nullptr) {
            int row_begin = split_offsets[bid];
            int row_end = split_offsets[bid + 1];
            int row_splits = row_end - row_begin;
            if (row_splits > 1) {
                __shared__ int do_merge;
                __threadfence();
                if (tid == 0) {
                    int counter_index = bid * Hkv + kvh;
                    int old = atomicAdd(&merge_counters[counter_index], 1);
                    do_merge = (old == row_splits - 1);
                }
                __syncthreads();

                if (do_merge) {
                    #pragma unroll
                    for (int head_slot = 0; head_slot < HEADS_PER_WARP; ++head_slot) {
                    int g = warp + head_slot * PKV_WARPS;
                    if (g < G) {
                        int qh = qh0 + g;
                        int out_bid =
                            out_indices == nullptr ? bid : out_indices[bid];

                        float global_m = -FLT_MAX;
                        for (int slot = row_begin; slot < row_end; ++slot)
                            global_m = fmaxf(
                                global_m,
                                partial_m[(size_t)slot * Hq + qh]);

                        float global_l = 0.0f;
                        float global_acc[PKV_D_PER_LANE];
                        #pragma unroll
                        for (int j = 0; j < PKV_D_PER_LANE; ++j)
                            global_acc[j] = 0.0f;

                        for (int slot = row_begin; slot < row_end; ++slot) {
                            size_t state_index = (size_t)slot * Hq + qh;
                            float correction =
                                pkv_exp(partial_m[state_index] - global_m);
                            global_l += correction * partial_l[state_index];
                            #pragma unroll
                            for (int j = 0; j < PKV_D_PER_LANE; ++j) {
                                int dim = lane + j * 32;
                                global_acc[j] = fmaf(
                                    correction,
                                    partial_acc[state_index * PKV_D + dim],
                                    global_acc[j]);
                            }
                        }

                        size_t q_index = (size_t)out_bid * Hq + qh;
                        float inv_l = global_l > 0.0f ? 1.0f / global_l : 0.0f;
                        #pragma unroll
                        for (int j = 0; j < PKV_D_PER_LANE; ++j) {
                            int dim = lane + j * 32;
                            O[q_index * PKV_D + dim] =
                                __float2half(global_acc[j] * inv_l);
                        }
                    }
                    }

                    __syncthreads();
                    if (tid == 0)
                        merge_counters[bid * Hkv + kvh] = 0;
                }
            }
        }
    }
}

__global__ void persistentkv_merge_splits_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    __half* __restrict__ O,
    const int* __restrict__ out_indices,
    int Hq,
    int num_splits)
{
    int qh = blockIdx.x;
    int bid = blockIdx.y;
    int out_bid = out_indices == nullptr ? bid : out_indices[bid];
    int dim = threadIdx.x;
    size_t q_index = (size_t)bid * Hq + qh;
    size_t state_base = q_index * num_splits;

    float global_m = -FLT_MAX;
    for (int split = 0; split < num_splits; ++split)
        global_m = fmaxf(global_m, partial_m[state_base + split]);

    float global_l = 0.0f;
    float global_acc = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
        float correction =
            pkv_exp(partial_m[state_base + split] - global_m);
        global_l += correction * partial_l[state_base + split];
        global_acc += correction *
            partial_acc[(state_base + split) * PKV_D + dim];
    }

    O[((size_t)out_bid * Hq + qh) * PKV_D + dim] =
        __float2half(global_l > 0.0f ? global_acc / global_l : 0.0f);
}

__global__ void persistentkv_merge_workqueue_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const int* __restrict__ split_offsets,
    const int* __restrict__ out_indices,
    __half* __restrict__ O,
    int Hq)
{
    int qh = blockIdx.x;
    int bid = blockIdx.y;
    int out_bid = out_indices == nullptr ? bid : out_indices[bid];
    int dim = threadIdx.x;
    int begin = split_offsets[bid];
    int end = split_offsets[bid + 1];

    float global_m = -FLT_MAX;
    for (int slot = begin; slot < end; ++slot)
        global_m = fmaxf(global_m, partial_m[(size_t)slot * Hq + qh]);

    float global_l = 0.0f;
    float global_acc = 0.0f;
    for (int slot = begin; slot < end; ++slot) {
        size_t state_index = (size_t)slot * Hq + qh;
        float correction = pkv_exp(partial_m[state_index] - global_m);
        global_l += correction * partial_l[state_index];
        global_acc += correction * partial_acc[state_index * PKV_D + dim];
    }

    O[((size_t)out_bid * Hq + qh) * PKV_D + dim] =
        __float2half(global_l > 0.0f ? global_acc / global_l : 0.0f);
}

// Exact query-head-centric baseline. Every query head independently streams its
// shared KV head; this intentionally exposes the redundant GQA traffic.
__global__ __launch_bounds__(PKV_THREADS, 2)
void pkv_query_head_baseline_kernel(
    const __half* __restrict__ Q,
    PkvContiguousAccessor kv,
    __half* __restrict__ O,
    int Hq,
    int N,
    int group_size,
    float scale)
{
    const int qh = blockIdx.x;
    const int bid = blockIdx.y;
    const int kvh = qh / group_size;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    __shared__ __half smem_k[2][PKV_TILE * PKV_D];
    __shared__ __half smem_v[2][PKV_TILE * PKV_D];

    float q_reg[PKV_D_PER_LANE] = {};
    float acc[PKV_D_PER_LANE] = {};
    float m = -FLT_MAX;
    float l = 0.0f;
    if (tid < 32) {
        #pragma unroll
        for (int j = 0; j < PKV_D_PER_LANE; ++j) {
            int dim = lane + j * 32;
            q_reg[j] =
                __half2float(Q[(size_t)(bid * Hq + qh) * PKV_D + dim]);
        }
    }

    int total_tiles = (N + PKV_TILE - 1) / PKV_TILE;
#define PKV_BASELINE_PREFETCH(tile_, slot_) do { \
    int _t0 = (tile_) * PKV_TILE; \
    int _tsz = min(PKV_TILE, N - _t0); \
    int _chunks = (_tsz * PKV_D) / 8; \
    for (int _chunk = tid; _chunk < _chunks; _chunk += PKV_THREADS) { \
        int _elem = _chunk * 8; \
        int _token = _t0 + _elem / PKV_D; \
        int _dim = _elem % PKV_D; \
        pkv_cp_async16(&smem_k[(slot_)][_elem], \
                       kv.k_ptr(bid, kvh, _token, _dim)); \
        pkv_cp_async16(&smem_v[(slot_)][_elem], \
                       kv.v_ptr(bid, kvh, _token, _dim)); \
    } \
    pkv_cp_commit(); \
} while (0)

    PKV_BASELINE_PREFETCH(0, 0);
    for (int tile = 0; tile < total_tiles; ++tile) {
        int cur = tile & 1;
        int next = 1 - cur;
        if (tile + 1 < total_tiles)
            PKV_BASELINE_PREFETCH(tile + 1, next);
        if (tile + 1 < total_tiles)
            pkv_cp_wait<1>();
        else
            pkv_cp_wait<0>();
        __syncthreads();

        if (tid < 32) {
            int tile_size = min(PKV_TILE, N - tile * PKV_TILE);
            for (int token = 0; token < tile_size; ++token) {
                float partial = 0.0f;
                #pragma unroll
                for (int j = 0; j < PKV_D_PER_LANE; ++j) {
                    int dim = lane + j * 32;
                    partial = fmaf(
                        q_reg[j],
                        __half2float(smem_k[cur][token * PKV_D + dim]),
                        partial);
                }
                float score = pkv_warp_sum(partial) * scale;
                float new_m = fmaxf(m, score);
                float correction = expf(m - new_m);
                float weight = expf(score - new_m);
                l = correction * l + weight;
                #pragma unroll
                for (int j = 0; j < PKV_D_PER_LANE; ++j) {
                    int dim = lane + j * 32;
                    float value =
                        __half2float(smem_v[cur][token * PKV_D + dim]);
                    acc[j] = correction * acc[j] + weight * value;
                }
                m = new_m;
            }
        }
        __syncthreads();
    }
#undef PKV_BASELINE_PREFETCH

    if (tid < 32) {
        float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        #pragma unroll
        for (int j = 0; j < PKV_D_PER_LANE; ++j) {
            int dim = lane + j * 32;
            O[(size_t)(bid * Hq + qh) * PKV_D + dim] =
                __float2half(acc[j] * inv_l);
        }
    }
}

inline size_t persistentkv_workspace_bytes(
    int B, int Hq, int num_splits)
{
    if (num_splits <= 1)
        return 0;
    size_t states = (size_t)B * Hq * num_splits;
    return states * (PKV_D + 2) * sizeof(float);
}

inline void persistentkv_partition_workspace(
    void* workspace,
    int B,
    int Hq,
    int num_splits,
    float** partial_m,
    float** partial_l,
    float** partial_acc)
{
    size_t states = (size_t)B * Hq * num_splits;
    *partial_m = static_cast<float*>(workspace);
    *partial_l = *partial_m + states;
    *partial_acc = *partial_l + states;
}

inline void persistentkv_partition_workqueue_workspace(
    void* workspace,
    int total_partial_slots,
    int Hq,
    float** partial_m,
    float** partial_l,
    float** partial_acc)
{
    size_t states = (size_t)total_partial_slots * Hq;
    *partial_m = static_cast<float*>(workspace);
    *partial_l = *partial_m + states;
    *partial_acc = *partial_l + states;
}

template<int G, typename Accessor>
inline void persistentkv_launch(
    const __half* Q,
    Accessor kv,
    __half* O,
    void* workspace,
    int B,
    int Hq,
    int Hkv,
    int N,
    int num_splits,
    cudaStream_t stream,
    const int* q_indices = nullptr,
    const int* out_indices = nullptr)
{
    float scale = 1.0f / sqrtf((float)PKV_D);
    dim3 block(PKV_THREADS);

    if (num_splits == 1) {
        dim3 grid(Hkv, B);
        persistentkv_decode_kernel<G, Accessor, false>
            <<<grid, block, 0, stream>>>(
                Q, kv, O, nullptr, nullptr, nullptr,
                q_indices, out_indices, nullptr, nullptr, nullptr,
                Hq, Hkv, N, 1, scale);
    } else {
        float* partial_m;
        float* partial_l;
        float* partial_acc;
        persistentkv_partition_workspace(
            workspace, B, Hq, num_splits,
            &partial_m, &partial_l, &partial_acc);

        dim3 partial_grid(Hkv * num_splits, B);
        persistentkv_decode_kernel<G, Accessor, true>
            <<<partial_grid, block, 0, stream>>>(
                Q, kv, nullptr, partial_m, partial_l, partial_acc,
                q_indices, out_indices, nullptr, nullptr, nullptr,
                Hq, Hkv, N, num_splits, scale);
        dim3 merge_grid(Hq, B);
        persistentkv_merge_splits_kernel
            <<<merge_grid, block, 0, stream>>>(
                partial_m, partial_l, partial_acc, O, out_indices,
                Hq, num_splits);
    }
}

template<int G, typename Accessor>
inline void persistentkv_workqueue_launch(
    const __half* Q,
    Accessor kv,
    const PkvWorkItem* work_items,
    const int* split_offsets,
    int* merge_counters,
    const int* q_indices,
    const int* out_indices,
    __half* O,
    void* workspace,
    int B,
    int Hq,
    int Hkv,
    int num_tasks,
    int total_partial_slots,
    int N,
    cudaStream_t stream)
{
    assert(workspace != nullptr && "Workqueue dispatch requires workspace");
    float* partial_m;
    float* partial_l;
    float* partial_acc;
    persistentkv_partition_workqueue_workspace(
        workspace, total_partial_slots, Hq,
        &partial_m, &partial_l, &partial_acc);

    float scale = 1.0f / sqrtf((float)PKV_D);
    dim3 block(PKV_THREADS);
    dim3 decode_grid(num_tasks);
    persistentkv_decode_kernel<G, Accessor, true>
        <<<decode_grid, block, 0, stream>>>(
            Q, kv, O, partial_m, partial_l, partial_acc,
            q_indices, out_indices, work_items, split_offsets,
            merge_counters, Hq, Hkv, N, 1, scale);
    if (merge_counters == nullptr) {
        dim3 merge_grid(Hq, B);
        persistentkv_merge_workqueue_kernel
            <<<merge_grid, block, 0, stream>>>(
                partial_m, partial_l, partial_acc, split_offsets,
                out_indices, O, Hq);
    }
}

template<typename Accessor>
inline void persistentkv_dispatch_accessor(
    const __half* Q,
    Accessor kv,
    __half* O,
    void* workspace,
    int B,
    int Hq,
    int Hkv,
    int N,
    int d,
    int num_splits,
    cudaStream_t stream,
    const int* q_indices = nullptr,
    const int* out_indices = nullptr)
{
    assert(d == PKV_D && "Head dimension must equal PKV_D");
    assert(Hkv > 0 && Hq % Hkv == 0 && "Invalid GQA head counts");
    int G = Hq / Hkv;
    assert(pkv_is_supported_g(G) && G <= PKV_MAX_G &&
           "Unsupported G; supported values are 1, 2, 4, and 8");

    int total_tiles = (N + PKV_TILE - 1) / PKV_TILE;
    assert(num_splits >= 1 && num_splits <= total_tiles);
    assert((num_splits == 1 || workspace != nullptr) &&
           "Split dispatch requires workspace");

    switch (G) {
        case 1:
            persistentkv_launch<1>(
                Q, kv, O, workspace, B, Hq, Hkv, N, num_splits,
                stream, q_indices, out_indices);
            break;
        case 2:
            persistentkv_launch<2>(
                Q, kv, O, workspace, B, Hq, Hkv, N, num_splits,
                stream, q_indices, out_indices);
            break;
        case 4:
            persistentkv_launch<4>(
                Q, kv, O, workspace, B, Hq, Hkv, N, num_splits,
                stream, q_indices, out_indices);
            break;
        case 8:
            persistentkv_launch<8>(
                Q, kv, O, workspace, B, Hq, Hkv, N, num_splits,
                stream, q_indices, out_indices);
            break;
        default:
            fprintf(
                stderr,
                "Unsupported G=%d; supported values are 1, 2, 4, and 8\n",
                G);
            return;
    }
    PKV_CHECK(cudaGetLastError());
}

inline void persistentkv_dispatch(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int B,
    int Hq,
    int Hkv,
    int N,
    int d,
    cudaStream_t stream = 0)
{
    PkvContiguousAccessor kv{K, V, Hkv, N};
    persistentkv_dispatch_accessor(
        Q, kv, O, nullptr, B, Hq, Hkv, N, d, 1, stream);
}

inline void persistentkv_split_dispatch(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    void* workspace,
    int B,
    int Hq,
    int Hkv,
    int N,
    int d,
    int num_splits,
    cudaStream_t stream = 0)
{
    PkvContiguousAccessor kv{K, V, Hkv, N};
    persistentkv_dispatch_accessor(
        Q, kv, O, workspace, B, Hq, Hkv, N, d, num_splits, stream);
}

inline void persistentkv_paged_dispatch(
    const __half* Q,
    const __half* K_pages,
    const __half* V_pages,
    const int* block_table,
    __half* O,
    void* workspace,
    int B,
    int Hq,
    int Hkv,
    int N,
    int d,
    int page_size,
    int max_pages,
    int num_splits,
    cudaStream_t stream = 0)
{
    assert(page_size > 0 && max_pages >= (N + page_size - 1) / page_size);
    PkvPagedAccessor kv{
        K_pages, V_pages, block_table, nullptr, max_pages, Hkv, page_size};
    persistentkv_dispatch_accessor(
        Q, kv, O, workspace, B, Hq, Hkv, N, d, num_splits, stream);
}

inline void persistentkv_paged_masked_dispatch(
    const __half* Q,
    const __half* K_pages,
    const __half* V_pages,
    const int* block_table,
    const int* seq_lens,
    __half* O,
    void* workspace,
    int B,
    int Hq,
    int Hkv,
    int N,
    int d,
    int page_size,
    int max_pages,
    int num_splits,
    cudaStream_t stream = 0)
{
    assert(page_size > 0 && max_pages >= (N + page_size - 1) / page_size);
    PkvPagedAccessor kv{
        K_pages, V_pages, block_table, seq_lens, max_pages, Hkv, page_size};
    persistentkv_dispatch_accessor(
        Q, kv, O, workspace, B, Hq, Hkv, N, d, num_splits, stream);
}

inline void persistentkv_paged_masked_indexed_dispatch(
    const __half* Q,
    const __half* K_pages,
    const __half* V_pages,
    const int* block_table,
    const int* seq_lens,
    const int* q_indices,
    const int* out_indices,
    __half* O,
    void* workspace,
    int B,
    int Hq,
    int Hkv,
    int N,
    int d,
    int page_size,
    int max_pages,
    int num_splits,
    cudaStream_t stream = 0)
{
    assert(page_size > 0 && max_pages >= (N + page_size - 1) / page_size);
    PkvPagedAccessor kv{
        K_pages, V_pages, block_table, seq_lens, max_pages, Hkv, page_size};
    persistentkv_dispatch_accessor(
        Q, kv, O, workspace, B, Hq, Hkv, N, d, num_splits, stream,
        q_indices, out_indices);
}

inline void persistentkv_paged_workqueue_indexed_dispatch(
    const __half* Q,
    const __half* K_pages,
    const __half* V_pages,
    const int* block_table,
    const int* seq_lens,
    const PkvWorkItem* work_items,
    const int* split_offsets,
    int* merge_counters,
    const int* q_indices,
    const int* out_indices,
    __half* O,
    void* workspace,
    int B,
    int Hq,
    int Hkv,
    int N,
    int d,
    int page_size,
    int max_pages,
    int num_tasks,
    int total_partial_slots,
    cudaStream_t stream = 0)
{
    assert(page_size > 0 && max_pages >= (N + page_size - 1) / page_size);
    assert(d == PKV_D && "Head dimension must equal PKV_D");
    assert(Hkv > 0 && Hq % Hkv == 0 && "Invalid GQA head counts");
    assert(num_tasks > 0 && total_partial_slots > 0);
    PkvPagedAccessor kv{
        K_pages, V_pages, block_table, seq_lens, max_pages, Hkv, page_size};
    int G = Hq / Hkv;
    assert(pkv_is_supported_g(G) && G <= PKV_MAX_G &&
           "Unsupported G; supported values are 1, 2, 4, and 8");
    switch (G) {
        case 1:
            persistentkv_workqueue_launch<1>(
                Q, kv, work_items, split_offsets, merge_counters,
                q_indices, out_indices, O, workspace, B, Hq, Hkv,
                num_tasks, total_partial_slots,
                N, stream);
            break;
        case 2:
            persistentkv_workqueue_launch<2>(
                Q, kv, work_items, split_offsets, merge_counters,
                q_indices, out_indices, O, workspace, B, Hq, Hkv,
                num_tasks, total_partial_slots,
                N, stream);
            break;
        case 4:
            persistentkv_workqueue_launch<4>(
                Q, kv, work_items, split_offsets, merge_counters,
                q_indices, out_indices, O, workspace, B, Hq, Hkv,
                num_tasks, total_partial_slots,
                N, stream);
            break;
        case 8:
            persistentkv_workqueue_launch<8>(
                Q, kv, work_items, split_offsets, merge_counters,
                q_indices, out_indices, O, workspace, B, Hq, Hkv,
                num_tasks, total_partial_slots,
                N, stream);
            break;
        default:
            fprintf(
                stderr,
                "Unsupported G=%d; supported values are 1, 2, 4, and 8\n",
                G);
            return;
    }
    PKV_CHECK(cudaGetLastError());
}

inline void persistentkv_baseline_dispatch(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int B,
    int Hq,
    int Hkv,
    int N,
    int d,
    cudaStream_t stream = 0)
{
    assert(d == PKV_D && Hkv > 0 && Hq % Hkv == 0);
    PkvContiguousAccessor kv{K, V, Hkv, N};
    dim3 grid(Hq, B);
    dim3 block(PKV_THREADS);
    pkv_query_head_baseline_kernel<<<grid, block, 0, stream>>>(
        Q, kv, O, Hq, N, Hq / Hkv, 1.0f / sqrtf((float)d));
    PKV_CHECK(cudaGetLastError());
}
