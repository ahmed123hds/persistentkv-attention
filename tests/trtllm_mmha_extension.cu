#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionLaunch.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

namespace
{

using tensorrt_llm::kernels::KVBlockArray;
using tensorrt_llm::kernels::KVLinearBuffer;
using tensorrt_llm::kernels::Masked_multihead_attention_params;
using tensorrt_llm::kernels::PositionEmbeddingType;

void check_cuda_half(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be FP16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor trtllm_paged_mmha(
    torch::Tensor q,
    torch::Tensor k_last,
    torch::Tensor v_last,
    torch::Tensor block_ptrs,
    torch::Tensor sequence_lengths,
    torch::Tensor output,
    torch::Tensor partial_out,
    torch::Tensor partial_sum,
    torch::Tensor partial_max,
    torch::Tensor block_counter,
    int64_t context,
    int64_t page_size)
{
    constexpr int kNumQHeads = 32;
    constexpr int kNumKVHeads = 8;
    constexpr int kHeadDim = 128;
    constexpr int kMaxSeqTiles = 8;

    check_cuda_half(q, "q");
    check_cuda_half(k_last, "k_last");
    check_cuda_half(v_last, "v_last");
    TORCH_CHECK(q.numel() == kNumQHeads * kHeadDim, "unexpected q shape");
    TORCH_CHECK(k_last.numel() == kNumKVHeads * kHeadDim, "unexpected k_last shape");
    TORCH_CHECK(v_last.numel() == kNumKVHeads * kHeadDim, "unexpected v_last shape");
    TORCH_CHECK(block_ptrs.is_cuda() && block_ptrs.scalar_type() == torch::kInt64
            && block_ptrs.is_contiguous(),
        "block_ptrs must be contiguous CUDA int64");
    TORCH_CHECK(sequence_lengths.is_cuda()
            && sequence_lengths.scalar_type() == torch::kInt32
            && sequence_lengths.numel() == 1,
        "sequence_lengths must be one CUDA int32 value");
    TORCH_CHECK(context > 0 && context % page_size == 0, "invalid context/page size");
    TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0,
        "TensorRT-LLM MMHA page size must be a positive power of two");

    const int logical_pages = static_cast<int>(context / page_size);
    TORCH_CHECK(block_ptrs.numel() == 2 * logical_pages,
        "block_ptrs must contain K and V pointers for every logical page");
    check_cuda_half(output, "output");
    check_cuda_half(partial_out, "partial_out");
    TORCH_CHECK(output.numel() == q.numel(), "unexpected output shape");
    TORCH_CHECK(partial_out.numel()
            == kMaxSeqTiles * kNumQHeads * kHeadDim,
        "unexpected partial_out shape");
    TORCH_CHECK(partial_sum.is_cuda()
            && partial_sum.scalar_type() == torch::kFloat32
            && partial_sum.is_contiguous()
            && partial_sum.numel() == kMaxSeqTiles * kNumQHeads,
        "unexpected partial_sum");
    TORCH_CHECK(partial_max.is_cuda()
            && partial_max.scalar_type() == torch::kFloat32
            && partial_max.is_contiguous()
            && partial_max.numel() == partial_sum.numel(),
        "unexpected partial_max");
    TORCH_CHECK(block_counter.is_cuda()
            && block_counter.scalar_type() == torch::kInt32
            && block_counter.is_contiguous()
            && block_counter.numel() == kNumQHeads,
        "unexpected block_counter");

    cudaDeviceProp properties{};
    const int device = q.get_device();
    TORCH_CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
        "cudaGetDeviceProperties failed");

    Masked_multihead_attention_params<uint16_t> params{};
    params.out = reinterpret_cast<uint16_t*>(output.data_ptr<at::Half>());
    params.q = reinterpret_cast<const uint16_t*>(q.data_ptr<at::Half>());
    params.k = reinterpret_cast<const uint16_t*>(k_last.data_ptr<at::Half>());
    params.v = reinterpret_cast<const uint16_t*>(v_last.data_ptr<at::Half>());
    params.stride = kNumQHeads * kHeadDim + 2 * kNumKVHeads * kHeadDim;
    params.batch_size = 1;
    params.beam_width = 1;
    params.max_attention_window_size = static_cast<int>(context);
    params.cyclic_attention_window_size = static_cast<int>(context);
    params.num_heads = kNumQHeads;
    params.num_kv_heads = kNumKVHeads;
    params.hidden_size_per_head = kHeadDim;
    params.position_embedding_type = PositionEmbeddingType::kLEARNED_ABSOLUTE;
    params.timestep = static_cast<int>(context - 1);
    params.inv_sqrt_dh = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    params.length_per_sample = sequence_lengths.data_ptr<int32_t>();
    params.multi_processor_count = properties.multiProcessorCount;

    const int max_dynamic_smem = properties.sharedMemPerBlockOptin - 2048;
    const int qk_elements = static_cast<int>((context + 3) / 4);
    const int elements_per_block = max_dynamic_smem / 24;
    const int min_seq_tiles = std::max(
        1, (qk_elements + elements_per_block - 1) / elements_per_block);
    TORCH_CHECK(min_seq_tiles <= kMaxSeqTiles, "context needs too many MMHA tiles");
    params.multi_block_mode = min_seq_tiles > 1;
    params.min_seq_len_tile = min_seq_tiles;
    params.max_seq_len_tile = kMaxSeqTiles;
    params.partial_out = reinterpret_cast<uint16_t*>(
        partial_out.data_ptr<at::Half>());
    params.partial_sum = partial_sum.data_ptr<float>();
    params.partial_max = partial_max.data_ptr<float>();
    params.block_counter = block_counter.data_ptr<int32_t>();

    KVBlockArray cache(
        1,
        logical_pages,
        static_cast<int>(page_size),
        kNumKVHeads * kHeadDim * sizeof(uint16_t),
        static_cast<int>(context),
        0,
        false);
    cache.data = block_ptrs.data_ptr<int64_t>();
    KVLinearBuffer unused_shift_cache;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
    if (params.multi_block_mode)
    {
        TORCH_CHECK(cudaMemsetAsync(
                        block_counter.data_ptr<int32_t>(),
                        0,
                        block_counter.nbytes(),
                        stream)
                == cudaSuccess,
            "TensorRT-LLM block-counter reset failed");
    }
    tensorrt_llm::kernels::mmha::mmha_launch_kernel<
        uint16_t,
        KVBlockArray,
        Masked_multihead_attention_params<uint16_t>,
        kHeadDim,
        false>(params, cache, unused_shift_cache, stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TensorRT-LLM MMHA launch failed");
    return output;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("paged_mmha", &trtllm_paged_mmha);
}
