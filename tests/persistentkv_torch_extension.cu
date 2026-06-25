#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "../kernels/persistentkv_attention.cuh"

namespace {

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    int64_t dims)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(tensor.dim() == dims, name, " has wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_supported_gqa(int Hq, int Hkv)
{
    TORCH_CHECK(Hkv > 0, "Hkv must be positive");
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
    int G = Hq / Hkv;
    TORCH_CHECK(
        G == 1 || G == 2 || G == 4 || G == 8,
        "PersistentKV supports G=Hq/Hkv in {1,2,4,8}; got G=", G,
        " from Hq=", Hq, " Hkv=", Hkv);
    TORCH_CHECK(G <= PKV_MAX_G, "G exceeds PKV_MAX_G");
}

}  // namespace

void persistentkv_paged_attention(
    torch::Tensor q,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor block_table,
    torch::Tensor output,
    torch::Tensor workspace,
    int64_t context,
    int64_t page_size,
    int64_t num_splits)
{
    check_tensor(q, "q", at::kHalf, 3);
    check_tensor(k_pages, "k_pages", at::kHalf, 4);
    check_tensor(v_pages, "v_pages", at::kHalf, 4);
    check_tensor(block_table, "block_table", at::kInt, 2);
    check_tensor(output, "output", at::kHalf, 3);
    TORCH_CHECK(workspace.is_cuda(), "workspace must be a CUDA tensor");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");

    int B = static_cast<int>(q.size(0));
    int Hq = static_cast<int>(q.size(1));
    int d = static_cast<int>(q.size(2));
    int Hkv = static_cast<int>(k_pages.size(1));
    int physical_pages = static_cast<int>(k_pages.size(0));
    int max_pages = static_cast<int>(block_table.size(1));

    TORCH_CHECK(d == PKV_D, "PersistentKV extension currently requires d=128");
    TORCH_CHECK(output.size(0) == B && output.size(1) == Hq &&
                output.size(2) == d, "output shape must match q");
    TORCH_CHECK(v_pages.size(0) == physical_pages &&
                v_pages.size(1) == Hkv &&
                v_pages.size(2) == page_size &&
                v_pages.size(3) == d, "v_pages shape mismatch");
    TORCH_CHECK(k_pages.size(2) == page_size && k_pages.size(3) == d,
                "k_pages shape mismatch");
    TORCH_CHECK(block_table.size(0) == B, "block_table batch mismatch");
    check_supported_gqa(Hq, Hkv);
    TORCH_CHECK(context > 0, "context must be positive");
    TORCH_CHECK(page_size > 0, "page_size must be positive");
    TORCH_CHECK(num_splits > 0, "num_splits must be positive");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    persistentkv_paged_dispatch(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_pages.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_pages.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        workspace.numel() > 0 ? workspace.data_ptr() : nullptr,
        B,
        Hq,
        Hkv,
        static_cast<int>(context),
        d,
        static_cast<int>(page_size),
        max_pages,
        static_cast<int>(num_splits),
        stream);
}

void persistentkv_paged_attention_masked(
    torch::Tensor q,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor output,
    torch::Tensor workspace,
    int64_t bucket_context,
    int64_t page_size,
    int64_t num_splits)
{
    check_tensor(q, "q", at::kHalf, 3);
    check_tensor(k_pages, "k_pages", at::kHalf, 4);
    check_tensor(v_pages, "v_pages", at::kHalf, 4);
    check_tensor(block_table, "block_table", at::kInt, 2);
    check_tensor(seq_lens, "seq_lens", at::kInt, 1);
    check_tensor(output, "output", at::kHalf, 3);
    TORCH_CHECK(workspace.is_cuda(), "workspace must be a CUDA tensor");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");

    int B = static_cast<int>(q.size(0));
    int Hq = static_cast<int>(q.size(1));
    int d = static_cast<int>(q.size(2));
    int Hkv = static_cast<int>(k_pages.size(1));
    int physical_pages = static_cast<int>(k_pages.size(0));
    int max_pages = static_cast<int>(block_table.size(1));

    TORCH_CHECK(d == PKV_D, "PersistentKV extension currently requires d=128");
    TORCH_CHECK(output.size(0) == B && output.size(1) == Hq &&
                output.size(2) == d, "output shape must match q");
    TORCH_CHECK(seq_lens.size(0) == B, "seq_lens batch mismatch");
    TORCH_CHECK(v_pages.size(0) == physical_pages &&
                v_pages.size(1) == Hkv &&
                v_pages.size(2) == page_size &&
                v_pages.size(3) == d, "v_pages shape mismatch");
    TORCH_CHECK(k_pages.size(2) == page_size && k_pages.size(3) == d,
                "k_pages shape mismatch");
    TORCH_CHECK(block_table.size(0) == B, "block_table batch mismatch");
    check_supported_gqa(Hq, Hkv);
    TORCH_CHECK(bucket_context > 0, "bucket_context must be positive");
    TORCH_CHECK(page_size > 0, "page_size must be positive");
    TORCH_CHECK(num_splits > 0, "num_splits must be positive");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    persistentkv_paged_masked_dispatch(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_pages.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_pages.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        workspace.numel() > 0 ? workspace.data_ptr() : nullptr,
        B,
        Hq,
        Hkv,
        static_cast<int>(bucket_context),
        d,
        static_cast<int>(page_size),
        max_pages,
        static_cast<int>(num_splits),
        stream);
}

void persistentkv_paged_attention_indexed(
    torch::Tensor q_source,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor q_indices,
    torch::Tensor out_indices,
    torch::Tensor output,
    torch::Tensor workspace,
    int64_t bucket_context,
    int64_t page_size,
    int64_t num_splits)
{
    check_tensor(q_source, "q_source", at::kHalf, 3);
    check_tensor(k_pages, "k_pages", at::kHalf, 4);
    check_tensor(v_pages, "v_pages", at::kHalf, 4);
    check_tensor(block_table, "block_table", at::kInt, 2);
    check_tensor(seq_lens, "seq_lens", at::kInt, 1);
    check_tensor(q_indices, "q_indices", at::kInt, 1);
    check_tensor(out_indices, "out_indices", at::kInt, 1);
    check_tensor(output, "output", at::kHalf, 3);
    TORCH_CHECK(workspace.is_cuda(), "workspace must be a CUDA tensor");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");

    int B = static_cast<int>(block_table.size(0));
    int Hq = static_cast<int>(q_source.size(1));
    int d = static_cast<int>(q_source.size(2));
    int Hkv = static_cast<int>(k_pages.size(1));
    int physical_pages = static_cast<int>(k_pages.size(0));
    int max_pages = static_cast<int>(block_table.size(1));

    TORCH_CHECK(d == PKV_D, "PersistentKV extension currently requires d=128");
    TORCH_CHECK(output.size(1) == Hq && output.size(2) == d,
                "output shape must match q head dimensions");
    TORCH_CHECK(seq_lens.size(0) == B, "seq_lens batch mismatch");
    TORCH_CHECK(q_indices.size(0) == B, "q_indices batch mismatch");
    TORCH_CHECK(out_indices.size(0) == B, "out_indices batch mismatch");
    TORCH_CHECK(v_pages.size(0) == physical_pages &&
                v_pages.size(1) == Hkv &&
                v_pages.size(2) == page_size &&
                v_pages.size(3) == d, "v_pages shape mismatch");
    TORCH_CHECK(k_pages.size(2) == page_size && k_pages.size(3) == d,
                "k_pages shape mismatch");
    check_supported_gqa(Hq, Hkv);
    TORCH_CHECK(bucket_context > 0, "bucket_context must be positive");
    TORCH_CHECK(page_size > 0, "page_size must be positive");
    TORCH_CHECK(num_splits > 0, "num_splits must be positive");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    persistentkv_paged_masked_indexed_dispatch(
        reinterpret_cast<const __half*>(q_source.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_pages.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_pages.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        q_indices.data_ptr<int>(),
        out_indices.data_ptr<int>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        workspace.numel() > 0 ? workspace.data_ptr() : nullptr,
        B,
        Hq,
        Hkv,
        static_cast<int>(bucket_context),
        d,
        static_cast<int>(page_size),
        max_pages,
        static_cast<int>(num_splits),
        stream);
}

void persistentkv_paged_attention_workqueue(
    torch::Tensor q_source,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor work_items,
    torch::Tensor split_offsets,
    torch::Tensor merge_counters,
    torch::Tensor q_indices,
    torch::Tensor out_indices,
    torch::Tensor output,
    torch::Tensor workspace,
    int64_t bucket_context,
    int64_t page_size,
    int64_t total_partial_slots,
    bool fused_merge)
{
    check_tensor(q_source, "q_source", at::kHalf, 3);
    check_tensor(k_pages, "k_pages", at::kHalf, 4);
    check_tensor(v_pages, "v_pages", at::kHalf, 4);
    check_tensor(block_table, "block_table", at::kInt, 2);
    check_tensor(seq_lens, "seq_lens", at::kInt, 1);
    check_tensor(work_items, "work_items", at::kInt, 2);
    check_tensor(split_offsets, "split_offsets", at::kInt, 1);
    check_tensor(merge_counters, "merge_counters", at::kInt, 2);
    check_tensor(q_indices, "q_indices", at::kInt, 1);
    check_tensor(out_indices, "out_indices", at::kInt, 1);
    check_tensor(output, "output", at::kHalf, 3);
    TORCH_CHECK(workspace.is_cuda(), "workspace must be a CUDA tensor");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");

    int B = static_cast<int>(block_table.size(0));
    int Hq = static_cast<int>(q_source.size(1));
    int d = static_cast<int>(q_source.size(2));
    int Hkv = static_cast<int>(k_pages.size(1));
    int physical_pages = static_cast<int>(k_pages.size(0));
    int max_pages = static_cast<int>(block_table.size(1));
    int num_tasks = static_cast<int>(work_items.size(0));
    int partial_slots = static_cast<int>(total_partial_slots);

    TORCH_CHECK(d == PKV_D, "PersistentKV extension currently requires d=128");
    TORCH_CHECK(work_items.size(1) == 5, "work_items must have shape [T, 5]");
    TORCH_CHECK(split_offsets.size(0) == B + 1, "split_offsets batch mismatch");
    TORCH_CHECK(merge_counters.size(0) == B && merge_counters.size(1) == Hkv,
                "merge_counters shape mismatch");
    TORCH_CHECK(output.size(1) == Hq && output.size(2) == d,
                "output shape must match q head dimensions");
    TORCH_CHECK(seq_lens.size(0) == B, "seq_lens batch mismatch");
    TORCH_CHECK(q_indices.size(0) == B, "q_indices batch mismatch");
    TORCH_CHECK(out_indices.size(0) == B, "out_indices batch mismatch");
    TORCH_CHECK(v_pages.size(0) == physical_pages &&
                v_pages.size(1) == Hkv &&
                v_pages.size(2) == page_size &&
                v_pages.size(3) == d, "v_pages shape mismatch");
    TORCH_CHECK(k_pages.size(2) == page_size && k_pages.size(3) == d,
                "k_pages shape mismatch");
    check_supported_gqa(Hq, Hkv);
    TORCH_CHECK(bucket_context > 0, "bucket_context must be positive");
    TORCH_CHECK(page_size > 0, "page_size must be positive");
    TORCH_CHECK(num_tasks > 0, "work_items must not be empty");
    TORCH_CHECK(partial_slots > 0, "total_partial_slots must be positive");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    persistentkv_paged_workqueue_indexed_dispatch(
        reinterpret_cast<const __half*>(q_source.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_pages.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_pages.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        reinterpret_cast<const PkvWorkItem*>(work_items.data_ptr<int>()),
        split_offsets.data_ptr<int>(),
        fused_merge ? merge_counters.data_ptr<int>() : nullptr,
        q_indices.data_ptr<int>(),
        out_indices.data_ptr<int>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        workspace.numel() > 0 ? workspace.data_ptr() : nullptr,
        B,
        Hq,
        Hkv,
        static_cast<int>(bucket_context),
        d,
        static_cast<int>(page_size),
        max_pages,
        num_tasks,
        partial_slots,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "paged_attention",
        &persistentkv_paged_attention,
        "PersistentKV native paged attention");
    m.def(
        "paged_attention_masked",
        &persistentkv_paged_attention_masked,
        "PersistentKV native paged attention with per-request sequence lengths");
    m.def(
        "paged_attention_indexed",
        &persistentkv_paged_attention_indexed,
        "PersistentKV native paged attention with indexed Q/O rows");
    m.def(
        "paged_attention_workqueue",
        &persistentkv_paged_attention_workqueue,
        "PersistentKV native paged attention with compact work queue");
}
