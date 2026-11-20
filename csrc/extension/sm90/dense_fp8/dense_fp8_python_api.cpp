#include <algorithm>
#include <limits>

#include <torch/all.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>

#include "dense_fp8_api.h"
#include "flash_mla.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace {

std::vector<at::Tensor>
fwd_kvcache_mla_fp8_impl(
    const at::Tensor &q_input,                   // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &causal_seqlens_k, // None or batch_size x seqlen_q
    const std::optional<at::Tensor> &q_lens,     // None or batch_size
    const std::optional<at::Tensor> &descale_q,  // None or batch_size
    const std::optional<at::Tensor> &descale_k,  // None or batch_size
    bool use_fp32_output,
    int canonical_num_chunks,
    bool canonical_return_leaves,
    const std::optional<at::Tensor> &out_opt,
    const std::optional<at::Tensor> &softmax_lse_opt,
    const std::optional<at::Tensor> &canonical_leaf_out_opt
) {
    int head_size_v_int = static_cast<int>(head_size_v);

    // Check the architecture
    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 9 && dprops->minor == 0, "Dense FP8 MLA is only supported on SM90");

    // Check data types
    TORCH_CHECK(q_input.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(kcache.dtype() == q_input.dtype(), "query and key must have the same dtype");
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");
    TORCH_CHECK(
        canonical_num_chunks == 0 || canonical_num_chunks == 1 ||
            canonical_num_chunks == 2 ||
            canonical_num_chunks == 4 || canonical_num_chunks == 8,
        "canonical_num_chunks must be one of 0, 1, 2, 4, or 8");
    TORCH_CHECK(
        canonical_num_chunks == 0 || use_fp32_output,
        "canonical dense FP8 attention requires FP32 output");
    TORCH_CHECK(
        !canonical_return_leaves || canonical_num_chunks > 0,
        "canonical leaf export requires canonical metadata");
    TORCH_CHECK(
        !canonical_return_leaves || !q_lens.has_value(),
        "canonical leaf export does not yet support q_lens");

    // Check device
    CHECK_DEVICE(q_input);
    CHECK_DEVICE(kcache);
    CHECK_DEVICE(seqlens_k);
    CHECK_DEVICE(block_table);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_DEVICE(num_splits);
    if (causal_seqlens_k.has_value()) CHECK_DEVICE(causal_seqlens_k.value());
    if (q_lens.has_value()) CHECK_DEVICE(q_lens.value());
    if (descale_q.has_value()) CHECK_DEVICE(descale_q.value());
    if (descale_k.has_value()) CHECK_DEVICE(descale_k.value());

    // Check layout
    TORCH_CHECK(q_input.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "kcache must have contiguous last dimension");
    CHECK_CONTIGUOUS(seqlens_k);
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(num_splits);
    if (causal_seqlens_k.has_value()) {
        CHECK_CONTIGUOUS(causal_seqlens_k.value());
    }
    if (q_lens.has_value()) CHECK_CONTIGUOUS(q_lens.value());

    const auto sizes = q_input.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_q = sizes[2];
    const int head_size_k = sizes[3];
    TORCH_CHECK(head_size_k == 576, "Only head_size_k == 576 is supported");
    TORCH_CHECK(head_size_v_int == 512, "Only head_size_v == 512 is supported");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    TORCH_CHECK(page_block_size == 64, "Currently page_block_size must be 64");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads_q % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (causal_seqlens_k.has_value()) {
        const auto &causal_seqlens_k_ = causal_seqlens_k.value();
        TORCH_CHECK(
            causal_seqlens_k_.dtype() == torch::kInt32,
            "causal_seqlens_k must have dtype int32");
        CHECK_SHAPE(causal_seqlens_k_, batch_size, seqlen_q_ori);
        is_causal = true;
    }
    if (q_lens.has_value()) {
        const auto &q_lens_ = q_lens.value();
        TORCH_CHECK(
            q_lens_.dtype() == torch::kInt32,
            "q_lens must have dtype int32");
        CHECK_SHAPE(q_lens_, batch_size);
        TORCH_CHECK(
            q_lens_.get_device() == q_input.get_device(),
            "q_lens must be on the same CUDA device as q");
        TORCH_CHECK(
            seqlen_q_ori <= 8,
            "q_lens currently supports fixed physical Qmax <= 8");
    }

    TORCH_CHECK(descale_q.has_value() && descale_k.has_value(), "descale is required when input dtype is fp8");
    auto descale_q_ = descale_q.value();
    auto descale_k_ = descale_k.value();
    CHECK_DEVICE(descale_q_);
    CHECK_DEVICE(descale_k_);
    TORCH_CHECK(descale_q_.stride(-1) == 1);
    TORCH_CHECK(descale_k_.stride(-1) == 1);
    TORCH_CHECK(descale_q_.dtype() == torch::kFloat);
    TORCH_CHECK(descale_k_.dtype() == torch::kFloat);
    CHECK_SHAPE(descale_q_, 1);
    CHECK_SHAPE(descale_k_, 1);

    if (seqlen_q_ori == 1 && !causal_seqlens_k.has_value()) {
        is_causal = false;
    }

    const int num_q_heads_per_hk = num_heads_q / num_heads_k;
    const int q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk;
    const int num_heads = num_heads_k;
    at::Tensor q =
        q_input
            .view(
                {batch_size,
                 seqlen_q_ori,
                 num_heads_k,
                 num_q_heads_per_hk,
                 head_size_k})
            .transpose(2, 3)
            .reshape(
                {batch_size, q_seq_per_hk, num_heads, head_size_k});

    CHECK_SHAPE(q, batch_size, q_seq_per_hk, num_heads, head_size_k);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    CHECK_SHAPE(seqlens_k, batch_size);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_SHAPE(num_splits, batch_size+1);
    if (canonical_num_chunks > 0) {
        TORCH_CHECK(
            tile_scheduler_metadata.size(0) ==
                batch_size * canonical_num_chunks,
            "canonical metadata must contain batch_size * num_chunks rows");
    }

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    caffe2::TypeMeta out_type;
    out_type = use_fp32_output ? torch::kFloat : torch::kBFloat16;
    TORCH_CHECK(
        out_opt.has_value() == softmax_lse_opt.has_value(),
        "out and softmax_lse must either both be supplied or both be omitted");
    const bool use_preallocated_output = out_opt.has_value();
    const bool use_preallocated_leaf_output =
        canonical_leaf_out_opt.has_value();
    TORCH_CHECK(
        !use_fp32_output || !use_preallocated_output,
        "FP32 dense FP8 output does not support caller-owned output buffers");
    TORCH_CHECK(
        !use_preallocated_leaf_output || canonical_return_leaves,
        "canonical leaf output requires the leaf-export path");
    at::Tensor public_out;
    at::Tensor public_softmax_lse;
    at::Tensor out;
    at::Tensor softmax_lse;
    if (use_preallocated_leaf_output) {
        TORCH_CHECK(
            num_heads_k == 1 && canonical_num_chunks == 1,
            "preallocated canonical leaf output currently requires MQA and "
            "one local chunk");
        public_out = canonical_leaf_out_opt.value();
        CHECK_DEVICE(public_out);
        TORCH_CHECK(
            public_out.get_device() == q.get_device(),
            "preallocated leaf output must be on the same CUDA device as q");
        TORCH_CHECK(
            public_out.dtype() == torch::kFloat,
            "preallocated canonical leaf output must have dtype float32");
        CHECK_SHAPE(
            public_out,
            batch_size,
            canonical_num_chunks,
            seqlen_q_ori,
            num_heads_q,
            head_size_v_int);
        CHECK_CONTIGUOUS(public_out);
        out = public_out.select(1, 0).view(
            {batch_size, q_seq_per_hk, num_heads, head_size_v_int});
        // The final-output pointer is unused when leaves are returned. Keep a
        // correctly shaped small LSE tensor so the shared parameter setup
        // remains valid without allocating another B*Q*H*Dv output.
        softmax_lse = torch::empty(
            {batch_size, num_heads, q_seq_per_hk}, opts.dtype(at::kFloat));
    } else if (use_preallocated_output) {
        // With one KV head the kernel's [B, Q * H, 1, Dv] row order is
        // exactly the contiguous [B, Q, H, Dv] public order. The same is true
        // for native LSE [B, 1, Q * H] and public [B, Q, H].
        TORCH_CHECK(
            num_heads_k == 1,
            "preallocated dense FP8 output currently requires MQA "
            "(num_heads_k == 1)");
        public_out = out_opt.value();
        public_softmax_lse = softmax_lse_opt.value();
        CHECK_DEVICE(public_out);
        CHECK_DEVICE(public_softmax_lse);
        TORCH_CHECK(
            public_out.get_device() == q.get_device() &&
                public_softmax_lse.get_device() == q.get_device(),
            "preallocated outputs must be on the same CUDA device as q");
        TORCH_CHECK(
            public_out.dtype() == torch::kBFloat16,
            "out must have dtype bfloat16");
        TORCH_CHECK(
            public_softmax_lse.dtype() == torch::kFloat,
            "softmax_lse must have dtype float32");
        CHECK_SHAPE(
            public_out,
            batch_size,
            seqlen_q_ori,
            num_heads_q,
            head_size_v_int);
        CHECK_SHAPE(
            public_softmax_lse,
            batch_size,
            seqlen_q_ori,
            num_heads_q);
        CHECK_CONTIGUOUS(public_out);
        CHECK_CONTIGUOUS(public_softmax_lse);
        out = public_out.view(
            {batch_size, q_seq_per_hk, num_heads, head_size_v_int});
        softmax_lse =
            public_softmax_lse.view({batch_size, num_heads, q_seq_per_hk});
    } else {
        out = torch::empty(
            {batch_size, q_seq_per_hk, num_heads, head_size_v_int},
            opts.dtype(out_type));
        softmax_lse = torch::empty(
            {batch_size, num_heads, q_seq_per_hk},
            opts.dtype(at::kFloat));
    }
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(softmax_lse);

    // Set up parameters for the dense FP8 kernel
    DecodingParams_fp8 params = {};
    // Set the sizes.
    params.b = batch_size;
    params.s_q = seqlen_q_ori;
    params.q_seq_per_hk = q_seq_per_hk;
    params.seqlens_k_ptr = seqlens_k.data_ptr<int>();
    params.h_q = num_heads_q;
    params.h_k = num_heads_k;
    params.num_blocks = num_blocks;
    params.q_head_per_hk = num_q_heads_per_hk;
    params.is_causal = is_causal;
    params.d = head_size_k;
    params.d_v = head_size_v_int;
    params.scale_softmax = static_cast<float>(softmax_scale);
    params.scale_softmax_log2 = float(static_cast<float>(softmax_scale) * M_LOG2E);
    params.topk = -1; // Dense attention

    // FP8-specific parameters
    params.h_h_k_ratio = 1;
    params.descale_q_ptr = reinterpret_cast<float *>(descale_q.value().data_ptr());
    params.descale_k_ptr = reinterpret_cast<float *>(descale_k.value().data_ptr());

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.indices_ptr = nullptr;
    params.softmax_lse_ptr = softmax_lse.data_ptr();

    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(1);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(2);
    params.o_head_stride = out.stride(-2);
    params.indices_batch_stride = 0;
    params.indices_row_stride = 0;

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;
    params.causal_seqlens_k_ptr =
        causal_seqlens_k.has_value()
            ? causal_seqlens_k.value().data_ptr<int>()
            : nullptr;
    params.causal_seqlens_k_batch_stride =
        causal_seqlens_k.has_value() ? causal_seqlens_k.value().stride(0) : 0;
    params.q_lens_ptr =
        q_lens.has_value() ? q_lens.value().data_ptr<int>() : nullptr;

    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    params.num_splits_ptr = num_splits.data_ptr<int>();
    params.canonical_num_chunks = canonical_num_chunks;
    params.canonical_return_leaves = canonical_return_leaves;

    // Set up accumulation tensors
    const int total_num_splits = canonical_return_leaves
        ? batch_size * canonical_num_chunks
        : batch_size + params.num_sm_parts;
    at::Tensor softmax_lse_accum = torch::empty({total_num_splits, num_heads, q_seq_per_hk}, opts.dtype(at::kFloat));
    at::Tensor out_accum = use_preallocated_leaf_output
        ? public_out.view(
              {total_num_splits, num_heads, q_seq_per_hk, head_size_v_int})
        : torch::empty(
              {total_num_splits, num_heads, q_seq_per_hk, head_size_v_int},
              opts.dtype(at::kFloat));
    CHECK_CONTIGUOUS(softmax_lse_accum);
    CHECK_CONTIGUOUS(out_accum);
    params.total_num_splits = total_num_splits;
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Call the actual kernel implementation
#ifdef FLASH_MLA_DISABLE_FP8
    TORCH_CHECK(false, "FlashMLA is compiled with -DFLASH_MLA_DISABLE_FP8. Please remove this flag from your environment and re-compile FlashMLA.");
#else
    if (use_fp32_output) {
        run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, float, 576>(params, stream);
    } else {
        run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, 576>(params, stream);
    }
#endif

    if (canonical_return_leaves) {
        at::Tensor leaf_out =
            out_accum
                .view({batch_size,
                       canonical_num_chunks,
                       num_heads_k,
                       seqlen_q_ori,
                       num_q_heads_per_hk,
                       head_size_v_int})
                .permute({0, 1, 3, 2, 4, 5})
                .reshape({batch_size,
                          canonical_num_chunks,
                          seqlen_q_ori,
                          num_heads_q,
                          head_size_v_int});
        at::Tensor leaf_lse =
            softmax_lse_accum
                .view({batch_size,
                       canonical_num_chunks,
                       num_heads_k,
                       seqlen_q_ori,
                       num_q_heads_per_hk})
                .permute({0, 1, 2, 4, 3})
                .reshape({batch_size,
                          canonical_num_chunks,
                          num_heads_q,
                          seqlen_q_ori});
        return {leaf_out, leaf_lse};
    }

    if (use_preallocated_output) {
        return {public_out, public_softmax_lse};
    }

    // Reshape outputs back to the legacy public format.
    out = out.view({batch_size, seqlen_q_ori, num_q_heads_per_hk, num_heads_k, head_size_v_int}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_q, head_size_v_int});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk}).transpose(2, 3)
            .reshape({batch_size, num_heads_q, seqlen_q_ori});

    return {out, softmax_lse};
}

} // namespace

std::vector<at::Tensor>
fwd_kvcache_mla_fp8(
    const at::Tensor &q_input,
    const at::Tensor &kcache,
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const std::optional<at::Tensor> &causal_seqlens_k,
    const std::optional<at::Tensor> &q_lens,
    const std::optional<at::Tensor> &descale_q,
    const std::optional<at::Tensor> &descale_k) {
    return fwd_kvcache_mla_fp8_impl(
        q_input,
        kcache,
        head_size_v,
        seqlens_k,
        block_table,
        softmax_scale,
        is_causal,
        tile_scheduler_metadata,
        num_splits,
        causal_seqlens_k,
        q_lens,
        descale_q,
        descale_k,
        false,
        0,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

std::vector<at::Tensor>
fwd_kvcache_mla_fp8_fp32(
    const at::Tensor &q_input,
    const at::Tensor &kcache,
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const std::optional<at::Tensor> &causal_seqlens_k,
    const std::optional<at::Tensor> &q_lens,
    const std::optional<at::Tensor> &descale_q,
    const std::optional<at::Tensor> &descale_k) {
    return fwd_kvcache_mla_fp8_impl(
        q_input,
        kcache,
        head_size_v,
        seqlens_k,
        block_table,
        softmax_scale,
        is_causal,
        tile_scheduler_metadata,
        num_splits,
        causal_seqlens_k,
        q_lens,
        descale_q,
        descale_k,
        true,
        0,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

std::vector<at::Tensor>
fwd_kvcache_mla_fp8_canonical(
    const at::Tensor &q_input,
    const at::Tensor &kcache,
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const int64_t canonical_num_chunks,
    const std::optional<at::Tensor> &causal_seqlens_k,
    const std::optional<at::Tensor> &q_lens,
    const std::optional<at::Tensor> &descale_q,
    const std::optional<at::Tensor> &descale_k) {
    TORCH_CHECK(
        canonical_num_chunks <= std::numeric_limits<int>::max(),
        "canonical_num_chunks is too large");
    return fwd_kvcache_mla_fp8_impl(
        q_input,
        kcache,
        head_size_v,
        seqlens_k,
        block_table,
        softmax_scale,
        is_causal,
        tile_scheduler_metadata,
        num_splits,
        causal_seqlens_k,
        q_lens,
        descale_q,
        descale_k,
        true,
        static_cast<int>(canonical_num_chunks),
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

std::vector<at::Tensor>
fwd_kvcache_mla_fp8_canonical_leaves(
    const at::Tensor &q_input,
    const at::Tensor &kcache,
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const int64_t canonical_num_chunks,
    const std::optional<at::Tensor> &causal_seqlens_k,
    const std::optional<at::Tensor> &descale_q,
    const std::optional<at::Tensor> &descale_k) {
    TORCH_CHECK(
        canonical_num_chunks <= std::numeric_limits<int>::max(),
        "canonical_num_chunks is too large");
    return fwd_kvcache_mla_fp8_impl(
        q_input,
        kcache,
        head_size_v,
        seqlens_k,
        block_table,
        softmax_scale,
        is_causal,
        tile_scheduler_metadata,
        num_splits,
        causal_seqlens_k,
        std::nullopt,
        descale_q,
        descale_k,
        true,
        static_cast<int>(canonical_num_chunks),
        true,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

std::vector<at::Tensor>
fwd_kvcache_mla_fp8_canonical_leaves_out(
    const at::Tensor &q_input,
    const at::Tensor &kcache,
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const int64_t canonical_num_chunks,
    const std::optional<at::Tensor> &causal_seqlens_k,
    const std::optional<at::Tensor> &descale_q,
    const std::optional<at::Tensor> &descale_k,
    const at::Tensor &leaf_out) {
    TORCH_CHECK(
        canonical_num_chunks <= std::numeric_limits<int>::max(),
        "canonical_num_chunks is too large");
    return fwd_kvcache_mla_fp8_impl(
        q_input,
        kcache,
        head_size_v,
        seqlens_k,
        block_table,
        softmax_scale,
        is_causal,
        tile_scheduler_metadata,
        num_splits,
        causal_seqlens_k,
        std::nullopt,
        descale_q,
        descale_k,
        true,
        static_cast<int>(canonical_num_chunks),
        true,
        std::nullopt,
        std::nullopt,
        leaf_out);
}

std::vector<at::Tensor>
fwd_kvcache_mla_fp8_out(
    const at::Tensor &q_input,
    const at::Tensor &kcache,
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const std::optional<at::Tensor> &causal_seqlens_k,
    const std::optional<at::Tensor> &q_lens,
    const std::optional<at::Tensor> &descale_q,
    const std::optional<at::Tensor> &descale_k,
    const at::Tensor &out,
    const at::Tensor &softmax_lse) {
    return fwd_kvcache_mla_fp8_impl(
        q_input,
        kcache,
        head_size_v,
        seqlens_k,
        block_table,
        softmax_scale,
        is_causal,
        tile_scheduler_metadata,
        num_splits,
        causal_seqlens_k,
        q_lens,
        descale_q,
        descale_k,
        false,
        0,
        false,
        out,
        softmax_lse,
        std::nullopt);
}

std::vector<at::Tensor>
get_mla_decoding_metadata_dense_fp8(
    at::Tensor &seqlens_k,
    const int64_t num_heads_per_head_k,
    const int64_t num_heads_k
) {
    int num_heads_per_head_k_int = static_cast<int>(num_heads_per_head_k);
    int num_heads_k_int = static_cast<int>(num_heads_k);
    // This should match the logic in the MLA kernel.
    static constexpr int block_size_m = 64;
    static constexpr int block_size_n = 64;
    static constexpr int fixed_overhead_num_blocks = 5;
    CHECK_DEVICE(seqlens_k);
    TORCH_CHECK(seqlens_k.is_contiguous());
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);
    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data_ptr<int>();
    auto options = seqlens_k.options();
    auto dprops = at::cuda::getCurrentDeviceProperties();
    int sm_count = dprops->multiProcessorCount;
    TORCH_CHECK(batch_size < 4096, "batch size must be smaller than 4096");
    TORCH_CHECK(num_heads_k_int > 0, "num_heads_k must be positive");
    TORCH_CHECK(
        num_heads_per_head_k_int > 0,
        "num_q_tokens_per_head_k must be positive");
    int num_m_blocks =
        cutlass::ceil_div(num_heads_per_head_k_int, block_size_m);
    int num_sm_parts = std::max(
        1, sm_count / num_heads_k_int / num_m_blocks);
    // Filling the last SM wave helps single-request latency, where the output
    // is already split. For batches, the extra partition can introduce new
    // partial outputs and make the combine kernel more expensive.
    if (batch_size == 1) {
        num_sm_parts = std::max(
            1,
            cutlass::ceil_div(
                sm_count, num_heads_k_int * num_m_blocks));
    }
    auto tile_scheduler_metadata = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::empty({batch_size + 1}, options);
    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();
    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    get_mla_metadata_func(params, stream);
    return {tile_scheduler_metadata, num_splits};
}


std::vector<at::Tensor>
get_mla_decoding_metadata_canonical_dense_fp8(
    at::Tensor &seqlens_k,
    const int64_t chunk_size,
    const int64_t num_chunks) {
    static constexpr int block_size_n = 64;
    CHECK_DEVICE(seqlens_k);
    CHECK_CONTIGUOUS(seqlens_k);
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);
    TORCH_CHECK(seqlens_k.dim() == 1, "seqlens_k must have shape [batch]");
    TORCH_CHECK(
        chunk_size > 0 && chunk_size % block_size_n == 0,
        "canonical chunk_size must be a positive multiple of 64");
    TORCH_CHECK(
        num_chunks == 1 || num_chunks == 2 || num_chunks == 4 ||
            num_chunks == 8,
        "canonical num_chunks must be one of 1, 2, 4, or 8");
    TORCH_CHECK(
        chunk_size <= std::numeric_limits<int>::max(),
        "canonical chunk_size is too large");

    const int batch_size = seqlens_k.size(0);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(
        batch_size * num_chunks <= 160,
        "canonical prototype supports at most 160 batch-chunk partitions");
    auto options = seqlens_k.options();
    auto tile_scheduler_metadata = torch::empty(
        {batch_size * num_chunks, TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::empty({batch_size + 1}, options);

    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    Mla_canonical_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k.data_ptr<int>();
    params.tile_scheduler_metadata_ptr =
        tile_scheduler_metadata.data_ptr<int>();
    params.num_splits_ptr = num_splits.data_ptr<int>();
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.chunk_size = static_cast<int>(chunk_size);
    params.num_chunks = static_cast<int>(num_chunks);
    get_mla_canonical_metadata_func(params, stream);
    return {tile_scheduler_metadata, num_splits};
}
