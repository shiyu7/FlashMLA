/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Keep a self-contained fp8 decode params struct so this extension can stay
// compatible when upstream refactors csrc/params.h.
struct DecodingParams_fp8 {
    using index_t = int64_t;

    int b;              // batch size
    int s_q;
    int q_seq_per_hk;   // Number of q(s) per KV head
    int d, d_v;         // K/V dimension
    int h_q, h_k;       // Number of Q/K heads
    int num_blocks;     // Number of blocks in total
    int q_head_per_hk;  // Number of q heads per KV head
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int topk;

    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ o_ptr;
    void* __restrict__ softmax_lse_ptr;
    int* __restrict__ indices_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;
    index_t indices_batch_stride;
    index_t indices_row_stride;

    int* __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int* __restrict__ seqlens_k_ptr;
    // Optional per-query right boundary in the rank-local KV sequence.
    //
    // Shape: [b, s_q], int32. Entry [batch, query] is the number of local
    // keys visible to that query (an exclusive right boundary). When null,
    // the kernel uses the standard contiguous bottom-right causal mask.
    //
    // Keeping this local-layout contract free of cp_size/cp_rank makes the
    // operator usable by cyclic DCP and by other externally planned KV
    // layouts. The caller owns the global-to-local mapping.
    int* __restrict__ causal_seqlens_k_ptr;
    index_t causal_seqlens_k_batch_stride;
    // Optional valid query-prefix length for a fixed physical Q dimension.
    // Shape: [b], int32. Rows whose query index is >= q_lens[batch] are
    // neutralized by the final split-combine kernel.
    int* __restrict__ q_lens_ptr;

    int* __restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int* __restrict__ num_splits_ptr;

    int total_num_splits;
    void* __restrict__ softmax_lseaccum_ptr;
    void* __restrict__ oaccum_ptr;

    // Opt-in R14 path. A positive value means that every scheduler partition
    // is one fixed-size canonical leaf and the final combine must use the
    // adjacent balanced tree instead of the regular flat split reduction.
    // The current fused prototype accepts 2, 4, or 8 leaves per batch.
    int canonical_num_chunks;
    bool canonical_return_leaves;

    int h_h_k_ratio;
    float* __restrict__ descale_q_ptr = nullptr;
    float* __restrict__ descale_k_ptr = nullptr;
};

static constexpr int TileSchedulerMetaDataSize = 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(DecodingParams_fp8 &params, cudaStream_t stream);

struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};
void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);

struct Mla_canonical_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int chunk_size;
    int num_chunks;
};
void get_mla_canonical_metadata_func(
    Mla_canonical_metadata_params &params,
    cudaStream_t stream);
