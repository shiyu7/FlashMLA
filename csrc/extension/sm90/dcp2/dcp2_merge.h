#pragma once

#include <torch/all.h>

// Wait until the owner of every output row has consumed the previous epoch.
// This must run immediately before FlashMLA overwrites its local symmetric
// output/LSE buffers.
void dcp2_prepare_attention(
    const at::Tensor& flag_ptrs,
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t dcp_rank);

// Publish this rank's FlashMLA state, wait for both rank publications, and
// merge natural-log LSE plus bfloat16 output directly from peer mappings.
// Only the dcp_rank-owned half of the query heads is materialized.
void dcp2_merge_attention(
    const at::Tensor& rank0_out,
    const at::Tensor& rank1_out,
    const at::Tensor& rank0_lse,
    const at::Tensor& rank1_lse,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& merged_out,
    const at::Tensor& merged_lse);
