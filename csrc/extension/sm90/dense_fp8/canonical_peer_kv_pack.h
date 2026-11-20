#pragma once

#include <torch/all.h>

// Reconstruct one contiguous 1024-token canonical chunk per rank from the
// existing token-cyclic DCP KV layout. The staging and flag tensors are
// symmetric-memory allocations rendezvoused across exactly eight SM90 ranks.
void canonical_peer_pack_token_cyclic_kv(
    const at::Tensor& staging_ptrs,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& local_cache,
    const at::Tensor& block_table,
    const at::Tensor& local_seqlens,
    const at::Tensor& local_staging,
    const at::Tensor& canonical_cache);
