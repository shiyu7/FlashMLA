#pragma once

#include <torch/all.h>

// Epoch gate for the eight-rank head-sharded canonical tree. Each row waits
// until its unique parent consumed the previous replay before it is replaced.
void canonical_peer_reduce_scatter_prepare(
    const at::Tensor& flag_ptrs,
    int64_t rows,
    int64_t dcp_rank);

// Reduce eight fixed global-chunk FP32 states with the canonical adjacent tree
// while placing the final contiguous H/8 head shard on its owning rank.
void canonical_peer_reduce_scatter(
    const at::Tensor& output_ptrs,
    const at::Tensor& lse_ptrs,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& local_out,
    const at::Tensor& local_lse);

// Equivalent fixed tree, but write the final owned H/8 shard directly into
// compact caller-owned BQHD/BHQ tensors instead of leaving it strided inside
// the full symmetric leaf buffer.
void canonical_peer_reduce_scatter_compact(
    const at::Tensor& output_ptrs,
    const at::Tensor& lse_ptrs,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& local_out,
    const at::Tensor& local_lse,
    const at::Tensor& compact_out,
    const at::Tensor& compact_lse);
