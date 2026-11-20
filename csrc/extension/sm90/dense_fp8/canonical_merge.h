#pragma once

#include <torch/all.h>

// Merge two FP32 normalized attention states in a fixed left/right order.
// Output and LSE tensors may alias the left inputs, which lets each owner in a
// distributed balanced tree update its surviving state in place.
void canonical_merge_pair_out(
    const at::Tensor& left_out,
    const at::Tensor& left_lse,
    const at::Tensor& right_out,
    const at::Tensor& right_lse,
    const at::Tensor& merged_out,
    const at::Tensor& merged_lse);

// Wait until this rank's parent consumed the previous tree epoch before the
// caller overwrites its persistent symmetric-memory state.
void canonical_peer_tree_prepare(
    const at::Tensor& flag_ptrs,
    int64_t rows,
    int64_t dcp_rank);

// Publish a symmetric-memory leaf and execute the fixed 8-rank adjacent tree.
// Rank zero owns the final in-place state; other ranks return after their
// parent has enough information to consume their current stage.
void canonical_peer_tree_merge(
    const at::Tensor& output_ptrs,
    const at::Tensor& lse_ptrs,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& local_out,
    const at::Tensor& local_lse);
