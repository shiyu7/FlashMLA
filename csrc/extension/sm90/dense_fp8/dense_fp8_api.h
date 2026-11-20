#pragma once

#include <optional>
#include <vector>

#include <torch/all.h>

// SM90 dense-FP8 MLA decode.
//
// seqlens_k and block_table describe the rank-local compact KV cache.
// causal_seqlens_k, when present, has shape [batch_size, seqlen_q] and stores
// the exclusive rank-local causal boundary for every query. This lets an
// external runtime implement cyclic decode context parallelism without
// teaching FlashMLA about process groups or a particular sharding policy.
// q_lens, when present, has shape [batch_size] and marks the valid query
// prefix within a fixed physical seqlen_q <= 8. Invalid suffix rows are
// returned as zero output and negative-infinity LSE.
//
// The returned LSE is natural-log based and represents only this local KV
// shard. A distributed caller must merge (out, lse) across ranks.
std::vector<at::Tensor> fwd_kvcache_mla_fp8(
    const at::Tensor& q,
    const at::Tensor& kcache,
    int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    const std::optional<at::Tensor>& causal_seqlens_k,
    const std::optional<at::Tensor>& q_lens,
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k);

// FP32-output variant for distributed DCP merge.
//
// The local attention state remains float32 until all rank-local states have
// been LSE-merged.  This avoids the rank-local bfloat16 rounding performed by
// the regular API.  LSE keeps the regular [batch_size, num_heads_q,
// seqlen_q] float32 layout.
std::vector<at::Tensor> fwd_kvcache_mla_fp8_fp32(
    const at::Tensor& q,
    const at::Tensor& kcache,
    int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    const std::optional<at::Tensor>& causal_seqlens_k,
    const std::optional<at::Tensor>& q_lens,
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k);

// R14 fused canonical-chunk prototype. The supplied metadata must come from
// get_mla_decoding_metadata_canonical_dense_fp8. Each scheduler partition
// produces one FP32 leaf and the epilogue applies a fixed adjacent tree.
std::vector<at::Tensor> fwd_kvcache_mla_fp8_canonical(
    const at::Tensor& q,
    const at::Tensor& kcache,
    int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    int64_t canonical_num_chunks,
    const std::optional<at::Tensor>& causal_seqlens_k,
    const std::optional<at::Tensor>& q_lens,
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k);

// Leaf-export ABI for R14-P2. Returns FP32 tensors shaped
// [B, chunks, Q, Hq, Dv] and [B, chunks, Hq, Q] without applying a local
// tree, so distributed code can restore global chunk-id order first.
std::vector<at::Tensor> fwd_kvcache_mla_fp8_canonical_leaves(
    const at::Tensor& q,
    const at::Tensor& kcache,
    int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    int64_t canonical_num_chunks,
    const std::optional<at::Tensor>& causal_seqlens_k,
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k);

// One-chunk MQA leaf export that writes FP32 output directly into caller-owned
// [B, 1, Q, Hq, Dv] storage. LSE is still returned as [B, 1, Hq, Q].
std::vector<at::Tensor> fwd_kvcache_mla_fp8_canonical_leaves_out(
    const at::Tensor& q,
    const at::Tensor& kcache,
    int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    int64_t canonical_num_chunks,
    const std::optional<at::Tensor>& causal_seqlens_k,
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k,
    const at::Tensor& leaf_out);

// Allocation-free output variant for the DCP2 symmetric-memory path.
//
// Kimi-K3 uses one KV head. For that MQA layout, the kernel-native flattened
// query order aliases contiguous public tensors without a copy:
//   out:         [batch_size, seqlen_q, num_heads_q, head_size_v], bfloat16
//   softmax_lse: [batch_size, seqlen_q, num_heads_q], float32 (natural log)
//
// The regular API above keeps returning LSE as [batch_size, num_heads_q,
// seqlen_q]. Keeping this out-variant explicit prevents an accidental
// transpose/copy before publishing a symmetric-memory slot.
std::vector<at::Tensor> fwd_kvcache_mla_fp8_out(
    const at::Tensor& q,
    const at::Tensor& kcache,
    int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    const std::optional<at::Tensor>& causal_seqlens_k,
    const std::optional<at::Tensor>& q_lens,
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k,
    const at::Tensor& out,
    const at::Tensor& softmax_lse);

std::vector<at::Tensor> get_mla_decoding_metadata_dense_fp8(
    at::Tensor& seqlens_k,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k);

std::vector<at::Tensor> get_mla_decoding_metadata_canonical_dense_fp8(
    at::Tensor& seqlens_k,
    int64_t chunk_size,
    int64_t num_chunks);
