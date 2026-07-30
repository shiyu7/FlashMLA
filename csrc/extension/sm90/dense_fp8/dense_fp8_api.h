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
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k);

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
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k,
    const at::Tensor& out,
    const at::Tensor& softmax_lse);

std::vector<at::Tensor> get_mla_decoding_metadata_dense_fp8(
    at::Tensor& seqlens_k,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k);
