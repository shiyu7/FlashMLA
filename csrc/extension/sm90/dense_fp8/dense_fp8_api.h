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

std::vector<at::Tensor> get_mla_decoding_metadata_dense_fp8(
    at::Tensor& seqlens_k,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k);
