# SM90 dense FP8 MLA with external DCP

This extension adds a dense FP8 MLA decode path for Hopper and a minimal
contract for decode context parallelism (DCP). The CUDA operator owns only
rank-local attention. KV placement, communication, and cross-rank reduction
remain runtime responsibilities.

The dense FP8 kernel was derived from
[FlashMLA PR #54](https://github.com/deepseek-ai/FlashMLA/pull/54) and the
version maintained in the SGLang FlashMLA fork. Its local causal-boundary
input and empty-shard behavior are added here for cyclic DCP and multi-token
prediction (MTP).

## Supported configuration

| Property | Supported value |
| --- | --- |
| GPU | SM90 |
| Q and KV dtype | `torch.float8_e4m3fn` |
| output dtype | `torch.bfloat16` |
| Q/K head dimension | 576 |
| V/output head dimension | 512 |
| page size | 64 |
| KV heads | MQA/GQA; Q heads must be divisible by KV heads |
| MTP query length | fixed dense `q_len`, initially tested for 1, 2, 4, 8 |
| KV sharding | externally planned; cyclic helpers are included |

This is not the mixed FP8/BF16 656-byte sparse-cache layout. Every one of the
576 Q/K elements is E4M3, with one scalar `descale_q` and `descale_k` per
operator invocation.

## Operator contract

```python
out, local_lse = flash_mla.flash_mla_dense_fp8(
    q,
    local_k_cache,
    local_block_table,
    local_cache_seqlens,
    head_dim_v=512,
    tile_scheduler_metadata=metadata,
    num_splits=num_splits,
    softmax_scale=576**-0.5,
    causal=True,
    causal_seqlens_k=local_causal_seqlens,
    q_lens=q_lens,
    descale_q=descale_q,
    descale_k=descale_k,
)
```

| Input | Shape | Meaning |
| --- | --- | --- |
| `q` | `[B, Q, Hq, 576]` | FP8 query; last dimension contiguous |
| `local_k_cache` | `[pages, 64, Hkv, 576]` | FP8 rank-local paged KV |
| `local_block_table` | `[B, max_local_pages]` | int32 local page IDs |
| `local_cache_seqlens` | `[B]` | int32 number of local KV tokens |
| `causal_seqlens_k` | optional `[B, Q]` | int32 exclusive local KV boundary for each query |
| `q_lens` | optional `[B]` | contiguous CUDA int32 valid-query prefix length inside fixed physical `Qmax <= 8` |
| `descale_q`, `descale_k` | `[1]` | float32 quantization descales |

`causal_seqlens_k` must be nondecreasing along the query dimension and each
entry must be between zero and the corresponding `local_cache_seqlens`.
When it is supplied, causal masking is enabled even for `Q == 1`. This is
necessary for ranks that do not own the newest global token.

For continuous batching, callers may keep `q` physically rectangular as
`[B, Qmax, Hq, 576]` and pass `q_lens[b]` for the logical query count of
each request. Only the prefix `q[b, :q_lens[b]]` is valid. FlashMLA writes
the padded suffix directly as `out = 0` and `lse = -inf`; it does not
overload `causal_seqlens_k` as a query-validity mask. The current
first-version contract requires `0 <= q_lens[b] <= Qmax <= 8`.
The attention kernel still schedules the physical `Qmax`; `q_lens` only
guarantees correct neutral outputs for padding in this first version.
It is a separate validity input: callers must still pass
`causal_seqlens_k` when per-query causal KV boundaries are required.

For cyclic DCP, pass the same tensor to
`make_cyclic_local_causal_seqlens(..., q_lens=q_lens)`. This aligns each
request's valid prefix with its own trailing query window instead of
deriving every request from the physical `Qmax`.

The outputs are:

- `out`: `[B, Q, Hq, 512]`, BF16 rank-local normalized attention output.
- `local_lse`: `[B, Hq, Q]`, FP32 natural-log LSE for the local shard.

An empty local shard returns `out = 0` and `local_lse = -inf`, the neutral
state for cross-rank merging.

### Preallocated output ABI

The DCP2 fast path uses an explicit MQA-only out-variant:

```python
flash_mla.flash_mla_dense_fp8_out(
    q,
    local_k_cache,
    local_block_table,
    local_cache_seqlens,
    512,
    metadata,
    num_splits,
    symmetric_out,       # contiguous [B, Q, H, 512], bfloat16
    symmetric_lse,       # contiguous [B, Q, H], float32
    causal=True,
    causal_seqlens_k=local_causal_seqlens,
    q_lens=q_lens,
)
```

Both tensors are written in the kernel's native flattened-query order. No
post-kernel staging copy is performed. The native `BQH` LSE layout is
intentionally different from the legacy API's `BHQ` return layout.
`q_lens` has the same `[B]` valid-prefix contract as the allocating API; the
out-variant writes every padded suffix row directly as zero output and
negative-infinity LSE.

## Cyclic DCP

For cyclic placement, global token `i` belongs to rank `i % cp_size`.
FlashMLA provides helpers for the rank-local lengths and the final merge:

```python
local_cache_seqlens = flash_mla.cyclic_local_length(
    global_cache_seqlens, cp_size, cp_rank
).to(torch.int32)

local_causal_seqlens = flash_mla.make_cyclic_local_causal_seqlens(
    global_cache_seqlens,
    q_len,
    cp_size,
    cp_rank,
    q_lens=q_lens,
)

global_out, global_lse = flash_mla.merge_local_attention(
    outputs_stacked_on_rank_dim,
    lses_stacked_on_rank_dim,
)
```

The merge uses natural exponentials. A runtime whose existing attention state
uses base-2 LSE must convert this kernel's LSE with `log2(e)` before using its
base-2 merge.

## DCP2 NVLink fast path

`create_dcp2_attention_workspace` collectively allocates persistent symmetric
output, LSE, and epoch buffers. The intended call order is:

```python
workspace = flash_mla.create_dcp2_attention_workspace(
    max_batch_size,
    max_q_len,       # <= 8
    num_heads,       # fixed for this workspace
    512,
    dcp_process_group,
)

local_out, local_lse = workspace.prepare(B, Q, H, 512)
flash_mla.flash_mla_dense_fp8_out(
    q, local_k_cache, block_table, local_seqlens, 512,
    metadata, num_splits, local_out, local_lse,
    causal=True, causal_seqlens_k=local_causal_seqlens,
    q_lens=q_lens,
)
out_head_shard, global_lse_head_shard = workspace.merge(B, Q, H, 512)
```

`workspace.run_flash_mla_dense_fp8(...)` combines those three calls and
forwards `q_lens` unchanged to the out-variant:

```python
out_head_shard, global_lse_head_shard = (
    workspace.run_flash_mla_dense_fp8(
        q,
        local_k_cache,
        block_table,
        local_seqlens,
        512,
        metadata,
        num_splits,
        causal=True,
        causal_seqlens_k=local_causal_seqlens,
        q_lens=q_lens,
        descale_q=descale_q,
        descale_k=descale_k,
    )
)
```

The physical `Q` remains part of the workspace's fixed graph shape while the
values in `q_lens` may change between graph replays, provided the tensor
storage and shape remain stable and `causal_seqlens_k` is updated to match.
Each `B x Q x H` row owns a pair of device-side 64-bit counters:

1. `prepare` waits until the row owner consumed the previous producer epoch.
2. FlashMLA overwrites the symmetric row directly.
3. The fused kernel publishes both rank-local states, waits for both producer
   epochs, merges natural-log LSE and BF16 output from peer mappings, and
   acknowledges consumption.

Only the rank owning that half of the query heads materializes the merged row,
so the result matches an LSE correction plus output reduce-scatter. Device-side
epochs increase on every replay; the captured graph does not depend on a
host-updated buffer index.

The workspace capacity may exceed the active shape, but its first
`(B, Q, H, Dv)` use locks the compact row geometry. Use a separate workspace
for each captured graph shape; changing compact strides while an older peer
epoch may still be consumed is intentionally rejected.

This implementation is deliberately limited to DCP2, a single node with CUDA
peer access, SM90, and `Q <= 8`. CUDA peer-access validation cannot distinguish
NVLink from PCIe, so the serving topology policy must reserve the fast path for
an NVLink pair. DCP4/8 always use `nccl_merge_attention`; they do not extend the
DCP2 kernel into an all-peer-read algorithm.

`merge_dcp_attention` is the routing helper. It uses the peer workspace only
for eligible DCP2 shapes and keeps NCCL all-gather plus reduce-scatter as the
fallback. An optional `max_peer_payload_bytes` may be supplied after measuring
the target system.

## Build and validation

On an H100/H800 with CUDA 12.8, disable the SM100 sources:

```bash
FLASH_MLA_DISABLE_SM100=1 pip install -v .
pytest -q tests/test_dense_fp8_dcp.py
python benchmark/bench_dense_fp8_dcp.py --cp-sizes 1,2,4,8 --q-len 4

torchrun --standalone --nproc-per-node=2 -m pytest -q \
    tests/test_dcp2_peer_merge.py

torchrun --standalone --nproc-per-node=2 \
    benchmark/bench_dcp2_crossover.py \
    --batches 1,2,4,8,16,32 --q-lens 1,2,4,8 \
    --heads 32,64,128 --head-dims-v 128,256,512 \
    --csv dcp2_crossover.csv
```

The benchmark simulates ranks sequentially on one GPU. Its `critical_ms`
metric is the maximum local-kernel time and estimates ideal parallel rank
execution. It intentionally excludes Q distribution, cross-rank LSE/output
communication, and process-launch overhead; it reports the local merge cost
separately.

The two-GPU crossover benchmark times the device-epoch prepare plus fused peer
merge against the NCCL LSE-all-gather/correction/output-reduce-scatter path. It
reports every real `(B, Q, H, Dv)` shape and summarizes peer-winning batches
per exact `(Q, H, Dv)` tuple instead of collapsing the policy to payload bytes.
