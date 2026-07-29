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
| `descale_q`, `descale_k` | `[1]` | float32 quantization descales |

`causal_seqlens_k` must be nondecreasing along the query dimension and each
entry must be between zero and the corresponding `local_cache_seqlens`.
When it is supplied, causal masking is enabled even for `Q == 1`. This is
necessary for ranks that do not own the newest global token.

The outputs are:

- `out`: `[B, Q, Hq, 512]`, BF16 rank-local normalized attention output.
- `local_lse`: `[B, Hq, Q]`, FP32 natural-log LSE for the local shard.

An empty local shard returns `out = 0` and `local_lse = -inf`, the neutral
state for cross-rank merging.

## Cyclic DCP

For cyclic placement, global token `i` belongs to rank `i % cp_size`.
FlashMLA provides helpers for the rank-local lengths and the final merge:

```python
local_cache_seqlens = flash_mla.cyclic_local_length(
    global_cache_seqlens, cp_size, cp_rank
).to(torch.int32)

local_causal_seqlens = flash_mla.make_cyclic_local_causal_seqlens(
    global_cache_seqlens, q_len, cp_size, cp_rank
)

global_out, global_lse = flash_mla.merge_local_attention(
    outputs_stacked_on_rank_dim,
    lses_stacked_on_rank_dim,
)
```

The merge uses natural exponentials. A runtime whose existing attention state
uses base-2 LSE must convert this kernel's LSE with `log2(e)` before using its
base-2 merge.

## Build and validation

On an H100/H800 with CUDA 12.8, disable the SM100 sources:

```bash
FLASH_MLA_DISABLE_SM100=1 pip install -v .
pytest -q tests/test_dense_fp8_dcp.py
python benchmark/bench_dense_fp8_dcp.py --cp-sizes 1,2,4,8 --q-len 4
```

The benchmark simulates ranks sequentially on one GPU. Its `critical_ms`
metric is the maximum local-kernel time and estimates ideal parallel rank
execution. It intentionally excludes Q distribution, cross-rank LSE/output
communication, and process-launch overhead; it reports the local merge cost
separately.
