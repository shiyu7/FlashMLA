"""Accuracy oracles and deterministic inputs for dense FP8 DCP evaluation.

This module is intentionally outside the :mod:`flash_mla` package.  It is a
development dependency used by tests and benchmarks, not a public runtime API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch


PAGE_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512


@dataclass(frozen=True)
class AccuracyMetrics:
    """Numerical and invariant metrics for one end-to-end DCP case."""

    output_abs_p50: float
    output_abs_p99: float
    output_abs_max: float
    output_rel_p99: float
    output_nrmse: float
    output_cosine_p01: float
    output_cosine_min: float
    lse_abs_p50: float
    lse_abs_p99: float
    lse_abs_max: float
    lse_inf_mask_mismatches: int
    output_nan_count: int
    lse_nan_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), q).item())


def fp64_merge_reference(
    local_outputs: torch.Tensor,
    local_lses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge native BQHD/BQH local states with FP64 accumulation.

    Args:
        local_outputs: ``[R, B, Q, H, Dv]`` rank-local normalized outputs.
        local_lses: ``[R, B, Q, H]`` natural-log rank-local LSE values.

    Fully empty rows produce the exact neutral state ``out=0, lse=-inf``.
    """
    if local_outputs.ndim != 5 or local_lses.ndim != 4:
        raise ValueError("expected local outputs [R,B,Q,H,Dv] and LSE [R,B,Q,H]")
    if tuple(local_outputs.shape[:-1]) != tuple(local_lses.shape):
        raise ValueError("local output and LSE shapes do not match")

    lses = local_lses.double()
    global_lse = torch.logsumexp(lses, dim=0)
    finite = torch.isfinite(global_lse)
    log_weights = torch.where(
        finite.unsqueeze(0),
        lses - global_lse.unsqueeze(0),
        torch.full_like(lses, -torch.inf),
    )
    weights = torch.exp(log_weights).unsqueeze(-1)
    safe_outputs = torch.nan_to_num(local_outputs.double())
    output = (safe_outputs * weights).sum(dim=0)
    output = torch.where(finite.unsqueeze(-1), output, torch.zeros_like(output))
    return output, global_lse


def global_attention_reference(
    q: torch.Tensor,
    global_kv: torch.Tensor,
    global_seqlens: torch.Tensor,
    q_lens: torch.Tensor,
    *,
    softmax_scale: float,
    descale_q: float,
    descale_k: float,
    head_dim_v: int = HEAD_DIM_V,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the unsharded causal attention reference in FP32.

    The output layouts are kernel-native ``BQHD`` and ``BQH``.  Invalid query
    suffixes use the exact neutral state required by the DCP contract.
    """
    if q.ndim != 4 or global_kv.ndim != 4:
        raise ValueError("q/global_kv must have BQHD/BSHD layouts")
    batch, physical_q, num_heads, head_dim = q.shape
    if global_kv.shape[0] != batch or global_kv.shape[-1] != head_dim:
        raise ValueError("q/global_kv batch or head dimension mismatch")
    num_heads_k = global_kv.shape[2]
    if num_heads % num_heads_k:
        raise ValueError("query heads must be divisible by KV heads")
    if tuple(global_seqlens.shape) != (batch,) or tuple(q_lens.shape) != (batch,):
        raise ValueError("global_seqlens and q_lens must have shape [B]")

    output = torch.zeros(
        batch,
        physical_q,
        num_heads,
        head_dim_v,
        dtype=torch.float32,
        device=q.device,
    )
    lse = torch.full(
        (batch, physical_q, num_heads),
        -torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    heads_per_kv = num_heads // num_heads_k

    for batch_idx in range(batch):
        seqlen = int(global_seqlens[batch_idx].item())
        logical_q = int(q_lens[batch_idx].item())
        if not 0 <= logical_q <= physical_q:
            raise ValueError("q_lens values must satisfy 0 <= q_lens <= Q")
        if logical_q == 0:
            continue
        if seqlen < logical_q:
            raise ValueError("each global sequence must cover its logical queries")

        query = (
            q[batch_idx, :logical_q]
            .float()
            .mul(descale_q)
            .permute(1, 0, 2)
        )
        kv = global_kv[batch_idx, :seqlen].float().mul(descale_k)
        kv_by_head = kv.permute(1, 0, 2).repeat_interleave(
            heads_per_kv, dim=0
        )
        scores = torch.matmul(query, kv_by_head.transpose(1, 2))
        scores.mul_(softmax_scale)
        visible = seqlen - logical_q + torch.arange(
            1, logical_q + 1, dtype=torch.int64, device=q.device
        )
        key_indices = torch.arange(seqlen, device=q.device)
        scores.masked_fill_(
            key_indices.view(1, 1, -1) >= visible.view(1, -1, 1),
            -torch.inf,
        )
        local_lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        local_output = torch.matmul(
            probabilities, kv_by_head[..., :head_dim_v]
        )
        output[batch_idx, :logical_q] = local_output.permute(1, 0, 2)
        lse[batch_idx, :logical_q] = local_lse.permute(1, 0)
    return output, lse


def _sample_tensor(
    shape: Sequence[int],
    distribution: str,
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    if distribution == "normal":
        return torch.randn(
            tuple(shape), device=device, generator=generator
        ).mul_(0.1).clamp_(-1, 1)
    if distribution == "uniform":
        return torch.rand(
            tuple(shape), device=device, generator=generator
        ).mul_(2).sub_(1)
    if distribution == "near_zero":
        return torch.randn(
            tuple(shape), device=device, generator=generator
        ).mul_(0.002).clamp_(-1, 1)
    if distribution == "signed":
        random = torch.rand(tuple(shape), device=device, generator=generator)
        return torch.where(random < 0.5, -torch.ones_like(random), torch.ones_like(random))
    raise ValueError(
        "distribution must be one of normal, uniform, near_zero, signed"
    )


def make_fp8_inputs(
    *,
    global_seqlens: Sequence[int],
    q_len: int,
    num_heads: int,
    num_heads_k: int = 1,
    distribution: str = "normal",
    seed: int = 2026,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic FP8 Q/KV tensors and CUDA int32 sequence lengths."""
    if not global_seqlens or min(global_seqlens) < 0:
        raise ValueError("global_seqlens must be a non-empty non-negative sequence")
    if num_heads % num_heads_k:
        raise ValueError("num_heads must be divisible by num_heads_k")
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch = len(global_seqlens)
    max_seqlen = max(max(global_seqlens), 1)
    q = _sample_tensor(
        (batch, q_len, num_heads, HEAD_DIM),
        distribution,
        generator,
        device,
    ).to(torch.float8_e4m3fn)
    global_kv = _sample_tensor(
        (batch, max_seqlen, num_heads_k, HEAD_DIM),
        distribution,
        generator,
        device,
    ).to(torch.float8_e4m3fn)
    seqlens = torch.tensor(global_seqlens, dtype=torch.int32, device=device)
    return q, global_kv, seqlens


def pack_cyclic_local_cache(
    global_kv: torch.Tensor,
    global_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    *,
    permute_pages: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a global cache into one cyclic rank-local paged cache."""
    import flash_mla

    batch, _, num_heads_k, head_dim = global_kv.shape
    pages_per_batch: list[int] = []
    local_tokens_per_batch: list[torch.Tensor] = []
    for batch_idx in range(batch):
        seqlen = int(global_seqlens[batch_idx].item())
        local_tokens = global_kv[
            batch_idx, cp_rank:seqlen:cp_size
        ].contiguous()
        local_tokens_per_batch.append(local_tokens)
        pages_per_batch.append((local_tokens.shape[0] + PAGE_SIZE - 1) // PAGE_SIZE)

    max_pages = max(max(pages_per_batch), 1)
    block_table = torch.full(
        (batch, max_pages),
        -1,
        dtype=torch.int32,
        device=global_kv.device,
    )
    pages: list[torch.Tensor] = []
    for batch_idx, local_tokens in enumerate(local_tokens_per_batch):
        for page_idx in range(pages_per_batch[batch_idx]):
            page = torch.zeros(
                PAGE_SIZE,
                num_heads_k,
                head_dim,
                dtype=global_kv.dtype,
                device=global_kv.device,
            )
            begin = page_idx * PAGE_SIZE
            end = min(begin + PAGE_SIZE, local_tokens.shape[0])
            page[: end - begin] = local_tokens[begin:end]
            block_table[batch_idx, page_idx] = len(pages)
            pages.append(page)
    if not pages:
        pages.append(
            torch.zeros(
                PAGE_SIZE,
                num_heads_k,
                head_dim,
                dtype=global_kv.dtype,
                device=global_kv.device,
            )
        )
    local_cache = torch.stack(pages)

    if permute_pages and len(pages) > 1:
        generator = torch.Generator(device=global_kv.device)
        generator.manual_seed(seed + cp_rank * 1009)
        permutation = torch.randperm(
            len(pages), generator=generator, device=global_kv.device
        )
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            len(pages), device=global_kv.device
        )
        local_cache = local_cache.index_select(0, permutation)
        valid = block_table >= 0
        block_table[valid] = inverse[block_table[valid].long()].to(torch.int32)

    local_seqlens = flash_mla.cyclic_local_length(
        global_seqlens, cp_size, cp_rank
    ).to(torch.int32)
    return local_cache, block_table, local_seqlens


def run_cyclic_case(
    q: torch.Tensor,
    global_kv: torch.Tensor,
    global_seqlens: torch.Tensor,
    q_lens: torch.Tensor,
    *,
    cp_size: int,
    softmax_scale: float,
    descale_q: float,
    descale_k: float,
    permute_pages: bool = False,
    page_seed: int = 0,
    local_output_dtype: str = "bf16",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run every cyclic rank sequentially and return result plus reference.

    Sequential rank execution is intended for numerical evaluation only.  It
    must not be used as a distributed performance measurement.
    """
    import flash_mla

    if local_output_dtype not in ("bf16", "fp32"):
        raise ValueError("local_output_dtype must be 'bf16' or 'fp32'")

    _, q_len, num_heads, _ = q.shape
    num_heads_k = global_kv.shape[2]
    descale_q_tensor = torch.tensor(
        [descale_q], dtype=torch.float32, device=q.device
    )
    descale_k_tensor = torch.tensor(
        [descale_k], dtype=torch.float32, device=q.device
    )
    local_outputs: list[torch.Tensor] = []
    local_lses_native: list[torch.Tensor] = []
    for cp_rank in range(cp_size):
        local_cache, block_table, local_seqlens = pack_cyclic_local_cache(
            global_kv,
            global_seqlens,
            cp_size,
            cp_rank,
            permute_pages=permute_pages,
            seed=page_seed,
        )
        causal_seqlens = flash_mla.make_cyclic_local_causal_seqlens(
            global_seqlens,
            q_len,
            cp_size,
            cp_rank,
            q_lens=q_lens,
        )
        metadata, num_splits = flash_mla.get_mla_metadata_dense_fp8(
            local_seqlens,
            q_len * num_heads // num_heads_k,
            num_heads_k,
        )
        attention_fn = (
            flash_mla.flash_mla_dense_fp8_fp32
            if local_output_dtype == "fp32"
            else flash_mla.flash_mla_dense_fp8
        )
        local_out, local_lse_bhq = attention_fn(
            q,
            local_cache,
            block_table,
            local_seqlens,
            HEAD_DIM_V,
            metadata,
            num_splits,
            softmax_scale=softmax_scale,
            causal=True,
            causal_seqlens_k=causal_seqlens,
            q_lens=q_lens,
            descale_q=descale_q_tensor,
            descale_k=descale_k_tensor,
        )
        local_outputs.append(local_out)
        local_lses_native.append(local_lse_bhq.transpose(1, 2).contiguous())

    merged_out, merged_lse_bhq = flash_mla.merge_local_attention(
        local_outputs,
        [lse.transpose(1, 2).contiguous() for lse in local_lses_native],
    )
    # Mirror the runtime ABI: accurate DCP keeps local states and the global
    # merge in FP32, then performs one final BF16 cast before the projection.
    if local_output_dtype == "fp32":
        merged_out = merged_out.to(torch.bfloat16)
    reference_out, reference_lse = global_attention_reference(
        q,
        global_kv,
        global_seqlens,
        q_lens,
        softmax_scale=softmax_scale,
        descale_q=descale_q,
        descale_k=descale_k,
    )
    return (
        merged_out,
        merged_lse_bhq.transpose(1, 2).contiguous(),
        reference_out,
        reference_lse,
    )


def compute_accuracy_metrics(
    output: torch.Tensor,
    lse: torch.Tensor,
    output_reference: torch.Tensor,
    lse_reference: torch.Tensor,
    q_lens: torch.Tensor,
    *,
    relative_epsilon: float = 1e-5,
) -> AccuracyMetrics:
    """Summarize numerical error over valid query rows."""
    if output.shape != output_reference.shape or lse.shape != lse_reference.shape:
        raise ValueError("actual and reference shapes must match")
    batch, physical_q, _, _ = output.shape
    valid_rows = torch.arange(
        physical_q, device=output.device
    ).view(1, -1) < q_lens.view(batch, 1)
    output_valid = valid_rows[:, :, None, None].expand_as(output)
    lse_valid = valid_rows[:, :, None].expand_as(lse)

    actual_out = output.float()
    reference_out = output_reference.float()
    abs_output = (actual_out - reference_out).abs()[output_valid]
    reference_magnitude = reference_out.abs()[output_valid]
    relative_output = abs_output / reference_magnitude.clamp_min(relative_epsilon)
    squared_error = abs_output.square().mean() if abs_output.numel() else torch.tensor(0.0)
    squared_reference = (
        reference_magnitude.square().mean()
        if reference_magnitude.numel()
        else torch.tensor(0.0)
    )
    nrmse = torch.sqrt(squared_error) / torch.sqrt(squared_reference).clamp_min(
        relative_epsilon
    )

    row_actual = actual_out[valid_rows].reshape(-1, output.shape[-1])
    row_reference = reference_out[valid_rows].reshape(-1, output.shape[-1])
    if row_actual.numel():
        actual_norm = torch.linalg.vector_norm(row_actual, dim=-1)
        reference_norm = torch.linalg.vector_norm(row_reference, dim=-1)
        nonzero = (actual_norm > relative_epsilon) & (
            reference_norm > relative_epsilon
        )
        cosine = torch.ones_like(actual_norm)
        if bool(nonzero.any()):
            cosine[nonzero] = torch.nn.functional.cosine_similarity(
                row_actual[nonzero], row_reference[nonzero], dim=-1
            )
    else:
        cosine = torch.ones(1, device=output.device)

    finite_lse = torch.isfinite(lse_reference) & lse_valid
    abs_lse = (lse.float() - lse_reference.float()).abs()[finite_lse]
    inf_mismatches = int(
        (
            torch.isneginf(lse[lse_valid])
            != torch.isneginf(lse_reference[lse_valid])
        ).sum().item()
    )
    return AccuracyMetrics(
        output_abs_p50=_quantile(abs_output, 0.50),
        output_abs_p99=_quantile(abs_output, 0.99),
        output_abs_max=float(abs_output.max().item()) if abs_output.numel() else 0.0,
        output_rel_p99=_quantile(relative_output, 0.99),
        output_nrmse=float(nrmse.item()),
        output_cosine_p01=_quantile(cosine, 0.01),
        output_cosine_min=float(cosine.min().item()),
        lse_abs_p50=_quantile(abs_lse, 0.50),
        lse_abs_p99=_quantile(abs_lse, 0.99),
        lse_abs_max=float(abs_lse.max().item()) if abs_lse.numel() else 0.0,
        lse_inf_mask_mismatches=inf_mismatches,
        output_nan_count=int(torch.isnan(output).sum().item()),
        lse_nan_count=int(torch.isnan(lse).sum().item()),
    )
