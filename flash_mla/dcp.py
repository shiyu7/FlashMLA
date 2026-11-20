from typing import Optional, Sequence, Tuple, Union

import torch

import flash_mla.cuda as flash_mla_cuda


TensorOrInt = Union[torch.Tensor, int]


def _validate_cp(cp_size: int, cp_rank: int) -> None:
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    if not 0 <= cp_rank < cp_size:
        raise ValueError("cp_rank must satisfy 0 <= cp_rank < cp_size")


def cyclic_local_length(
    global_length: TensorOrInt,
    cp_size: int,
    cp_rank: int,
) -> TensorOrInt:
    """Map a global prefix length to its cyclic rank-local length.

    Global token ``i`` belongs to rank ``i % cp_size``.
    """
    _validate_cp(cp_size, cp_rank)
    if isinstance(global_length, int):
        if global_length < 0:
            raise ValueError("global_length must be non-negative")
        return max((global_length + cp_size - 1 - cp_rank) // cp_size, 0)

    # Do not inspect tensor values here: this helper is used on CUDA decode
    # metadata and a Python-side check would introduce a device sync.
    return torch.clamp(
        torch.div(
            global_length + cp_size - 1 - cp_rank,
            cp_size,
            rounding_mode="floor",
        ),
        min=0,
    )


def make_cyclic_local_causal_seqlens(
    global_cache_seqlens: torch.Tensor,
    q_len: int,
    cp_size: int,
    cp_rank: int,
    q_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the rank-local exclusive causal boundary for trailing queries.

    ``q_len`` is the fixed physical Q dimension. By default,
    ``global_cache_seqlens`` includes that many trailing query tokens.
    ``q_lens`` may instead provide a per-request valid prefix length; valid
    rows are aligned to each request's own trailing query window and padded
    suffix boundaries are clamped to the final cache boundary.

    The result has shape ``[batch, q_len]`` and dtype ``torch.int32``.
    """
    _validate_cp(cp_size, cp_rank)
    if global_cache_seqlens.ndim != 1:
        raise ValueError("global_cache_seqlens must have shape [batch]")
    if q_len <= 0:
        raise ValueError("q_len must be positive")
    if q_lens is not None:
        if q_lens.ndim != 1 or q_lens.shape != global_cache_seqlens.shape:
            raise ValueError("q_lens must have shape [batch]")
        if q_lens.dtype != torch.int32:
            raise ValueError("q_lens must have dtype torch.int32")
        if q_lens.device != global_cache_seqlens.device:
            raise ValueError(
                "q_lens and global_cache_seqlens must be on the same device"
            )
    # The caller must ensure every cache length covers its corresponding
    # logical query length and 0 <= q_lens <= q_len. Avoid value-dependent
    # Python checks because they would synchronize CUDA.

    query_indices = torch.arange(
        q_len,
        device=global_cache_seqlens.device,
        dtype=global_cache_seqlens.dtype,
    )
    logical_q_lens = (
        q_lens
        if q_lens is not None
        else torch.full_like(global_cache_seqlens, q_len)
    )
    global_boundaries = (
        global_cache_seqlens[:, None]
        + query_indices[None, :]
        + 1
        - logical_q_lens[:, None]
    )
    global_boundaries = torch.minimum(
        global_boundaries,
        global_cache_seqlens[:, None],
    )
    local_boundaries = cyclic_local_length(
        global_boundaries,
        cp_size,
        cp_rank,
    )
    return local_boundaries.to(torch.int32).contiguous()


def merge_local_attention(
    local_outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    local_lses: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge rank-local attention states using natural-log LSE.

    Inputs are stacked on the leading rank dimension. Outputs have shape
    ``[rank, batch, q_len, heads, d_v]`` and LSEs have shape
    ``[rank, batch, heads, q_len]``.
    """
    if not isinstance(local_outputs, torch.Tensor):
        local_outputs = torch.stack(tuple(local_outputs), dim=0)
    if not isinstance(local_lses, torch.Tensor):
        local_lses = torch.stack(tuple(local_lses), dim=0)
    if local_outputs.ndim != 5:
        raise ValueError(
            "local_outputs must have shape [rank, batch, q_len, heads, d_v]"
        )
    if local_lses.ndim != 4:
        raise ValueError(
            "local_lses must have shape [rank, batch, heads, q_len]"
        )
    expected_lse_shape = (
        local_outputs.shape[0],
        local_outputs.shape[1],
        local_outputs.shape[3],
        local_outputs.shape[2],
    )
    if tuple(local_lses.shape) != expected_lse_shape:
        raise ValueError(
            f"local_lses has shape {tuple(local_lses.shape)}, "
            f"expected {expected_lse_shape}"
        )

    global_lse = torch.logsumexp(local_lses.float(), dim=0)
    finite_global = torch.isfinite(global_lse)
    log_weights = torch.where(
        finite_global.unsqueeze(0),
        local_lses.float() - global_lse.unsqueeze(0),
        torch.full_like(local_lses, -torch.inf, dtype=torch.float32),
    )
    weights = torch.exp(log_weights).permute(0, 1, 3, 2).unsqueeze(-1)
    output = (local_outputs.float() * weights).sum(dim=0)
    return output.to(local_outputs.dtype), global_lse


def canonical_merge_pair_out(
    left_output: torch.Tensor,
    left_lse: torch.Tensor,
    right_output: torch.Tensor,
    right_lse: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one ordered FP32 edge of the canonical attention merge tree.

    Outputs use ``[B, Q, H, Dv]`` and LSEs use ``[B, H, Q]``. ``output`` and
    ``lse`` may alias the left state, enabling an in-place owner-side tree.
    """
    flash_mla_cuda.canonical_merge_pair_out(
        left_output,
        left_lse,
        right_output,
        right_lse,
        output,
        lse,
    )
    return output, lse
