"""Prototype for DCP-size-invariant attention reduction.

This is an evaluation oracle, not a runtime implementation.  It fixes global
KV chunk boundaries and the floating-point merge tree independently of how
chunks are assigned to DCP ranks.  A production kernel can use the same state
algebra while replacing the Python loop with fused SM90 kernels.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def merge_attention_states(
    left_output: torch.Tensor,
    left_lse: torch.Tensor,
    right_output: torch.Tensor,
    right_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two normalized attention states in FP32.

    The state is ``(O, L)`` where ``O`` is the normalized attention output and
    ``L`` is natural-log LSE.  ``(0, -inf)`` is the exact neutral element.
    Operand order is significant in floating point and is therefore fixed by
    :func:`canonical_tree_merge`.
    """
    if left_output.shape != right_output.shape:
        raise ValueError("left/right output shapes must match")
    if left_lse.shape != right_lse.shape:
        raise ValueError("left/right LSE shapes must match")
    if tuple(left_output.shape[:-1]) != tuple(left_lse.shape):
        raise ValueError("output and LSE shapes must match except for Dv")

    left_lse_fp32 = left_lse.float()
    right_lse_fp32 = right_lse.float()
    merged_lse = torch.logaddexp(left_lse_fp32, right_lse_fp32)
    finite = torch.isfinite(merged_lse)
    left_weight = torch.where(
        finite,
        torch.exp(left_lse_fp32 - merged_lse),
        torch.zeros_like(merged_lse),
    )
    right_weight = torch.where(
        finite,
        torch.exp(right_lse_fp32 - merged_lse),
        torch.zeros_like(merged_lse),
    )
    merged_output = (
        torch.nan_to_num(left_output.float()) * left_weight.unsqueeze(-1)
        + torch.nan_to_num(right_output.float()) * right_weight.unsqueeze(-1)
    )
    merged_output = torch.where(
        finite.unsqueeze(-1), merged_output, torch.zeros_like(merged_output)
    )
    return merged_output, merged_lse


def canonical_tree_merge(
    outputs: torch.Tensor | Sequence[torch.Tensor],
    lses: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge global-chunk states with a fixed adjacent balanced tree.

    Leading order is the global chunk id.  The tree pairs ``(0,1)``, ``(2,3)``
    and so on at every level; an odd final node is carried unchanged.  The
    result therefore does not depend on DCP ownership or arrival order.
    """
    output_nodes = list(outputs.unbind(0)) if isinstance(outputs, torch.Tensor) else list(outputs)
    lse_nodes = list(lses.unbind(0)) if isinstance(lses, torch.Tensor) else list(lses)
    if not output_nodes or len(output_nodes) != len(lse_nodes):
        raise ValueError("outputs/lses must contain the same non-zero chunk count")

    while len(output_nodes) > 1:
        next_outputs: list[torch.Tensor] = []
        next_lses: list[torch.Tensor] = []
        for index in range(0, len(output_nodes), 2):
            if index + 1 == len(output_nodes):
                next_outputs.append(output_nodes[index])
                next_lses.append(lse_nodes[index])
                continue
            output, lse = merge_attention_states(
                output_nodes[index],
                lse_nodes[index],
                output_nodes[index + 1],
                lse_nodes[index + 1],
            )
            next_outputs.append(output)
            next_lses.append(lse)
        output_nodes = next_outputs
        lse_nodes = next_lses
    return output_nodes[0], lse_nodes[0]


def canonical_chunk_attention(
    q: torch.Tensor,
    global_kv: torch.Tensor,
    global_seqlens: torch.Tensor,
    q_lens: torch.Tensor,
    *,
    chunk_size: int,
    cp_size: int,
    softmax_scale: float,
    descale_q: float = 1.0,
    descale_k: float = 1.0,
    head_dim_v: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate fixed-chunk attention while simulating DCP chunk ownership.

    Chunk ``i`` owns the contiguous global interval
    ``[i * chunk_size, (i + 1) * chunk_size)`` and is assigned to rank
    ``i % cp_size``.  States are always restored to global chunk-id order and
    reduced by :func:`canonical_tree_merge`.  Consequently CP=1/2/4/8 use the
    same leaf computations and the same floating-point merge graph.

    This function intentionally materializes every leaf state and is suitable
    only for correctness experiments.
    """
    if q.ndim != 4 or global_kv.ndim != 4:
        raise ValueError("q/global_kv must have BQHD/BSHD layouts")
    if chunk_size <= 0 or cp_size <= 0:
        raise ValueError("chunk_size and cp_size must be positive")
    batch, physical_q, num_heads, head_dim = q.shape
    if global_kv.shape[0] != batch or global_kv.shape[-1] != head_dim:
        raise ValueError("q/global_kv batch or head dimension mismatch")
    if tuple(global_seqlens.shape) != (batch,) or tuple(q_lens.shape) != (batch,):
        raise ValueError("global_seqlens and q_lens must have shape [B]")
    num_heads_k = global_kv.shape[2]
    if num_heads % num_heads_k:
        raise ValueError("query heads must be divisible by KV heads")
    if head_dim_v is None:
        head_dim_v = head_dim
    if not 0 < head_dim_v <= head_dim:
        raise ValueError("head_dim_v must satisfy 0 < head_dim_v <= head_dim")

    max_seqlen = int(global_seqlens.max().item())
    num_chunks = max(math.ceil(max(max_seqlen, 1) / chunk_size), 1)
    chunk_outputs: list[torch.Tensor | None] = [None] * num_chunks
    chunk_lses: list[torch.Tensor | None] = [None] * num_chunks
    heads_per_kv = num_heads // num_heads_k

    # Iterate by simulated rank to make ownership explicit.  Storing by global
    # chunk id removes scheduling/arrival order from the numerical contract.
    for cp_rank in range(cp_size):
        for chunk_index in range(cp_rank, num_chunks, cp_size):
            chunk_begin = chunk_index * chunk_size
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
            for batch_index in range(batch):
                seqlen = int(global_seqlens[batch_index].item())
                logical_q = int(q_lens[batch_index].item())
                if not 0 <= logical_q <= physical_q:
                    raise ValueError("q_lens values must satisfy 0 <= q_lens <= Q")
                if seqlen < logical_q:
                    raise ValueError("each global sequence must cover its queries")
                chunk_end = min(chunk_begin + chunk_size, seqlen)
                if logical_q == 0 or chunk_begin >= chunk_end:
                    continue

                query = (
                    q[batch_index, :logical_q]
                    .float()
                    .mul(descale_q)
                    .permute(1, 0, 2)
                )
                kv = (
                    global_kv[batch_index, chunk_begin:chunk_end]
                    .float()
                    .mul(descale_k)
                    .permute(1, 0, 2)
                    .repeat_interleave(heads_per_kv, dim=0)
                )
                scores = torch.matmul(query, kv.transpose(1, 2))
                scores.mul_(softmax_scale)
                visible = seqlen - logical_q + torch.arange(
                    1,
                    logical_q + 1,
                    dtype=torch.int64,
                    device=q.device,
                )
                key_indices = torch.arange(
                    chunk_begin, chunk_end, device=q.device
                )
                scores.masked_fill_(
                    key_indices.view(1, 1, -1) >= visible.view(1, -1, 1),
                    -torch.inf,
                )
                local_lse = torch.logsumexp(scores, dim=-1)
                finite = torch.isfinite(local_lse)
                probabilities = torch.where(
                    finite.unsqueeze(-1),
                    torch.exp(scores - local_lse.unsqueeze(-1)),
                    torch.zeros_like(scores),
                )
                local_output = torch.matmul(
                    probabilities, kv[..., :head_dim_v]
                )
                output[batch_index, :logical_q] = local_output.permute(1, 0, 2)
                lse[batch_index, :logical_q] = local_lse.permute(1, 0)
            chunk_outputs[chunk_index] = output
            chunk_lses[chunk_index] = lse

    if any(state is None for state in chunk_outputs + chunk_lses):
        raise RuntimeError("internal error: not every canonical chunk was evaluated")
    return canonical_tree_merge(
        [state for state in chunk_outputs if state is not None],
        [state for state in chunk_lses if state is not None],
    )
