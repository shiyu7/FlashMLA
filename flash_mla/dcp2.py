"""DCP2 single-node peer merge for dense FP8 MLA on SM90.

The fast path is deliberately narrow:

* exactly two DCP ranks;
* one process per local SM90 GPU with CUDA peer access (NVLink deployment);
* decode/MTP query length no larger than eight;
* a persistent PyTorch symmetric-memory workspace;
* MQA FlashMLA output written directly into that workspace.

DCP4/8 are not implemented as an all-peer-read extension of this algorithm.
``merge_dcp_attention`` routes every non-DCP2 or ineligible call through the
NCCL reduce-scatter fallback.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

import flash_mla.cuda as flash_mla_cuda
from flash_mla.flash_mla_interface import flash_mla_dense_fp8_out


DCP2_MAX_Q = 8


def dcp2_attention_payload_bytes(
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    head_dim_v: int,
) -> int:
    """Bytes read from one peer state, including BF16 output and FP32 LSE."""
    return batch_size * seqlen_q * num_heads * (head_dim_v * 2 + 4)


def is_dcp2_peer_attention_eligible(
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    head_dim_v: int,
    world_size: int,
    *,
    max_peer_payload_bytes: Optional[int] = None,
) -> bool:
    """Shape-only fast-path predicate.

    ``max_peer_payload_bytes`` is intentionally optional. It should be set
    from ``benchmark/bench_dcp2_crossover.py`` on the target NVLink machine;
    there is no topology-independent byte threshold that is honest for all
    ``B x Q x H x Dv`` combinations.
    """
    if (
        world_size != 2
        or batch_size <= 0
        or not 1 <= seqlen_q <= DCP2_MAX_Q
        or num_heads <= 0
        or num_heads % 2 != 0
        or head_dim_v <= 0
        or head_dim_v % 2 != 0
    ):
        return False
    if max_peer_payload_bytes is None:
        return True
    return (
        dcp2_attention_payload_bytes(
            batch_size, seqlen_q, num_heads, head_dim_v
        )
        <= max_peer_payload_bytes
    )


def _group_size(group: dist.ProcessGroup) -> int:
    return dist.get_world_size(group)


def _group_rank(group: dist.ProcessGroup) -> int:
    return dist.get_rank(group)


def _validate_single_node_peer_access(
    group: dist.ProcessGroup,
    device: torch.device,
) -> None:
    """Collectively reject cross-host and non-peer-accessible configurations.

    CUDA exposes peer accessibility here, not the physical link type. The
    deployment policy must still assign this fast path only to an NVLink pair.
    """
    if device.type != "cuda":
        raise RuntimeError("DCP2 peer attention requires a CUDA device")
    device_index = (
        torch.cuda.current_device() if device.index is None else device.index
    )
    local = (socket.gethostname(), device_index)
    placements: list[Optional[tuple[str, int]]] = [
        None for _ in range(_group_size(group))
    ]
    dist.all_gather_object(placements, local, group=group)
    hostnames = {placement[0] for placement in placements if placement is not None}
    if len(hostnames) != 1:
        raise RuntimeError(
            "DCP2 peer attention is single-node only; group spans "
            f"{sorted(hostnames)}"
        )
    peer_indices = [
        placement[1] for placement in placements if placement is not None
    ]
    if len(set(peer_indices)) != 2:
        raise RuntimeError(
            "DCP2 peer attention requires two distinct local CUDA devices"
        )
    peer_index = peer_indices[1 - _group_rank(group)]
    if not torch.cuda.can_device_access_peer(device_index, peer_index):
        raise RuntimeError(
            f"CUDA device {device_index} cannot directly access peer "
            f"device {peer_index}"
        )


@dataclass
class DCP2AttentionWorkspace:
    """Persistent symmetric storage and device-side epoch state.

    Buffers are allocated flat so every smaller ``B x Q x H x Dv`` view is
    contiguous. The flags have one ``(write_epoch, read_epoch)`` pair per
    attention row. Device kernels increment these counters, so captured graphs
    can replay without a host-side slot or epoch update.
    """

    group: dist.ProcessGroup
    rank: int
    max_batch_size: int
    max_seqlen_q: int
    max_num_heads: int
    max_head_dim_v: int
    output_handle: Any
    lse_handle: Any
    flag_handle: Any
    local_output_storage: Optional[torch.Tensor]
    local_lse_storage: Optional[torch.Tensor]
    local_flags: Optional[torch.Tensor]
    peer_output_storage: Optional[Tuple[torch.Tensor, torch.Tensor]]
    peer_lse_storage: Optional[Tuple[torch.Tensor, torch.Tensor]]
    flag_ptrs: Optional[torch.Tensor]
    active_shape: Optional[Tuple[int, int, int, int]] = None

    @property
    def world_size(self) -> int:
        return 2

    @property
    def max_rows(self) -> int:
        return self.max_batch_size * self.max_seqlen_q * self.max_num_heads

    @property
    def output_numel(self) -> int:
        return self.max_rows * self.max_head_dim_v

    def _check_open(self) -> None:
        if (
            self.local_output_storage is None
            or self.local_lse_storage is None
            or self.peer_output_storage is None
            or self.peer_lse_storage is None
            or self.flag_ptrs is None
        ):
            raise RuntimeError("DCP2 attention workspace is closed")

    def _check_shape(
        self,
        batch_size: int,
        seqlen_q: int,
        num_heads: int,
        head_dim_v: int,
    ) -> None:
        if not is_dcp2_peer_attention_eligible(
            batch_size, seqlen_q, num_heads, head_dim_v, self.world_size
        ):
            raise ValueError(
                "DCP2 peer attention requires B > 0, 1 <= Q <= 8, "
                "even H, and positive even Dv"
            )
        requested = (batch_size, seqlen_q, num_heads, head_dim_v)
        capacity = (
            self.max_batch_size,
            self.max_seqlen_q,
            self.max_num_heads,
            self.max_head_dim_v,
        )
        if any(value > limit for value, limit in zip(requested, capacity)):
            raise ValueError(
                f"requested shape {requested} exceeds workspace {capacity}"
            )
        if num_heads != self.max_num_heads:
            raise ValueError(
                "num_heads is fixed for a workspace because head ownership is "
                "part of the device epoch protocol; create one workspace per H"
            )
        if self.active_shape is None:
            self.active_shape = requested
        elif requested != self.active_shape:
            raise ValueError(
                "a DCP2 workspace is bound to its first active shape for "
                "epoch-safe compact storage: "
                f"active={self.active_shape}, requested={requested}. "
                "Create a separate workspace for each CUDA Graph shape."
            )

    def local_state(
        self,
        batch_size: int,
        seqlen_q: int,
        num_heads: int,
        head_dim_v: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous symmetric ``BQHD`` output and ``BQH`` LSE views."""
        self._check_open()
        self._check_shape(batch_size, seqlen_q, num_heads, head_dim_v)
        output_numel = batch_size * seqlen_q * num_heads * head_dim_v
        rows = batch_size * seqlen_q * num_heads
        output = self.local_output_storage[:output_numel].view(
            batch_size, seqlen_q, num_heads, head_dim_v
        )
        lse = self.local_lse_storage[:rows].view(
            batch_size, seqlen_q, num_heads
        )
        return output, lse

    def _peer_states(
        self,
        batch_size: int,
        seqlen_q: int,
        num_heads: int,
        head_dim_v: int,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        output_numel = batch_size * seqlen_q * num_heads * head_dim_v
        rows = batch_size * seqlen_q * num_heads
        output_shape = (batch_size, seqlen_q, num_heads, head_dim_v)
        lse_shape = (batch_size, seqlen_q, num_heads)
        outputs = tuple(
            storage[:output_numel].view(output_shape)
            for storage in self.peer_output_storage
        )
        lses = tuple(
            storage[:rows].view(lse_shape)
            for storage in self.peer_lse_storage
        )
        return outputs, lses

    def prepare(
        self,
        batch_size: int,
        seqlen_q: int,
        num_heads: int,
        head_dim_v: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wait for previous consumers, then return FlashMLA destination views."""
        local_out, local_lse = self.local_state(
            batch_size, seqlen_q, num_heads, head_dim_v
        )
        flash_mla_cuda.dcp2_prepare_attention(
            self.flag_ptrs,
            batch_size,
            seqlen_q,
            num_heads,
            self.rank,
        )
        return local_out, local_lse

    def merge(
        self,
        batch_size: int,
        seqlen_q: int,
        num_heads: int,
        head_dim_v: int,
        *,
        out: Optional[torch.Tensor] = None,
        softmax_lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Publish local state and fuse DCP2 peer LSE/output reduction."""
        self._check_open()
        self._check_shape(batch_size, seqlen_q, num_heads, head_dim_v)
        local_heads = num_heads // 2
        device = self.local_output_storage.device
        if out is None:
            out = torch.empty(
                batch_size,
                seqlen_q,
                local_heads,
                head_dim_v,
                dtype=torch.bfloat16,
                device=device,
            )
        if softmax_lse is None:
            softmax_lse = torch.empty(
                batch_size,
                seqlen_q,
                local_heads,
                dtype=torch.float32,
                device=device,
            )
        peer_outputs, peer_lses = self._peer_states(
            batch_size, seqlen_q, num_heads, head_dim_v
        )
        flash_mla_cuda.dcp2_merge_attention(
            peer_outputs[0],
            peer_outputs[1],
            peer_lses[0],
            peer_lses[1],
            self.flag_ptrs,
            self.rank,
            out,
            softmax_lse,
        )
        return out, softmax_lse

    def run_flash_mla_dense_fp8(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        head_dim_v: int,
        tile_scheduler_metadata: torch.Tensor,
        num_splits: torch.Tensor,
        *,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        causal_seqlens_k: Optional[torch.Tensor] = None,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        softmax_lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the direct-write FlashMLA + fused peer merge fast path."""
        batch_size, seqlen_q, num_heads, _ = q.shape
        local_out, local_lse = self.prepare(
            batch_size, seqlen_q, num_heads, head_dim_v
        )
        flash_mla_dense_fp8_out(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            tile_scheduler_metadata,
            num_splits,
            local_out,
            local_lse,
            softmax_scale=softmax_scale,
            causal=causal,
            causal_seqlens_k=causal_seqlens_k,
            descale_q=descale_q,
            descale_k=descale_k,
        )
        return self.merge(
            batch_size,
            seqlen_q,
            num_heads,
            head_dim_v,
            out=out,
            softmax_lse=softmax_lse,
        )

    def close(self) -> None:
        if self.local_output_storage is None:
            return
        torch.cuda.synchronize(self.local_output_storage.device)
        self.flag_ptrs = None
        self.peer_lse_storage = None
        self.peer_output_storage = None
        self.local_flags = None
        self.local_lse_storage = None
        self.local_output_storage = None
        self.flag_handle = None
        self.lse_handle = None
        self.output_handle = None
        self.active_shape = None


def create_dcp2_attention_workspace(
    max_batch_size: int,
    max_seqlen_q: int,
    max_num_heads: int,
    max_head_dim_v: int,
    group: dist.ProcessGroup,
    device: Optional[torch.device] = None,
    *,
    validate_single_node_peer_access: bool = True,
) -> DCP2AttentionWorkspace:
    """Collectively allocate the persistent DCP2 symmetric workspace."""
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except ImportError as exc:
        raise RuntimeError("PyTorch symmetric memory is unavailable") from exc

    if _group_size(group) != 2:
        raise RuntimeError(
            "the peer-read attention workspace is DCP2-only; "
            f"got DCP{_group_size(group)} (use NCCL fallback)"
        )
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    if validate_single_node_peer_access:
        _validate_single_node_peer_access(group, device)
    if not is_dcp2_peer_attention_eligible(
        max_batch_size,
        max_seqlen_q,
        max_num_heads,
        max_head_dim_v,
        2,
    ):
        raise ValueError(
            "workspace capacity requires B > 0, 1 <= max Q <= 8, "
            "even H, and positive even Dv"
        )
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("DCP2 peer attention requires SM90")

    max_rows = max_batch_size * max_seqlen_q * max_num_heads
    output_numel = max_rows * max_head_dim_v
    with torch.inference_mode(False), torch.no_grad():
        local_output = symm_mem.empty(
            (output_numel,), dtype=torch.bfloat16, device=device
        )
        local_lse = symm_mem.empty(
            (max_rows,), dtype=torch.float32, device=device
        )
        local_flags = symm_mem.empty(
            (max_rows, 2), dtype=torch.int64, device=device
        )
    local_flags.zero_()
    torch.cuda.synchronize(device)

    output_handle = symm_mem.rendezvous(local_output, group=group)
    lse_handle = symm_mem.rendezvous(local_lse, group=group)
    flag_handle = symm_mem.rendezvous(local_flags, group=group)
    if output_handle is None or lse_handle is None or flag_handle is None:
        raise RuntimeError("symmetric-memory rendezvous returned no handle")
    output_handle.barrier()
    lse_handle.barrier()
    flag_handle.barrier()

    peer_outputs = tuple(
        output_handle.get_buffer(rank, (output_numel,), torch.bfloat16)
        for rank in range(2)
    )
    peer_lses = tuple(
        lse_handle.get_buffer(rank, (max_rows,), torch.float32)
        for rank in range(2)
    )
    flag_ptrs = torch.tensor(
        flag_handle.buffer_ptrs,
        dtype=torch.int64,
        device=device,
    )
    if bool(flag_ptrs.eq(0).any()):
        raise RuntimeError("symmetric-memory peer mapping contains a null pointer")

    return DCP2AttentionWorkspace(
        group=group,
        rank=_group_rank(group),
        max_batch_size=max_batch_size,
        max_seqlen_q=max_seqlen_q,
        max_num_heads=max_num_heads,
        max_head_dim_v=max_head_dim_v,
        output_handle=output_handle,
        lse_handle=lse_handle,
        flag_handle=flag_handle,
        local_output_storage=local_output,
        local_lse_storage=local_lse,
        local_flags=local_flags,
        peer_output_storage=peer_outputs,
        peer_lse_storage=peer_lses,
        flag_ptrs=flag_ptrs,
    )


def nccl_merge_attention(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    out: Optional[torch.Tensor] = None,
    softmax_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Natural-log LSE correction plus NCCL reduce-scatter fallback.

    Inputs use the native ``BQHD/BQH`` layouts. Query heads are evenly
    reduce-scattered across the process group.
    """
    if local_out.ndim != 4 or local_lse.ndim != 3:
        raise ValueError("local_out/local_lse must have BQHD/BQH layouts")
    batch_size, seqlen_q, num_heads, head_dim_v = local_out.shape
    if tuple(local_lse.shape) != (batch_size, seqlen_q, num_heads):
        raise ValueError("local_lse shape does not match local_out")
    world_size = _group_size(group)
    rank = _group_rank(group)
    if num_heads % world_size != 0:
        raise ValueError("num_heads must be divisible by DCP world size")
    local_heads = num_heads // world_size

    gathered_lse_flat = torch.empty(
        world_size * batch_size,
        seqlen_q,
        num_heads,
        dtype=torch.float32,
        device=local_lse.device,
    )
    dist.all_gather_into_tensor(
        gathered_lse_flat, local_lse.float().contiguous(), group=group
    )
    gathered_lse = gathered_lse_flat.view(
        world_size, batch_size, seqlen_q, num_heads
    )
    global_lse = torch.logsumexp(gathered_lse, dim=0)
    finite = torch.isfinite(global_lse)
    weights = torch.where(
        finite,
        torch.exp(local_lse.float() - global_lse),
        torch.zeros_like(global_lse),
    )
    corrected = torch.nan_to_num(local_out.float()) * weights.unsqueeze(-1)

    # NCCL reduce-scatter splits dimension zero. Move heads first so each rank
    # receives the same contiguous head shard as the peer kernel.
    reduce_input = corrected.permute(2, 0, 1, 3).contiguous()
    reduce_output = torch.empty(
        local_heads,
        batch_size,
        seqlen_q,
        head_dim_v,
        dtype=torch.float32,
        device=local_out.device,
    )
    dist.reduce_scatter_tensor(
        reduce_output, reduce_input, op=dist.ReduceOp.SUM, group=group
    )
    reduced_bqhd = reduce_output.permute(1, 2, 0, 3)
    head_begin = rank * local_heads
    head_end = head_begin + local_heads
    local_global_lse = global_lse[:, :, head_begin:head_end]

    if out is None:
        out = reduced_bqhd.to(local_out.dtype).contiguous()
    else:
        out.copy_(reduced_bqhd)
    if softmax_lse is None:
        softmax_lse = local_global_lse.contiguous()
    else:
        softmax_lse.copy_(local_global_lse)
    return out, softmax_lse


def merge_dcp_attention(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    workspace: Optional[DCP2AttentionWorkspace] = None,
    max_peer_payload_bytes: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    softmax_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route eligible DCP2 shapes to peer merge and everything else to NCCL."""
    batch_size, seqlen_q, num_heads, head_dim_v = local_out.shape
    eligible = is_dcp2_peer_attention_eligible(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim_v,
        _group_size(group),
        max_peer_payload_bytes=max_peer_payload_bytes,
    )
    if workspace is not None and eligible:
        expected_out, expected_lse = workspace.local_state(
            batch_size, seqlen_q, num_heads, head_dim_v
        )
        if (
            local_out.data_ptr() != expected_out.data_ptr()
            or local_lse.data_ptr() != expected_lse.data_ptr()
        ):
            raise ValueError(
                "peer merge inputs must be the workspace views written "
                "directly by FlashMLA"
            )
        return workspace.merge(
            batch_size,
            seqlen_q,
            num_heads,
            head_dim_v,
            out=out,
            softmax_lse=softmax_lse,
        )
    return nccl_merge_attention(
        local_out,
        local_lse,
        group,
        out=out,
        softmax_lse=softmax_lse,
    )
