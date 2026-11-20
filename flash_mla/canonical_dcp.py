from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

import flash_mla.cuda as flash_mla_cuda
from flash_mla.dcp import cyclic_local_length
from flash_mla.flash_mla_interface import flash_mla_dense_fp8_canonical_leaves_out


_WORLD_SIZE = 8
_PAGE_SIZE = 64


def _group_size(group: dist.ProcessGroup) -> int:
    return dist.get_world_size(group)


def _group_rank(group: dist.ProcessGroup) -> int:
    return dist.get_rank(group)


def _validate_single_node_peer_access(
    group: dist.ProcessGroup,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        raise RuntimeError("canonical DCP8 peer attention requires CUDA")
    device_index = (
        torch.cuda.current_device() if device.index is None else device.index
    )
    local = (socket.gethostname(), device_index)
    placements: list[Optional[tuple[str, int]]] = [
        None for _ in range(_group_size(group))
    ]
    dist.all_gather_object(placements, local, group=group)
    resolved = [placement for placement in placements if placement is not None]
    hostnames = {placement[0] for placement in resolved}
    if len(hostnames) != 1:
        raise RuntimeError(
            "canonical DCP8 peer attention is single-node only; group spans "
            f"{sorted(hostnames)}"
        )
    peer_indices = [placement[1] for placement in resolved]
    if len(set(peer_indices)) != _WORLD_SIZE:
        raise RuntimeError(
            "canonical DCP8 requires eight distinct local CUDA devices"
        )
    for peer_index in peer_indices:
        if peer_index == device_index:
            continue
        if not torch.cuda.can_device_access_peer(device_index, peer_index):
            raise RuntimeError(
                f"CUDA device {device_index} cannot directly access peer "
                f"device {peer_index}"
            )


@dataclass
class CanonicalDCP8Workspace:
    """Persistent SM90 workspace for exact token-cyclic DCP8 attention.

    The allocator remains token-cyclic. Each call reconstructs rank ``r``'s
    fixed global chunk ``[r*C, (r+1)*C)`` in a local contiguous cache, writes
    one FP32 leaf into symmetric storage, and applies the fixed adjacent peer
    tree. The final H/8 shard is emitted directly in compact BQHD layout.
    """

    group: dist.ProcessGroup
    rank: int
    chunk_size: int
    max_batch_size: int
    max_seqlen_q: int
    num_heads: int
    head_dim: int
    head_dim_v: int
    staging_handle: Any
    staging_flag_handle: Any
    output_handle: Any
    lse_handle: Any
    merge_flag_handle: Any
    local_staging_storage: Optional[torch.Tensor]
    local_staging_flags: Optional[torch.Tensor]
    local_output_storage: Optional[torch.Tensor]
    local_lse_storage: Optional[torch.Tensor]
    local_merge_flags: Optional[torch.Tensor]
    staging_ptrs: Optional[torch.Tensor]
    staging_flag_ptrs: Optional[torch.Tensor]
    output_ptrs: Optional[torch.Tensor]
    lse_ptrs: Optional[torch.Tensor]
    merge_flag_ptrs: Optional[torch.Tensor]
    canonical_cache: Optional[torch.Tensor]
    canonical_block_table: Optional[torch.Tensor]
    compact_output_storage: Optional[torch.Tensor]
    compact_lse_storage: Optional[torch.Tensor]

    @property
    def world_size(self) -> int:
        return _WORLD_SIZE

    @property
    def max_rows(self) -> int:
        return self.max_batch_size * self.max_seqlen_q * self.num_heads

    @property
    def max_global_tokens(self) -> int:
        return self.chunk_size * self.world_size

    def _check_open(self) -> None:
        required = (
            self.local_staging_storage,
            self.local_output_storage,
            self.local_lse_storage,
            self.staging_ptrs,
            self.staging_flag_ptrs,
            self.output_ptrs,
            self.lse_ptrs,
            self.merge_flag_ptrs,
            self.canonical_cache,
            self.canonical_block_table,
            self.compact_output_storage,
            self.compact_lse_storage,
        )
        if any(value is None for value in required):
            raise RuntimeError("canonical DCP8 workspace is closed")

    def _check_inputs(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        global_cache_seqlens: torch.Tensor,
    ) -> Tuple[int, int]:
        self._check_open()
        if q.ndim != 4:
            raise ValueError("q must have shape [B,Q,H,D]")
        batch_size, seqlen_q, num_heads, head_dim = q.shape
        if not (0 < batch_size <= self.max_batch_size):
            raise ValueError("batch size exceeds canonical DCP8 workspace")
        if not (0 < seqlen_q <= self.max_seqlen_q):
            raise ValueError("Q exceeds canonical DCP8 workspace")
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError("q head shape does not match canonical workspace")
        if q.dtype != torch.float8_e4m3fn:
            raise ValueError("canonical DCP8 requires FP8 E4M3 q")
        if (
            k_cache.ndim != 4
            or k_cache.shape[1:] != (self._page_size, 1, self.head_dim)
            or k_cache.dtype != torch.float8_e4m3fn
        ):
            raise ValueError("k_cache must have shape [pages,64,1,D] FP8 E4M3")
        if (
            block_table.ndim != 2
            or block_table.shape[0] != batch_size
            or block_table.shape[1] < self.pages_per_chunk
            or block_table.dtype != torch.int32
        ):
            raise ValueError(
                "block_table must be int32 [B,>=chunk_size/64]"
            )
        if (
            global_cache_seqlens.ndim != 1
            or global_cache_seqlens.shape[0] != batch_size
            or global_cache_seqlens.dtype != torch.int32
        ):
            raise ValueError("global_cache_seqlens must be int32 [B]")
        tensors = (q, k_cache, block_table, global_cache_seqlens)
        if any(tensor.device != q.device for tensor in tensors):
            raise ValueError("canonical DCP8 inputs must be on one device")
        return batch_size, seqlen_q

    @property
    def _page_size(self) -> int:
        return _PAGE_SIZE

    @property
    def pages_per_chunk(self) -> int:
        return self.chunk_size // self._page_size

    def chunk_cache_seqlens(
        self,
        global_cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        chunk_begin = self.rank * self.chunk_size
        return torch.clamp(
            global_cache_seqlens - chunk_begin,
            min=0,
            max=self.chunk_size,
        ).to(torch.int32).contiguous()

    def chunk_causal_seqlens(
        self,
        global_cache_seqlens: torch.Tensor,
        seqlen_q: int,
    ) -> torch.Tensor:
        query = torch.arange(
            seqlen_q,
            dtype=global_cache_seqlens.dtype,
            device=global_cache_seqlens.device,
        )
        global_boundaries = (
            global_cache_seqlens[:, None] + query[None, :] + 1 - seqlen_q
        )
        chunk_begin = self.rank * self.chunk_size
        return torch.clamp(
            global_boundaries - chunk_begin,
            min=0,
            max=self.chunk_size,
        ).to(torch.int32).contiguous()

    def attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        global_cache_seqlens: torch.Tensor,
        tile_scheduler_metadata: torch.Tensor,
        num_splits: torch.Tensor,
        *,
        softmax_scale: float,
        descale_q: torch.Tensor,
        descale_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seqlen_q = self._check_inputs(
            q, k_cache, block_table, global_cache_seqlens
        )
        local_cache_seqlens = cyclic_local_length(
            global_cache_seqlens, self.world_size, self.rank
        ).to(torch.int32).contiguous()
        chunk_cache_seqlens = self.chunk_cache_seqlens(global_cache_seqlens)
        chunk_causal_seqlens = self.chunk_causal_seqlens(
            global_cache_seqlens, seqlen_q
        )

        staging_numel = batch_size * self.chunk_size * self.head_dim
        local_staging = self.local_staging_storage[:staging_numel].view(
            torch.float8_e4m3fn
        ).view(batch_size, self.chunk_size, self.head_dim)
        canonical_cache = self.canonical_cache[:batch_size]
        rows = batch_size * seqlen_q * self.num_heads
        output_numel = rows * self.head_dim_v
        local_output = self.local_output_storage[:output_numel].view(
            batch_size, seqlen_q, self.num_heads, self.head_dim_v
        )
        local_lse = self.local_lse_storage[:rows].view(
            batch_size, self.num_heads, seqlen_q
        )
        local_heads = self.num_heads // self.world_size
        compact_output_numel = (
            batch_size * seqlen_q * local_heads * self.head_dim_v
        )
        compact_output = self.compact_output_storage[
            :compact_output_numel
        ].view(batch_size, seqlen_q, local_heads, self.head_dim_v)
        compact_lse = self.compact_lse_storage[
            : batch_size * local_heads * seqlen_q
        ].view(batch_size, local_heads, seqlen_q)

        flash_mla_cuda.canonical_peer_pack_token_cyclic_kv(
            self.staging_ptrs,
            self.staging_flag_ptrs,
            self.rank,
            k_cache,
            block_table,
            local_cache_seqlens,
            local_staging,
            canonical_cache,
        )
        flash_mla_cuda.canonical_peer_reduce_scatter_prepare(
            self.merge_flag_ptrs,
            rows,
            self.rank,
        )
        _, leaf_lses = flash_mla_dense_fp8_canonical_leaves_out(
            q=q,
            k_cache=canonical_cache.view(-1, self._page_size, 1, self.head_dim),
            block_table=self.canonical_block_table[:batch_size],
            cache_seqlens=chunk_cache_seqlens,
            head_dim_v=self.head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            canonical_num_chunks=1,
            out=local_output.unsqueeze(1),
            softmax_scale=softmax_scale,
            causal=True,
            causal_seqlens_k=chunk_causal_seqlens,
            descale_q=descale_q,
            descale_k=descale_k,
        )
        local_lse.copy_(leaf_lses[:, 0])
        flash_mla_cuda.canonical_peer_reduce_scatter_compact(
            self.output_ptrs,
            self.lse_ptrs,
            self.merge_flag_ptrs,
            self.rank,
            local_output,
            local_lse,
            compact_output,
            compact_lse,
        )
        return compact_output, compact_lse

    def close(self) -> None:
        if self.local_output_storage is None:
            return
        torch.cuda.synchronize(self.local_output_storage.device)
        self.compact_lse_storage = None
        self.compact_output_storage = None
        self.canonical_block_table = None
        self.canonical_cache = None
        self.merge_flag_ptrs = None
        self.lse_ptrs = None
        self.output_ptrs = None
        self.staging_flag_ptrs = None
        self.staging_ptrs = None
        self.local_merge_flags = None
        self.local_lse_storage = None
        self.local_output_storage = None
        self.local_staging_flags = None
        self.local_staging_storage = None
        self.merge_flag_handle = None
        self.lse_handle = None
        self.output_handle = None
        self.staging_flag_handle = None
        self.staging_handle = None


def create_canonical_dcp8_workspace(
    max_batch_size: int,
    max_seqlen_q: int,
    num_heads: int,
    head_dim: int,
    head_dim_v: int,
    chunk_size: int,
    group: dist.ProcessGroup,
    device: Optional[torch.device] = None,
    *,
    validate_single_node_peer_access: bool = True,
) -> CanonicalDCP8Workspace:
    """Collectively create a persistent exact token-cyclic DCP8 workspace."""
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except ImportError as exc:
        raise RuntimeError("PyTorch symmetric memory is unavailable") from exc

    if _group_size(group) != _WORLD_SIZE:
        raise RuntimeError(
            f"canonical peer attention requires DCP8, got DCP{_group_size(group)}"
        )
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    if validate_single_node_peer_access:
        _validate_single_node_peer_access(group, device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("canonical DCP8 peer attention requires SM90")
    if (
        max_batch_size <= 0
        or not 0 < max_seqlen_q <= 8
        or num_heads <= 0
        or num_heads % _WORLD_SIZE
        or head_dim <= 0
        or head_dim % 16
        or head_dim_v <= 0
        or head_dim_v % 4
        or chunk_size <= 0
        or chunk_size % _PAGE_SIZE
    ):
        raise ValueError(
            "canonical workspace requires B>0, 1<=Q<=8, H%8=0, "
            "D%16=0, Dv%4=0, and chunk_size%64=0"
        )

    max_rows = max_batch_size * max_seqlen_q * num_heads
    staging_numel = max_batch_size * chunk_size * head_dim
    output_numel = max_rows * head_dim_v
    local_heads = num_heads // _WORLD_SIZE
    compact_output_numel = (
        max_batch_size * max_seqlen_q * local_heads * head_dim_v
    )
    compact_lse_numel = max_batch_size * local_heads * max_seqlen_q
    with torch.inference_mode(False), torch.no_grad():
        local_staging = symm_mem.empty(
            (staging_numel,), dtype=torch.uint8, device=device
        )
        staging_flags = symm_mem.empty((2,), dtype=torch.int64, device=device)
        local_output = symm_mem.empty(
            (output_numel,), dtype=torch.float32, device=device
        )
        local_lse = symm_mem.empty(
            (max_rows,), dtype=torch.float32, device=device
        )
        merge_flags = symm_mem.empty(
            (max_rows, 4), dtype=torch.int64, device=device
        )
    staging_flags.zero_()
    merge_flags.zero_()
    torch.cuda.synchronize(device)

    staging_handle = symm_mem.rendezvous(local_staging, group=group)
    staging_flag_handle = symm_mem.rendezvous(staging_flags, group=group)
    output_handle = symm_mem.rendezvous(local_output, group=group)
    lse_handle = symm_mem.rendezvous(local_lse, group=group)
    merge_flag_handle = symm_mem.rendezvous(merge_flags, group=group)
    handles = (
        staging_handle,
        staging_flag_handle,
        output_handle,
        lse_handle,
        merge_flag_handle,
    )
    if any(handle is None for handle in handles):
        raise RuntimeError("symmetric-memory rendezvous returned no handle")
    for handle in handles:
        handle.barrier()

    def pointer_tensor(handle: Any) -> torch.Tensor:
        pointers = torch.tensor(
            handle.buffer_ptrs, dtype=torch.int64, device=device
        )
        if bool(pointers.eq(0).any()):
            raise RuntimeError("symmetric-memory peer mapping contains null")
        return pointers

    canonical_cache = torch.empty(
        max_batch_size,
        chunk_size,
        1,
        head_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    pages_per_chunk = chunk_size // _PAGE_SIZE
    canonical_block_table = torch.arange(
        max_batch_size * pages_per_chunk,
        dtype=torch.int32,
        device=device,
    ).view(max_batch_size, pages_per_chunk)
    compact_output = torch.empty(
        compact_output_numel, dtype=torch.float32, device=device
    )
    compact_lse = torch.empty(
        compact_lse_numel, dtype=torch.float32, device=device
    )

    return CanonicalDCP8Workspace(
        group=group,
        rank=_group_rank(group),
        chunk_size=chunk_size,
        max_batch_size=max_batch_size,
        max_seqlen_q=max_seqlen_q,
        num_heads=num_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        staging_handle=staging_handle,
        staging_flag_handle=staging_flag_handle,
        output_handle=output_handle,
        lse_handle=lse_handle,
        merge_flag_handle=merge_flag_handle,
        local_staging_storage=local_staging,
        local_staging_flags=staging_flags,
        local_output_storage=local_output,
        local_lse_storage=local_lse,
        local_merge_flags=merge_flags,
        staging_ptrs=pointer_tensor(staging_handle),
        staging_flag_ptrs=pointer_tensor(staging_flag_handle),
        output_ptrs=pointer_tensor(output_handle),
        lse_ptrs=pointer_tensor(lse_handle),
        merge_flag_ptrs=pointer_tensor(merge_flag_handle),
        canonical_cache=canonical_cache,
        canonical_block_table=canonical_block_table,
        compact_output_storage=compact_output,
        compact_lse_storage=compact_lse,
    )
