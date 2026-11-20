"""Benchmark the R14-P3 eight-GPU canonical DCP tree on one SM90 node.

Run with::

    torchrun --standalone --nproc-per-node=8 \
      benchmark/bench_dcp_canonical_tree.py

Each rank owns one contiguous 1K global chunk. Adjacent owners exchange one
FP32 attention state per tree edge; the lower global chunk id survives and
applies the ordered CUDA pair merge in place. No all-gather is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.distributed as dist

import flash_mla
import flash_mla.cuda as flash_mla_cuda


PAGE_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512
WORLD_SIZE = 8


def _make_edge_groups() -> dict[tuple[int, int], dist.ProcessGroup]:
    edges = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (4, 6), (0, 4))
    return {edge: dist.new_group(list(edge)) for edge in edges}


def _exchange(
    output: torch.Tensor,
    lse: torch.Tensor,
    peer: int,
    group: dist.ProcessGroup,
    *,
    receive: bool,
) -> None:
    operation = dist.irecv if receive else dist.isend
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(operation, output, peer, group=group),
            dist.P2POp(operation, lse, peer, group=group),
        ]
    )
    for request in requests:
        request.wait()


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError("R14-P3 benchmark requires exactly eight ranks")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("R14-P3 benchmark requires SM90")
    edge_groups = _make_edge_groups() if args.transport == "nccl" else {}

    torch.manual_seed(args.seed)
    q_generator = torch.Generator(device=device)
    q_generator.manual_seed(args.seed)
    q = (
        torch.randn(
            args.batch,
            args.q_len,
            args.heads,
            HEAD_DIM,
            dtype=torch.float32,
            device=device,
            generator=q_generator,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    kv_generator = torch.Generator(device=device)
    kv_generator.manual_seed(args.seed + 1)
    global_capacity = args.chunk_size * WORLD_SIZE
    global_seqlen = (
        global_capacity if args.global_kv_len is None else args.global_kv_len
    )
    global_kv = (
        torch.randn(
            args.batch,
            global_capacity,
            1,
            HEAD_DIM,
            dtype=torch.float32,
            device=device,
            generator=kv_generator,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    chunk_begin = rank * args.chunk_size
    if args.kv_layout == "token-cyclic":
        local_cache = global_kv[:, rank::WORLD_SIZE].reshape(
            -1, PAGE_SIZE, 1, HEAD_DIM
        ).contiguous()
    else:
        local_cache = global_kv[
            :, chunk_begin : chunk_begin + args.chunk_size
        ].reshape(-1, PAGE_SIZE, 1, HEAD_DIM).contiguous()
    local_pages = args.chunk_size // PAGE_SIZE
    block_table = torch.arange(
        args.batch * local_pages, dtype=torch.int32, device=device
    ).view(args.batch, local_pages)
    local_token_count = max(
        0, (global_seqlen + WORLD_SIZE - 1 - rank) // WORLD_SIZE
    )
    staging_seqlens = torch.full(
        (args.batch,), local_token_count, dtype=torch.int32, device=device
    )
    chunk_token_count = min(
        args.chunk_size, max(0, global_seqlen - chunk_begin)
    )
    chunk_seqlens = torch.full(
        (args.batch,), chunk_token_count, dtype=torch.int32, device=device
    )
    visible_global = torch.arange(
        global_seqlen - args.q_len + 1,
        global_seqlen + 1,
        dtype=torch.int32,
        device=device,
    )
    local_causal = (
        visible_global.sub(chunk_begin).clamp_(0, args.chunk_size)
        .view(1, -1)
        .expand(args.batch, -1)
        .contiguous()
    )
    descale_q = torch.tensor([0.75], dtype=torch.float32, device=device)
    descale_k = torch.tensor([1.25], dtype=torch.float32, device=device)
    metadata, splits = flash_mla.get_mla_metadata_canonical_dense_fp8(
        chunk_seqlens, args.chunk_size, 1
    )
    common = dict(
        softmax_scale=1 / math.sqrt(HEAD_DIM),
        causal=True,
        causal_seqlens_k=local_causal,
        descale_q=descale_q,
        descale_k=descale_k,
    )
    recv_outputs = [
        torch.empty(
            args.batch,
            args.q_len,
            args.heads,
            HEAD_DIM_V,
            dtype=torch.float32,
            device=device,
        )
        for _ in range(3)
    ]
    recv_lses = [
        torch.empty(
            args.batch,
            args.heads,
            args.q_len,
            dtype=torch.float32,
            device=device,
        )
        for _ in range(3)
    ]

    output_handle = None
    lse_handle = None
    flag_handle = None
    local_peer_output = None
    local_peer_lse = None
    output_ptrs = None
    lse_ptrs = None
    flag_ptrs = None
    kv_staging_handle = None
    kv_flag_handle = None
    local_kv_staging = None
    canonical_cache = None
    kv_staging_ptrs = None
    kv_flag_ptrs = None
    if args.transport.startswith("peer"):
        import torch.distributed._symmetric_memory as symm_mem

        rows = args.batch * args.q_len * args.heads
        output_numel = rows * HEAD_DIM_V
        with torch.inference_mode(False), torch.no_grad():
            local_output_storage = symm_mem.empty(
                (output_numel,), dtype=torch.float32, device=device
            )
            local_lse_storage = symm_mem.empty(
                (rows,), dtype=torch.float32, device=device
            )
            local_flags = symm_mem.empty(
                (rows, 4), dtype=torch.int64, device=device
            )
        local_flags.zero_()
        torch.cuda.synchronize(device)
        output_handle = symm_mem.rendezvous(
            local_output_storage, group=dist.group.WORLD
        )
        lse_handle = symm_mem.rendezvous(
            local_lse_storage, group=dist.group.WORLD
        )
        flag_handle = symm_mem.rendezvous(
            local_flags, group=dist.group.WORLD
        )
        if output_handle is None or lse_handle is None or flag_handle is None:
            raise RuntimeError("symmetric-memory rendezvous returned no handle")
        output_handle.barrier()
        lse_handle.barrier()
        flag_handle.barrier()
        output_ptrs = torch.tensor(
            output_handle.buffer_ptrs, dtype=torch.int64, device=device
        )
        lse_ptrs = torch.tensor(
            lse_handle.buffer_ptrs, dtype=torch.int64, device=device
        )
        flag_ptrs = torch.tensor(
            flag_handle.buffer_ptrs, dtype=torch.int64, device=device
        )
        if bool(
            output_ptrs.eq(0).any()
            or lse_ptrs.eq(0).any()
            or flag_ptrs.eq(0).any()
        ):
            raise RuntimeError("symmetric-memory peer mapping contains null")
        local_peer_output = local_output_storage.view(
            args.batch, args.q_len, args.heads, HEAD_DIM_V
        )
        local_peer_lse = local_lse_storage.view(
            args.batch, args.heads, args.q_len
        )
        compact_output = None
        compact_lse = None
        if args.compact_output or args.copy_output:
            compact_output = torch.empty(
                args.batch,
                args.q_len,
                args.heads // WORLD_SIZE,
                HEAD_DIM_V,
                dtype=torch.float32,
                device=device,
            )
            compact_lse = torch.empty(
                args.batch,
                args.heads // WORLD_SIZE,
                args.q_len,
                dtype=torch.float32,
                device=device,
            )
        if args.kv_layout == "token-cyclic":
            with torch.inference_mode(False), torch.no_grad():
                local_kv_staging_u8 = symm_mem.empty(
                    (args.batch, args.chunk_size, HEAD_DIM),
                    dtype=torch.uint8,
                    device=device,
                )
                local_kv_flags = symm_mem.empty(
                    (2,), dtype=torch.int64, device=device
                )
            local_kv_flags.zero_()
            torch.cuda.synchronize(device)
            kv_staging_handle = symm_mem.rendezvous(
                local_kv_staging_u8, group=dist.group.WORLD
            )
            kv_flag_handle = symm_mem.rendezvous(
                local_kv_flags, group=dist.group.WORLD
            )
            if kv_staging_handle is None or kv_flag_handle is None:
                raise RuntimeError("KV symmetric-memory rendezvous returned no handle")
            kv_staging_handle.barrier()
            kv_flag_handle.barrier()
            kv_staging_ptrs = torch.tensor(
                kv_staging_handle.buffer_ptrs, dtype=torch.int64, device=device
            )
            kv_flag_ptrs = torch.tensor(
                kv_flag_handle.buffer_ptrs, dtype=torch.int64, device=device
            )
            local_kv_staging = local_kv_staging_u8.view(torch.float8_e4m3fn)
            canonical_cache = torch.empty(
                args.batch,
                args.chunk_size,
                1,
                HEAD_DIM,
                dtype=torch.float8_e4m3fn,
                device=device,
            )

    def run_tree() -> tuple[torch.Tensor, torch.Tensor]:
        attention_cache = local_cache
        if args.kv_layout == "token-cyclic":
            flash_mla_cuda.canonical_peer_pack_token_cyclic_kv(
                kv_staging_ptrs,
                kv_flag_ptrs,
                rank,
                local_cache,
                block_table,
                staging_seqlens,
                local_kv_staging,
                canonical_cache,
            )
            attention_cache = canonical_cache.view(
                -1, PAGE_SIZE, 1, HEAD_DIM
            )
        if args.transport.startswith("peer"):
            prepare = (
                flash_mla_cuda.canonical_peer_reduce_scatter_prepare
                if args.transport == "peer-rs"
                else flash_mla_cuda.canonical_peer_tree_prepare
            )
            prepare(
                flag_ptrs,
                args.batch * args.q_len * args.heads,
                rank,
            )
        if args.transport.startswith("peer") and args.leaf_output == "direct":
            leaf_outputs, leaf_lses = (
                flash_mla.flash_mla_dense_fp8_canonical_leaves_out(
                    q,
                    attention_cache,
                    block_table,
                    chunk_seqlens,
                    HEAD_DIM_V,
                    metadata,
                    splits,
                    1,
                    local_peer_output.unsqueeze(1),
                    **common,
                )
            )
            local_peer_lse.copy_(leaf_lses[:, 0])
        elif args.transport.startswith("peer"):
            leaf_outputs, leaf_lses = (
                flash_mla.flash_mla_dense_fp8_canonical_leaves(
                    q,
                    attention_cache,
                    block_table,
                    chunk_seqlens,
                    HEAD_DIM_V,
                    metadata,
                    splits,
                    1,
                    **common,
                )
            )
            local_peer_output.copy_(leaf_outputs[:, 0])
            local_peer_lse.copy_(leaf_lses[:, 0])

        if args.transport.startswith("peer"):
            if args.transport == "peer-rs" and args.compact_output:
                flash_mla_cuda.canonical_peer_reduce_scatter_compact(
                    output_ptrs,
                    lse_ptrs,
                    flag_ptrs,
                    rank,
                    local_peer_output,
                    local_peer_lse,
                    compact_output,
                    compact_lse,
                )
                return compact_output, compact_lse
            merge = (
                flash_mla_cuda.canonical_peer_reduce_scatter
                if args.transport == "peer-rs"
                else flash_mla_cuda.canonical_peer_tree_merge
            )
            merge(
                output_ptrs,
                lse_ptrs,
                flag_ptrs,
                rank,
                local_peer_output,
                local_peer_lse,
            )
            if args.copy_output:
                heads_per_rank = args.heads // WORLD_SIZE
                head_slice = slice(
                    rank * heads_per_rank, (rank + 1) * heads_per_rank
                )
                compact_output.copy_(local_peer_output[:, :, head_slice])
                compact_lse.copy_(local_peer_lse[:, head_slice])
                return compact_output, compact_lse
            return local_peer_output, local_peer_lse

        leaf_outputs, leaf_lses = flash_mla.flash_mla_dense_fp8_canonical_leaves(
            q,
            attention_cache,
            block_table,
            chunk_seqlens,
            HEAD_DIM_V,
            metadata,
            splits,
            1,
            **common,
        )

        output = leaf_outputs[:, 0]
        # The exported native [B, H, Q] view is strided across query rows.
        # Materialize only this small B*H*Q FP32 tensor for NCCL; the 2 MiB
        # output state remains a zero-copy contiguous view.
        lse = leaf_lses[:, 0].contiguous()
        for level, stride in enumerate((1, 2, 4)):
            period = stride * 2
            offset = rank % period
            if offset == stride:
                peer = rank - stride
                _exchange(
                    output,
                    lse,
                    peer,
                    edge_groups[(peer, rank)],
                    receive=False,
                )
                break
            if offset == 0 and rank + stride < WORLD_SIZE:
                peer = rank + stride
                _exchange(
                    recv_outputs[level],
                    recv_lses[level],
                    peer,
                    edge_groups[(rank, peer)],
                    receive=True,
                )
                flash_mla.canonical_merge_pair_out(
                    output,
                    lse,
                    recv_outputs[level],
                    recv_lses[level],
                    output,
                    lse,
                )
        return output, lse

    tree_graph = None
    if args.cuda_graph:
        run_tree()
        torch.cuda.synchronize()
        dist.barrier()
        tree_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(tree_graph):
            graph_output, graph_lse = run_tree()
        torch.cuda.synchronize()
        dist.barrier()

        def run_iteration() -> tuple[torch.Tensor, torch.Tensor]:
            tree_graph.replay()
            return graph_output, graph_lse

    else:
        run_iteration = run_tree

    output, lse = run_iteration()
    cp1_cache = global_kv.reshape(-1, PAGE_SIZE, 1, HEAD_DIM)
    cp1_block_table = torch.arange(
        args.batch * local_pages * WORLD_SIZE,
        dtype=torch.int32,
        device=device,
    ).view(args.batch, local_pages * WORLD_SIZE)
    cp1_seqlens = torch.full(
        (args.batch,), global_seqlen, dtype=torch.int32, device=device
    )
    cp1_causal = visible_global.view(1, -1).expand(args.batch, -1).contiguous()
    cp1_metadata, cp1_splits = (
        flash_mla.get_mla_metadata_canonical_dense_fp8(
            cp1_seqlens, args.chunk_size, WORLD_SIZE
        )
    )

    def run_cp1() -> tuple[torch.Tensor, torch.Tensor]:
        return flash_mla.flash_mla_dense_fp8_canonical(
            q,
            cp1_cache,
            cp1_block_table,
            cp1_seqlens,
            HEAD_DIM_V,
            cp1_metadata,
            cp1_splits,
            WORLD_SIZE,
            softmax_scale=1 / math.sqrt(HEAD_DIM),
            causal=True,
            causal_seqlens_k=cp1_causal,
            descale_q=descale_q,
            descale_k=descale_k,
        )

    if args.transport == "peer-rs":
        cp1_output, cp1_lse = run_cp1()
        heads_per_rank = args.heads // WORLD_SIZE
        head_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
        output_is_compact = args.compact_output or args.copy_output
        local_output = output if output_is_compact else output[:, :, head_slice]
        local_lse = lse if output_is_compact else lse[:, head_slice]
        ref_output = cp1_output[:, :, head_slice]
        ref_lse = cp1_lse[:, head_slice]
        accuracy = torch.tensor(
            [
                0.0 if torch.equal(local_output, ref_output) else 1.0,
                0.0 if torch.equal(local_lse, ref_lse) else 1.0,
                (local_output - ref_output).abs().max().item(),
                (local_lse - ref_lse).abs().max().item(),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(accuracy, op=dist.ReduceOp.MAX)
        bitwise_output = accuracy[0].item() == 0
        bitwise_lse = accuracy[1].item() == 0
        output_max_abs = accuracy[2].item()
        lse_max_abs = accuracy[3].item()
    elif rank == 0:
        cp1_output, cp1_lse = run_cp1()
        bitwise_output = torch.equal(output, cp1_output)
        bitwise_lse = torch.equal(lse, cp1_lse)
        output_max_abs = (output - cp1_output).abs().max().item()
        lse_max_abs = (lse - cp1_lse).abs().max().item()
    else:
        bitwise_output = False
        bitwise_lse = False
        output_max_abs = math.nan
        lse_max_abs = math.nan
    dist.barrier()

    for _ in range(args.warmup):
        run_iteration()
    torch.cuda.synchronize()
    dist.barrier()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(args.iterations):
        run_iteration()
    end.record()
    end.synchronize()
    local_tree_ms = begin.elapsed_time(end) / args.iterations
    if args.cuda_graph:
        output, lse = run_iteration()
        torch.cuda.synchronize()
        if args.transport == "peer-rs":
            output_is_compact = args.compact_output or args.copy_output
            local_output = output if output_is_compact else output[:, :, head_slice]
            local_lse = lse if output_is_compact else lse[:, head_slice]
            replay_accuracy = torch.tensor(
                [
                    0.0 if torch.equal(local_output, ref_output) else 1.0,
                    0.0 if torch.equal(local_lse, ref_lse) else 1.0,
                    (local_output - ref_output).abs().max().item(),
                    (local_lse - ref_lse).abs().max().item(),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(replay_accuracy, op=dist.ReduceOp.MAX)
            bitwise_output = replay_accuracy[0].item() == 0
            bitwise_lse = replay_accuracy[1].item() == 0
            output_max_abs = replay_accuracy[2].item()
            lse_max_abs = replay_accuracy[3].item()
        elif rank == 0:
            bitwise_output = torch.equal(output, cp1_output)
            bitwise_lse = torch.equal(lse, cp1_lse)
            output_max_abs = (output - cp1_output).abs().max().item()
            lse_max_abs = (lse - cp1_lse).abs().max().item()
    timing = torch.tensor([local_tree_ms], dtype=torch.float64, device=device)
    timings = [torch.empty_like(timing) for _ in range(WORLD_SIZE)]
    dist.all_gather(timings, timing)

    dist.barrier()
    cp1_ms = math.nan
    cp1_benchmark = run_cp1
    if rank == 0:
        if args.cuda_graph:
            run_cp1()
            torch.cuda.synchronize()
            cp1_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cp1_graph):
                cp1_graph_output, cp1_graph_lse = run_cp1()

            def cp1_benchmark() -> tuple[torch.Tensor, torch.Tensor]:
                cp1_graph.replay()
                return cp1_graph_output, cp1_graph_lse

        for _ in range(args.warmup):
            cp1_benchmark()
        torch.cuda.synchronize()
        cp1_begin = torch.cuda.Event(enable_timing=True)
        cp1_end = torch.cuda.Event(enable_timing=True)
        cp1_begin.record()
        for _ in range(args.iterations):
            cp1_benchmark()
        cp1_end.record()
        cp1_end.synchronize()
        cp1_ms = cp1_begin.elapsed_time(cp1_end) / args.iterations
    dist.barrier()

    if rank == 0:
        rank_ms = [value.item() for value in timings]
        revisions = {
            ("nccl", "allocated"): "R14-P3",
            ("peer", "allocated"): "R14-P4",
            ("peer-rs", "allocated"): "R14-P5",
            ("peer", "direct"): "R14-P6-root",
            ("peer-rs", "direct"): "R14-P6",
        }
        payload = {
            "revision_id": (
                (
                    "R14-P10"
                    if args.compact_output
                    else ("R14-P7-copy" if args.copy_output else "R14-P7")
                )
                if args.kv_layout == "token-cyclic"
                else revisions[(args.transport, args.leaf_output)]
            ),
            "gpu": torch.cuda.get_device_name(),
            "world_size": WORLD_SIZE,
            "transport": args.transport,
            "leaf_output": args.leaf_output,
            "kv_layout": args.kv_layout,
            "compact_output": args.compact_output,
            "copy_output": args.copy_output,
            "cuda_graph": args.cuda_graph,
            "output_ownership": (
                "contiguous H/8 shard per rank"
                if args.transport == "peer-rs"
                else "full state on rank 0"
            ),
            "shape": {
                "batch": args.batch,
                "q_len": args.q_len,
                "heads": args.heads,
                "global_kv_len": global_seqlen,
                "chunk_size": args.chunk_size,
                "head_dim": HEAD_DIM,
                "head_dim_v": HEAD_DIM_V,
            },
            "measurement": {
                "warmup": args.warmup,
                "iterations": args.iterations,
                "rank_tree_ms": rank_ms,
                "critical_tree_ms": max(rank_ms),
                "rank0_tree_ms": rank_ms[0],
                "cp1_fused_ms": cp1_ms,
                "cp8_vs_cp1_pct": (max(rank_ms) / cp1_ms - 1) * 100,
                "critical_path_payload_bytes": 3
                    * args.batch
                    * args.q_len
                    * args.heads
                    * (HEAD_DIM_V + 1)
                    * 4
                    if args.transport != "peer-rs"
                    else 7
                    * args.batch
                    * args.q_len
                    * (args.heads // WORLD_SIZE)
                    * (HEAD_DIM_V + 1)
                    * 4,
            },
            "accuracy": {
                "output_bitwise_equal_cp1": bitwise_output,
                "lse_bitwise_equal_cp1": bitwise_lse,
                "output_max_abs": output_max_abs,
                "lse_max_abs": lse_max_abs,
            },
        }
        print(json.dumps(payload, indent=2))
        if args.output:
            with open(args.output, "w") as output_file:
                json.dump(payload, output_file, indent=2)
                output_file.write("\n")
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--global-kv-len",
        type=int,
        help="valid global KV tokens; defaults to chunk-size * 8",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1430)
    parser.add_argument(
        "--transport", choices=("nccl", "peer", "peer-rs"), default="nccl"
    )
    parser.add_argument(
        "--leaf-output", choices=("allocated", "direct"), default="allocated"
    )
    parser.add_argument(
        "--kv-layout", choices=("chunk-cyclic", "token-cyclic"),
        default="chunk-cyclic"
    )
    parser.add_argument("--output")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--compact-output", action="store_true")
    parser.add_argument("--copy-output", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.chunk_size % PAGE_SIZE:
        parser.error("chunk-size must be a positive multiple of 64")
    if args.global_kv_len is not None and not (
        args.q_len <= args.global_kv_len <= args.chunk_size * WORLD_SIZE
    ):
        parser.error("global-kv-len must be in [q-len, chunk-size * 8]")
    if args.heads <= 0 or args.heads % WORLD_SIZE:
        parser.error("heads must be positive and divisible by 8")
    if args.transport == "nccl" and args.leaf_output != "allocated":
        parser.error("direct leaf output requires a peer transport")
    if args.kv_layout == "token-cyclic" and (
        args.transport != "peer-rs" or args.leaf_output != "direct"
    ):
        parser.error("token-cyclic adapter currently requires peer-rs + direct")
    if args.compact_output and args.transport != "peer-rs":
        parser.error("compact-output requires peer-rs")
    if args.copy_output and args.transport != "peer-rs":
        parser.error("copy-output requires peer-rs")
    if args.compact_output and args.copy_output:
        parser.error("compact-output and copy-output are mutually exclusive")
    return args


if __name__ == "__main__":
    main(parse_args())
