#!/usr/bin/env python3
"""Stress the persistent canonical DCP8 workspace in eager or CUDA Graph mode.

Run on one eight-GPU SM90 ECS node, for example::

    torchrun --standalone --nproc-per-node=8 stress_canonical_workspace.py \
        --mode graph --iterations 4096 --output result.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flash_mla


WORLD_SIZE = 8
PAGE_SIZE = 64
CHUNK_SIZE = 1216
HEAD_DIM = 576
HEAD_DIM_V = 512
NUM_HEADS = 32
GLOBAL_LENGTH = 8192


def make_fp8_random(
    shape: tuple[int, ...], seed: int, scale: float = 0.1
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return (
        torch.randn(
            shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        .mul_(scale)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--iterations", type=int, default=4096)
    parser.add_argument("--check-interval", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        raise RuntimeError("launch with torchrun --nproc-per-node=8")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group("nccl", device_id=device)
    except TypeError:
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("canonical workspace stress requires SM90")

    batch_size = args.batch_size
    q_len = 8
    capacity = CHUNK_SIZE * WORLD_SIZE
    q_variants = [
        make_fp8_random(
            (batch_size, q_len, NUM_HEADS, HEAD_DIM), 1510 + variant
        )
        for variant in range(2)
    ]
    global_kv = make_fp8_random(
        (batch_size, capacity, 1, HEAD_DIM), 1520
    )
    local_cache = global_kv[:, rank::WORLD_SIZE].reshape(
        -1, PAGE_SIZE, 1, HEAD_DIM
    ).contiguous()
    pages_per_chunk = CHUNK_SIZE // PAGE_SIZE
    local_block_table = torch.arange(
        batch_size * pages_per_chunk,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, pages_per_chunk)
    cp1_block_table = torch.arange(
        batch_size * pages_per_chunk * WORLD_SIZE,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, pages_per_chunk * WORLD_SIZE)
    global_seqlens = torch.full(
        (batch_size,), GLOBAL_LENGTH, dtype=torch.int32, device=device
    )
    descale_q = torch.tensor([0.75], dtype=torch.float32, device=device)
    descale_k = torch.tensor([1.25], dtype=torch.float32, device=device)
    softmax_scale = 1 / math.sqrt(HEAD_DIM)

    workspace = flash_mla.create_canonical_dcp8_workspace(
        max_batch_size=batch_size,
        max_seqlen_q=q_len,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        head_dim_v=HEAD_DIM_V,
        chunk_size=CHUNK_SIZE,
        group=dist.group.WORLD,
    )
    chunk_seqlens = workspace.chunk_cache_seqlens(global_seqlens)
    metadata, num_splits = flash_mla.get_mla_metadata_canonical_dense_fp8(
        chunk_seqlens, CHUNK_SIZE, 1
    )
    cp1_metadata, cp1_splits = (
        flash_mla.get_mla_metadata_canonical_dense_fp8(
            global_seqlens, CHUNK_SIZE, WORLD_SIZE
        )
    )
    visible_global = torch.arange(
        GLOBAL_LENGTH - q_len + 1,
        GLOBAL_LENGTH + 1,
        dtype=torch.int32,
        device=device,
    ).view(1, q_len).expand(batch_size, -1).contiguous()
    references = []
    local_heads = NUM_HEADS // WORLD_SIZE
    head_begin = rank * local_heads
    head_end = head_begin + local_heads
    for q in q_variants:
        reference, _ = flash_mla.flash_mla_dense_fp8_canonical(
            q,
            global_kv.view(-1, PAGE_SIZE, 1, HEAD_DIM),
            cp1_block_table,
            global_seqlens,
            HEAD_DIM_V,
            cp1_metadata,
            cp1_splits,
            WORLD_SIZE,
            softmax_scale=softmax_scale,
            causal=True,
            causal_seqlens_k=visible_global,
            descale_q=descale_q,
            descale_k=descale_k,
        )
        references.append(reference[:, :, head_begin:head_end].contiguous())

    static_q = q_variants[0].clone()

    def invoke() -> torch.Tensor:
        output, _ = workspace.attention(
            static_q,
            local_cache,
            local_block_table,
            global_seqlens,
            metadata,
            num_splits,
            softmax_scale=softmax_scale,
            descale_q=descale_q,
            descale_k=descale_k,
        )
        return output

    graph = None
    if args.mode == "graph":
        for warmup in range(4):
            static_q.copy_(q_variants[warmup & 1])
            output = invoke()
        torch.cuda.synchronize(device)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = invoke()
        dist.barrier()
    else:
        output = invoke()
        torch.cuda.synchronize(device)
        dist.barrier()

    started = time.perf_counter()
    checks = 0
    max_abs_error = 0.0
    first_failure = None
    for iteration in range(args.iterations):
        variant = iteration & 1
        static_q.copy_(q_variants[variant])
        if graph is None:
            output = invoke()
        else:
            graph.replay()
        if iteration % args.check_interval == 0 or iteration + 1 == args.iterations:
            torch.cuda.synchronize(device)
            error = (output - references[variant]).abs().max().item()
            max_abs_error = max(max_abs_error, error)
            checks += 1
            if error != 0.0 and first_failure is None:
                first_failure = iteration
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    stats = torch.tensor(
        [elapsed, max_abs_error, -1 if first_failure is None else first_failure],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(stats) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, stats)
    rows = [row.cpu().tolist() for row in gathered]
    payload = {
        "mode": args.mode,
        "iterations": args.iterations,
        "check_interval": args.check_interval,
        "batch_size": batch_size,
        "checks": checks,
        "ranks": [
            {
                "rank": index,
                "elapsed_seconds": row[0],
                "max_abs_error": row[1],
                "first_failure_iteration": None
                if row[2] < 0
                else int(row[2]),
            }
            for index, row in enumerate(rows)
        ],
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)

    dist.barrier()
    workspace.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
