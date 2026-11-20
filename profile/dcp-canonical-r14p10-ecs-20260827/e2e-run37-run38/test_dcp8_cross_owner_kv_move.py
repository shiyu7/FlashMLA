#!/usr/bin/env python3
"""Eight-GPU regression for cross-owner speculative MLA KV relocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.srt.mem_cache.memory_pool as memory_pool


WORLD_SIZE = 8


def expected_row(
    logical_location: int,
    layer: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    byte = torch.arange(width, dtype=torch.int64, device=device)
    return ((logical_location * 13 + layer * 31 + byte) % 256).to(torch.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
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

    fake_parallel = SimpleNamespace(
        dcp_enabled=True,
        attn_dcp_size=WORLD_SIZE,
        attn_dcp_rank=rank,
        dcp_group=SimpleNamespace(device_group=dist.group.WORLD),
    )
    memory_pool.get_parallel = lambda: fake_parallel

    pool = memory_pool.MLATokenToKVPool(
        size=64,
        page_size=64,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=3,
        device=str(device),
        enable_memory_saver=False,
    )
    width = pool.kv_cache_dim
    physical_rows = torch.arange(
        pool.size + pool.page_size, dtype=torch.int64, device=device
    )
    logical_rows = physical_rows * WORLD_SIZE + rank
    byte = torch.arange(width, dtype=torch.int64, device=device)
    for layer, kv_cache in enumerate(pool.kv_buffer):
        pattern = (
            logical_rows[:, None] * 13 + layer * 31 + byte[None, :]
        ).remainder(256).to(torch.uint8)
        kv_cache.view(torch.uint8)[:, 0].copy_(pattern)

    cross_owner_tgt = [WORLD_SIZE * (10 + i) + i for i in range(WORLD_SIZE)]
    cross_owner_src = [
        WORLD_SIZE * (20 + i) + ((i + 3) % WORLD_SIZE)
        for i in range(WORLD_SIZE)
    ]
    same_owner_tgt = [WORLD_SIZE * (30 + i) + i for i in range(WORLD_SIZE)]
    same_owner_src = [WORLD_SIZE * (40 + i) + i for i in range(WORLD_SIZE)]
    tgt = torch.tensor(
        cross_owner_tgt + same_owner_tgt, dtype=torch.int64, device=device
    )
    src = torch.tensor(
        cross_owner_src + same_owner_src, dtype=torch.int64, device=device
    )

    pool.move_kv_cache(tgt, src)
    torch.cuda.synchronize(device)

    failures = []
    for pair, (target, source) in enumerate(zip(tgt.tolist(), src.tolist())):
        if target % WORLD_SIZE != rank:
            continue
        target_local = target // WORLD_SIZE
        for layer, kv_cache in enumerate(pool.kv_buffer):
            actual = kv_cache.view(torch.uint8)[target_local, 0]
            expected = expected_row(source, layer, width, device)
            if not torch.equal(actual, expected):
                failures.append(
                    {
                        "pair": pair,
                        "layer": layer,
                        "target": target,
                        "source": source,
                        "max_byte_error": int(
                            (actual.to(torch.int16) - expected.to(torch.int16))
                            .abs()
                            .max()
                            .item()
                        ),
                    }
                )

    local_failure_count = torch.tensor(
        [len(failures)], dtype=torch.int64, device=device
    )
    gathered_counts = [torch.empty_like(local_failure_count) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_counts, local_failure_count)
    counts = [int(item.item()) for item in gathered_counts]
    if rank == 0:
        payload = {
            "world_size": WORLD_SIZE,
            "cross_owner_pairs": len(cross_owner_tgt),
            "same_owner_pairs": len(same_owner_tgt),
            "failure_count_by_rank": counts,
            "status": "passed" if sum(counts) == 0 else "failed",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise AssertionError(f"rank {rank} relocation failures: {failures}")


if __name__ == "__main__":
    main()
