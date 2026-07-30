"""Sweep the DCP2 peer-vs-NCCL crossover on real attention shapes.

Example:

torchrun --standalone --nproc-per-node=2 \
    benchmark/bench_dcp2_crossover.py \
    --batches 1,2,4,8,16,32 \
    --q-lens 1,2,4,8 \
    --heads 32,64,128 \
    --head-dims-v 128,256,512 \
    --csv dcp2_crossover.csv

The peer timing contains device-epoch prepare plus the fused LSE/output merge.
FlashMLA already owns the symmetric state, so no staging copy is timed. The
fallback timing contains LSE all-gather, correction, and output reduce-scatter.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import torch
import torch.distributed as dist

import flash_mla


@dataclass
class Result:
    batch: int
    q_len: int
    heads: int
    head_dim_v: int
    rows: int
    peer_payload_bytes: int
    peer_ms: float
    nccl_ms: float
    peer_speedup: float
    winner: str


def _parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",")})
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive ints")
    return result


def _time_distributed(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
    group: dist.ProcessGroup,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    local_ms = start.elapsed_time(end) / iterations
    critical_ms = torch.tensor(local_ms, dtype=torch.float64, device="cuda")
    dist.all_reduce(critical_ms, op=dist.ReduceOp.MAX, group=group)
    return critical_ms.item()


def _fill_state(
    output: torch.Tensor,
    lse: torch.Tensor,
    rank: int,
    shape_seed: int,
) -> None:
    generator = torch.Generator(device=output.device)
    generator.manual_seed(2027 + rank * 1009 + shape_seed)
    output.copy_(
        torch.randn(
            output.shape,
            dtype=torch.float32,
            device=output.device,
            generator=generator,
        ).to(torch.bfloat16)
    )
    lse.copy_(
        torch.randn(
            lse.shape,
            dtype=torch.float32,
            device=lse.device,
            generator=generator,
        )
    )
    lse[:, 0, rank::2] = -torch.inf


def _print_result(result: Result) -> None:
    print(
        f"B={result.batch:<3} Q={result.q_len:<2} H={result.heads:<3} "
        f"Dv={result.head_dim_v:<3} rows={result.rows:<6} "
        f"payload={result.peer_payload_bytes / 1024:>8.1f} KiB "
        f"peer={result.peer_ms:>8.4f} ms "
        f"nccl={result.nccl_ms:>8.4f} ms "
        f"speedup={result.peer_speedup:>6.2f}x "
        f"winner={result.winner}"
    )


def _print_crossover_summary(results: Iterable[Result]) -> None:
    grouped: dict[tuple[int, int, int], list[Result]] = defaultdict(list)
    for result in results:
        grouped[(result.q_len, result.heads, result.head_dim_v)].append(result)
    print("\nCrossover by exact (Q, H, Dv); B is swept, not collapsed to bytes:")
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda item: item.batch)
        winning_batches = [
            value.batch for value in values if value.winner == "peer"
        ]
        q_len, heads, head_dim_v = key
        if winning_batches:
            text = ",".join(str(batch) for batch in winning_batches)
            print(f"Q={q_len:<2} H={heads:<3} Dv={head_dim_v:<3}: peer B={text}")
        else:
            print(f"Q={q_len:<2} H={heads:<3} Dv={head_dim_v:<3}: NCCL all B")


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    if dist.get_world_size(group) != 2:
        raise RuntimeError("crossover benchmark requires exactly two ranks")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("DCP2 peer attention requires SM90")
    if any(q_len > 8 for q_len in args.q_lens):
        raise ValueError("first-version DCP2 fast path requires Q <= 8")
    if any(heads % 2 for heads in args.heads):
        raise ValueError("all head counts must be divisible by DCP2")
    if any(head_dim_v % 2 for head_dim_v in args.head_dims_v):
        raise ValueError("all Dv values must be even")

    if rank == 0:
        print(
            f"GPU={torch.cuda.get_device_name()} "
            f"warmup={args.warmup} iterations={args.iterations}"
        )

    results: list[Result] = []
    for heads in args.heads:
        for q_len in args.q_lens:
            for head_dim_v in args.head_dims_v:
                for batch in args.batches:
                    # Compact direct-write storage is bound to one graph shape
                    # so row epochs cannot alias after a stride change.
                    workspace = flash_mla.create_dcp2_attention_workspace(
                        batch,
                        q_len,
                        heads,
                        head_dim_v,
                        group,
                    )
                    try:
                        local_out, local_lse = workspace.local_state(
                            batch, q_len, heads, head_dim_v
                        )
                        shape_seed = (
                            batch * 1_000_003
                            + q_len * 10_007
                            + heads * 101
                            + head_dim_v
                        )
                        _fill_state(
                            local_out, local_lse, rank, shape_seed=shape_seed
                        )
                        nccl_source_out = local_out.clone()
                        nccl_source_lse = local_lse.clone()
                        local_heads = heads // 2
                        peer_out = torch.empty(
                            batch,
                            q_len,
                            local_heads,
                            head_dim_v,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        peer_lse = torch.empty(
                            batch,
                            q_len,
                            local_heads,
                            dtype=torch.float32,
                            device="cuda",
                        )
                        nccl_out = torch.empty_like(peer_out)
                        nccl_lse = torch.empty_like(peer_lse)

                        def run_peer():
                            workspace.prepare(
                                batch, q_len, heads, head_dim_v
                            )
                            return workspace.merge(
                                batch,
                                q_len,
                                heads,
                                head_dim_v,
                                out=peer_out,
                                softmax_lse=peer_lse,
                            )

                        def run_nccl():
                            return flash_mla.nccl_merge_attention(
                                nccl_source_out,
                                nccl_source_lse,
                                group,
                                out=nccl_out,
                                softmax_lse=nccl_lse,
                            )

                        run_peer()
                        run_nccl()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            peer_out.float(),
                            nccl_out.float(),
                            rtol=0.02,
                            atol=0.02,
                        )
                        torch.testing.assert_close(
                            peer_lse,
                            nccl_lse,
                            rtol=2e-5,
                            atol=2e-5,
                        )

                        peer_ms = _time_distributed(
                            run_peer, args.warmup, args.iterations, group
                        )
                        nccl_ms = _time_distributed(
                            run_nccl, args.warmup, args.iterations, group
                        )
                        result = Result(
                            batch=batch,
                            q_len=q_len,
                            heads=heads,
                            head_dim_v=head_dim_v,
                            rows=batch * q_len * heads,
                            peer_payload_bytes=(
                                flash_mla.dcp2_attention_payload_bytes(
                                    batch, q_len, heads, head_dim_v
                                )
                            ),
                            peer_ms=peer_ms,
                            nccl_ms=nccl_ms,
                            peer_speedup=nccl_ms / peer_ms,
                            winner="peer" if peer_ms < nccl_ms else "nccl",
                        )
                        results.append(result)
                        if rank == 0:
                            _print_result(result)
                    finally:
                        workspace.close()
                        dist.barrier(group=group)

    if rank == 0:
        _print_crossover_summary(results)
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=list(asdict(results[0]).keys())
                )
                writer.writeheader()
                writer.writerows(asdict(result) for result in results)
            print(f"\nWrote {args.csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        type=_parse_ints,
        default=_parse_ints("1,2,4,8,16,32"),
    )
    parser.add_argument(
        "--q-lens",
        type=_parse_ints,
        default=_parse_ints("1,2,4,8"),
    )
    parser.add_argument(
        "--heads",
        type=_parse_ints,
        default=_parse_ints("32,64,128"),
    )
    parser.add_argument(
        "--head-dims-v",
        dest="head_dims_v",
        type=_parse_ints,
        default=_parse_ints("128,256,512"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--csv", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
