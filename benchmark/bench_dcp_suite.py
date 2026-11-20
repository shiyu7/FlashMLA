"""Layered real-rank benchmark for dense FP8 DCP.

Examples:

torchrun --standalone --nproc-per-node=2 benchmark/bench_dcp_suite.py \
    --profile smoke --cuda-graph --output /tmp/dcp2.json

torchrun --standalone --nproc-per-node=8 benchmark/bench_dcp_suite.py \
    --profile smoke --cuda-graph --output /tmp/dcp8.json
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flash_mla  # noqa: E402
from dcp_eval.performance import (  # noqa: E402
    distributed_cuda_samples,
    environment_metadata,
    summarize_latency,
    write_json,
)


PAGE_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512


def _parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",")})
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive ints")
    return result


def _parse_strings(value: str) -> list[str]:
    result = list(dict.fromkeys(item.strip() for item in value.split(",") if item))
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return result


def _axes(args: argparse.Namespace):
    if args.profile == "smoke":
        defaults = {
            "batches": [1, 8],
            "q_lens": [1, 4, 8],
            "heads": [128],
            "kv_lens": [128, 4096],
            "q_profiles": ["full"],
        }
    else:
        defaults = {
            "batches": [1, 8, 32],
            "q_lens": [1, 4, 8],
            "heads": [64, 128],
            "kv_lens": [128, 4096, 32768],
            "q_profiles": ["full", "ragged"],
        }
    return tuple(
        getattr(args, name) if getattr(args, name) is not None else defaults[name]
        for name in ("batches", "q_lens", "heads", "kv_lens", "q_profiles")
    )


def _logical_q_lens(
    batch: int, q_len: int, profile: str, device: torch.device
) -> torch.Tensor:
    if profile == "full":
        values = [q_len] * batch
    elif profile == "ragged":
        pattern = [q_len, 0, max(1, q_len // 2), max(1, q_len - 1)]
        values = [pattern[index % len(pattern)] for index in range(batch)]
        if batch == 1:
            values[0] = max(1, q_len // 2)
    else:
        raise ValueError("q profile must be full or ragged")
    return torch.tensor(values, dtype=torch.int32, device=device)


def _make_rank_inputs(
    *,
    batch: int,
    q_len: int,
    heads: int,
    kv_len: int,
    q_profile: str,
    world_size: int,
    rank: int,
    device: torch.device,
    local_output_dtype: str,
) -> dict[str, torch.Tensor | float]:
    global_seqlens = torch.full(
        (batch,), kv_len, dtype=torch.int32, device=device
    )
    q_lens = _logical_q_lens(batch, q_len, q_profile, device)
    local_seqlens = flash_mla.cyclic_local_length(
        global_seqlens, world_size, rank
    ).to(torch.int32)
    max_local_len = int(local_seqlens.max().item())
    pages_per_request = max((max_local_len + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    block_table = torch.arange(
        batch * pages_per_request, dtype=torch.int32, device=device
    ).view(batch, pages_per_request)
    local_cache = torch.zeros(
        batch * pages_per_request,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    q = torch.zeros(
        batch,
        q_len,
        heads,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    causal_seqlens = flash_mla.make_cyclic_local_causal_seqlens(
        global_seqlens,
        q_len,
        world_size,
        rank,
        q_lens=q_lens,
    )
    metadata, num_splits = flash_mla.get_mla_metadata_dense_fp8(
        local_seqlens, q_len * heads, 1
    )
    output = torch.empty(
        batch,
        q_len,
        heads,
        HEAD_DIM_V,
        dtype=(
            torch.float32
            if local_output_dtype == "fp32"
            else torch.bfloat16
        ),
        device=device,
    )
    lse = torch.empty(
        batch, q_len, heads, dtype=torch.float32, device=device
    )
    descale_q = torch.ones(1, dtype=torch.float32, device=device)
    descale_k = torch.ones(1, dtype=torch.float32, device=device)
    return {
        "q": q,
        "local_cache": local_cache,
        "block_table": block_table,
        "local_seqlens": local_seqlens,
        "global_seqlens": global_seqlens,
        "q_lens": q_lens,
        "causal_seqlens": causal_seqlens,
        "metadata": metadata,
        "num_splits": num_splits,
        "output": output,
        "lse": lse,
        "descale_q": descale_q,
        "descale_k": descale_k,
        "softmax_scale": 1 / math.sqrt(HEAD_DIM),
    }


def _effective_work(
    global_seqlens: torch.Tensor,
    q_lens: torch.Tensor,
    heads: int,
) -> tuple[int, int]:
    visible_pairs = 0
    valid_queries = 0
    for seqlen, logical_q in zip(
        global_seqlens.cpu().tolist(), q_lens.cpu().tolist()
    ):
        valid_queries += logical_q
        visible_pairs += logical_q * (seqlen - logical_q + 1)
        visible_pairs += logical_q * (logical_q - 1) // 2
    flops = 2 * heads * (HEAD_DIM + HEAD_DIM_V) * visible_pairs
    return valid_queries, flops


def _capture_graph(
    function: Callable[[], object], group: dist.ProcessGroup
) -> tuple[Callable[[], object], torch.cuda.CUDAGraph, float]:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    capture_begin = time.perf_counter()
    with torch.cuda.graph(graph):
        function()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    capture_ms = (time.perf_counter() - capture_begin) * 1000
    return graph.replay, graph, capture_ms


def _measure(
    *,
    name: str,
    function: Callable[[], object],
    args: argparse.Namespace,
    group: dist.ProcessGroup,
    shape: dict[str, object],
    valid_queries: int,
    effective_flops: int,
) -> dict[str, object]:
    graph = None
    capture_ms = None
    timed_function = function
    if args.cuda_graph:
        timed_function, graph, capture_ms = _capture_graph(function, group)

    critical_samples, rank_samples = distributed_cuda_samples(
        timed_function,
        warmup=args.warmup,
        samples=args.samples,
        calls_per_sample=args.calls_per_sample,
        group=group,
    )
    stats = summarize_latency(critical_samples)
    rank_stats = [summarize_latency(values) for values in rank_samples]
    p50_seconds = stats.p50_ms / 1000
    is_operator = name in ("local", "nccl_operator", "peer_operator")
    result = {
        **shape,
        "mode": name,
        "cuda_graph": args.cuda_graph,
        "capture_ms": capture_ms,
        **stats.to_dict(),
        "rank_p50_ms": [value.p50_ms for value in rank_stats],
        "rank_imbalance": max(value.p50_ms for value in rank_stats)
        / max(sum(value.p50_ms for value in rank_stats) / len(rank_stats), 1e-12),
        "valid_queries": valid_queries,
        "query_tokens_per_second": (
            valid_queries / p50_seconds if is_operator and p50_seconds else None
        ),
        "effective_tflops": (
            effective_flops / p50_seconds / 1e12
            if is_operator and p50_seconds
            else None
        ),
    }
    torch.cuda.synchronize()
    dist.barrier(group=group)
    if graph is not None:
        del timed_function
        del graph
    return result


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group("nccl", device_id=device)
    except TypeError:
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size not in (1, 2, 4, 8):
        raise RuntimeError("DCP benchmark world size must be 1, 2, 4, or 8")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("dense FP8 DCP requires SM90")

    batches, q_lens, heads_list, kv_lens, q_profiles = _axes(args)
    if any(q_len not in (1, 2, 4, 8) for q_len in q_lens):
        raise ValueError("Q must be drawn from 1,2,4,8")
    if min(kv_lens) < max(q_lens):
        raise ValueError("every KV length must be at least the maximum Q")
    if any(heads % world_size for heads in heads_list):
        raise ValueError("all head counts must be divisible by world size")

    default_modes = ["local", "nccl_merge", "nccl_operator"]
    if world_size == 2:
        default_modes.extend(["peer_merge", "peer_operator"])
    modes = args.modes or default_modes
    allowed_modes = {
        "local",
        "nccl_merge",
        "nccl_operator",
        "peer_merge",
        "peer_operator",
    }
    if not set(modes) <= allowed_modes:
        raise ValueError(f"unknown modes: {sorted(set(modes) - allowed_modes)}")
    if world_size != 2 and any(mode.startswith("peer") for mode in modes):
        raise ValueError("peer modes require exactly two ranks")
    if args.local_output_dtype == "fp32" and any(
        mode.startswith("peer") for mode in modes
    ):
        raise ValueError("peer modes currently require BF16 local output")

    records: list[dict[str, object]] = []
    case_count = 0
    for batch in batches:
        for q_len in q_lens:
            for heads in heads_list:
                for kv_len in kv_lens:
                    for q_profile in q_profiles:
                        case_count += 1
                        if args.max_cases and case_count > args.max_cases:
                            break
                        inputs = _make_rank_inputs(
                            batch=batch,
                            q_len=q_len,
                            heads=heads,
                            kv_len=kv_len,
                            q_profile=q_profile,
                            world_size=world_size,
                            rank=rank,
                            device=device,
                            local_output_dtype=args.local_output_dtype,
                        )

                        def run_local():
                            common_args = (
                                inputs["q"],
                                inputs["local_cache"],
                                inputs["block_table"],
                                inputs["local_seqlens"],
                                HEAD_DIM_V,
                                inputs["metadata"],
                                inputs["num_splits"],
                            )
                            common_kwargs = {
                                "softmax_scale": inputs["softmax_scale"],
                                "causal": True,
                                "causal_seqlens_k": inputs["causal_seqlens"],
                                "q_lens": inputs["q_lens"],
                                "descale_q": inputs["descale_q"],
                                "descale_k": inputs["descale_k"],
                            }
                            if args.local_output_dtype == "fp32":
                                output, lse = flash_mla.flash_mla_dense_fp8_fp32(
                                    *common_args, **common_kwargs
                                )
                                inputs["output"] = output
                                inputs["lse"] = lse.transpose(1, 2).contiguous()
                                return output, lse
                            return flash_mla.flash_mla_dense_fp8_out(
                                *common_args,
                                inputs["output"],
                                inputs["lse"],
                                **common_kwargs,
                            )

                        local_heads = heads // world_size
                        merged_out = torch.empty(
                            batch,
                            q_len,
                            local_heads,
                            HEAD_DIM_V,
                            dtype=inputs["output"].dtype,
                            device=device,
                        )
                        merged_lse = torch.empty(
                            batch,
                            q_len,
                            local_heads,
                            dtype=torch.float32,
                            device=device,
                        )

                        def run_nccl_merge():
                            return flash_mla.nccl_merge_attention(
                                inputs["output"],
                                inputs["lse"],
                                group,
                                out=merged_out,
                                softmax_lse=merged_lse,
                            )

                        def run_nccl_operator():
                            run_local()
                            if world_size == 1:
                                return inputs["output"], inputs["lse"]
                            return run_nccl_merge()

                        run_local()
                        torch.cuda.synchronize()
                        workspace = None
                        peer_out = None
                        peer_lse = None
                        if world_size == 2 and any(
                            mode.startswith("peer") for mode in modes
                        ):
                            workspace = flash_mla.create_dcp2_attention_workspace(
                                batch, q_len, heads, HEAD_DIM_V, group
                            )
                            peer_local_out, peer_local_lse = workspace.prepare(
                                batch, q_len, heads, HEAD_DIM_V
                            )
                            peer_local_out.copy_(inputs["output"])
                            peer_local_lse.copy_(inputs["lse"])
                            peer_out = torch.empty_like(merged_out)
                            peer_lse = torch.empty_like(merged_lse)
                            workspace.merge(
                                batch,
                                q_len,
                                heads,
                                HEAD_DIM_V,
                                out=peer_out,
                                softmax_lse=peer_lse,
                            )
                            run_nccl_merge()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(
                                peer_out.float(),
                                merged_out.float(),
                                rtol=0.02,
                                atol=0.02,
                            )
                            torch.testing.assert_close(
                                peer_lse,
                                merged_lse,
                                rtol=2e-5,
                                atol=2e-5,
                            )

                        def run_peer_merge():
                            workspace.prepare(batch, q_len, heads, HEAD_DIM_V)
                            return workspace.merge(
                                batch,
                                q_len,
                                heads,
                                HEAD_DIM_V,
                                out=peer_out,
                                softmax_lse=peer_lse,
                            )

                        def run_peer_operator():
                            return workspace.run_flash_mla_dense_fp8(
                                inputs["q"],
                                inputs["local_cache"],
                                inputs["block_table"],
                                inputs["local_seqlens"],
                                HEAD_DIM_V,
                                inputs["metadata"],
                                inputs["num_splits"],
                                softmax_scale=inputs["softmax_scale"],
                                causal=True,
                                causal_seqlens_k=inputs["causal_seqlens"],
                                q_lens=inputs["q_lens"],
                                descale_q=inputs["descale_q"],
                                descale_k=inputs["descale_k"],
                                out=peer_out,
                                softmax_lse=peer_lse,
                            )

                        functions = {
                            "local": run_local,
                            "nccl_merge": run_nccl_merge,
                            "nccl_operator": run_nccl_operator,
                            "peer_merge": run_peer_merge,
                            "peer_operator": run_peer_operator,
                        }
                        valid_queries, effective_flops = _effective_work(
                            inputs["global_seqlens"], inputs["q_lens"], heads
                        )
                        shape = {
                            "world_size": world_size,
                            "batch": batch,
                            "q_len": q_len,
                            "heads": heads,
                            "head_dim": HEAD_DIM,
                            "head_dim_v": HEAD_DIM_V,
                            "kv_len": kv_len,
                            "q_profile": q_profile,
                            "local_output_dtype": args.local_output_dtype,
                            "state_bytes_per_rank": (
                                batch
                                * q_len
                                * heads
                                * (
                                    HEAD_DIM_V
                                    * inputs["output"].element_size()
                                    + 4
                                )
                            ),
                        }
                        for mode in modes:
                            if world_size == 1 and mode == "nccl_merge":
                                continue
                            record = _measure(
                                name=mode,
                                function=functions[mode],
                                args=args,
                                group=group,
                                shape=shape,
                                valid_queries=valid_queries,
                                effective_flops=effective_flops,
                            )
                            records.append(record)
                            if rank == 0:
                                print(
                                    f"CP={world_size} B={batch} Q={q_len} "
                                    f"H={heads} KV={kv_len} {q_profile} {mode}: "
                                    f"p50={record['p50_ms']:.4f} ms "
                                    f"p99={record['p99_ms']:.4f} ms "
                                    f"CV={record['cv']:.2%}"
                                )
                        if workspace is not None:
                            workspace.close()
                            dist.barrier(group=group)
                    if args.max_cases and case_count > args.max_cases:
                        break
                if args.max_cases and case_count > args.max_cases:
                    break
            if args.max_cases and case_count > args.max_cases:
                break
        if args.max_cases and case_count > args.max_cases:
            break

    if rank == 0:
        payload = {
            "schema_version": 1,
            "kind": "dcp_performance",
            "profile": args.profile,
            "command": " ".join(sys.argv),
            "measurement": {
                "warmup": args.warmup,
                "samples": args.samples,
                "calls_per_sample": args.calls_per_sample,
                "cuda_graph": args.cuda_graph,
            },
            "environment": environment_metadata(REPO_ROOT),
            "results": records,
        }
        if args.output:
            write_json(args.output, payload)
            print(f"Wrote {args.output}")
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as output:
                fieldnames = sorted(
                    key
                    for key in records[0]
                    if not isinstance(records[0][key], (list, dict))
                )
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow({key: record.get(key) for key in fieldnames})
            print(f"Wrote {args.csv}")
    dist.barrier(group=group)
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "nightly"), default="smoke")
    parser.add_argument("--batches", type=_parse_ints)
    parser.add_argument("--q-lens", type=_parse_ints)
    parser.add_argument("--heads", type=_parse_ints)
    parser.add_argument("--kv-lens", type=_parse_ints)
    parser.add_argument("--q-profiles", type=_parse_strings)
    parser.add_argument("--modes", type=_parse_strings)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--calls-per-sample", type=int, default=10)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--local-output-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", type=str)
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples <= 0 or args.calls_per_sample <= 0:
        parser.error("warmup must be non-negative; samples/calls must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("max-cases must be positive")
    if args.q_profiles and not set(args.q_profiles) <= {"full", "ragged"}:
        parser.error("q-profiles must contain only full and ragged")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
