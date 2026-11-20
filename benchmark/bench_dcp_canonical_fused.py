"""Benchmark the R14 fused canonical dense-FP8 prototype on one SM90 GPU.

This isolates both the local leaf + fixed-tree operator and the P2 leaf-export
ABI. Distributed communication is deliberately excluded from this benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import flash_mla


PAGE_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512


def _time_cuda(
    function, warmup: int, iterations: int, nvtx_name: str
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(nvtx_name)
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    torch.cuda.nvtx.range_pop()
    return begin.elapsed_time(end) / iterations


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("R14 canonical benchmark requires SM90")
    if args.chunk_size * args.num_chunks < args.kv_len:
        raise ValueError("chunk_size * num_chunks must cover kv_len")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    pages_per_batch = math.ceil(args.kv_len / PAGE_SIZE)
    block_table = torch.arange(
        args.batch * pages_per_batch, dtype=torch.int32, device=device
    ).view(args.batch, pages_per_batch)
    cache = (
        torch.randn(
            args.batch * pages_per_batch,
            PAGE_SIZE,
            1,
            HEAD_DIM,
            device=device,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    q = (
        torch.randn(
            args.batch,
            args.q_len,
            args.heads,
            HEAD_DIM,
            device=device,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    seqlens = torch.full(
        (args.batch,), args.kv_len, dtype=torch.int32, device=device
    )
    causal_seqlens = flash_mla.make_cyclic_local_causal_seqlens(
        seqlens, args.q_len, cp_size=1, cp_rank=0
    )
    descale_q = torch.tensor([0.75], dtype=torch.float32, device=device)
    descale_k = torch.tensor([1.25], dtype=torch.float32, device=device)
    common = dict(
        softmax_scale=1 / math.sqrt(HEAD_DIM),
        causal=True,
        causal_seqlens_k=causal_seqlens,
        descale_q=descale_q,
        descale_k=descale_k,
    )

    regular_metadata, regular_splits = flash_mla.get_mla_metadata_dense_fp8(
        seqlens, args.q_len * args.heads, 1
    )
    canonical_metadata, canonical_splits = (
        flash_mla.get_mla_metadata_canonical_dense_fp8(
            seqlens, args.chunk_size, args.num_chunks
        )
    )

    def regular_bf16():
        return flash_mla.flash_mla_dense_fp8(
            q,
            cache,
            block_table,
            seqlens,
            HEAD_DIM_V,
            regular_metadata,
            regular_splits,
            **common,
        )

    def regular_fp32():
        return flash_mla.flash_mla_dense_fp8_fp32(
            q,
            cache,
            block_table,
            seqlens,
            HEAD_DIM_V,
            regular_metadata,
            regular_splits,
            **common,
        )

    def canonical():
        return flash_mla.flash_mla_dense_fp8_canonical(
            q,
            cache,
            block_table,
            seqlens,
            HEAD_DIM_V,
            canonical_metadata,
            canonical_splits,
            args.num_chunks,
            **common,
        )

    def canonical_leaves():
        return flash_mla.flash_mla_dense_fp8_canonical_leaves(
            q,
            cache,
            block_table,
            seqlens,
            HEAD_DIM_V,
            canonical_metadata,
            canonical_splits,
            args.num_chunks,
            **common,
        )

    regular_out, regular_lse = regular_fp32()
    canonical_out, canonical_lse = canonical()
    repeated_out, repeated_lse = canonical()
    repeatable = torch.equal(canonical_out, repeated_out) and torch.equal(
        canonical_lse, repeated_lse
    )
    output_max_abs = (canonical_out - regular_out).abs().max().item()
    lse_max_abs = (canonical_lse - regular_lse).abs().max().item()

    regular_bf16_ms = _time_cuda(
        regular_bf16, args.warmup, args.iterations, "regular_bf16"
    )
    regular_fp32_ms = _time_cuda(
        regular_fp32, args.warmup, args.iterations, "regular_fp32"
    )
    canonical_ms = _time_cuda(
        canonical, args.warmup, args.iterations, "canonical_fused"
    )
    canonical_leaves_ms = _time_cuda(
        canonical_leaves, args.warmup, args.iterations, "canonical_leaves"
    )
    leaves_out, leaves_lse = canonical_leaves()
    payload = {
        "revision_id": "R14-P2",
        "gpu": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "q_len": args.q_len,
            "heads": args.heads,
            "kv_len": args.kv_len,
            "head_dim": HEAD_DIM,
            "head_dim_v": HEAD_DIM_V,
            "chunk_size": args.chunk_size,
            "num_chunks": args.num_chunks,
        },
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "regular_bf16_ms": regular_bf16_ms,
            "regular_fp32_ms": regular_fp32_ms,
            "canonical_fused_ms": canonical_ms,
            "canonical_leaves_ms": canonical_leaves_ms,
            "leaf_export_vs_fused_pct":
                (canonical_leaves_ms / canonical_ms - 1) * 100,
            "canonical_vs_bf16_pct":
                (canonical_ms / regular_bf16_ms - 1) * 100,
            "canonical_vs_fp32_pct":
                (canonical_ms / regular_fp32_ms - 1) * 100,
        },
        "accuracy": {
            "bitwise_repeatable": repeatable,
            "vs_regular_output_max_abs": output_max_abs,
            "vs_regular_lse_max_abs": lse_max_abs,
        },
        "leaf_export": {
            "output_shape": list(leaves_out.shape),
            "lse_shape": list(leaves_lse.shape),
            "bytes": leaves_out.numel() * leaves_out.element_size()
                + leaves_lse.numel() * leaves_lse.element_size(),
        },
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--num-chunks", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1414)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
