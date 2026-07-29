import argparse
import math
from dataclasses import dataclass
from typing import List

import torch

import flash_mla


PAGE_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512


@dataclass
class RankInputs:
    cache: torch.Tensor
    block_table: torch.Tensor
    seqlens: torch.Tensor
    causal_seqlens: torch.Tensor
    metadata: torch.Tensor
    num_splits: torch.Tensor


def _pack_rank(
    global_kv,
    global_seqlens,
    q_len,
    num_heads,
    cp_size,
    cp_rank,
):
    batch = global_kv.shape[0]
    local_seqlens = flash_mla.cyclic_local_length(
        global_seqlens,
        cp_size,
        cp_rank,
    ).to(torch.int32)
    max_pages = max(
        (int(local_seqlens.max().item()) + PAGE_SIZE - 1) // PAGE_SIZE,
        1,
    )
    block_table = torch.full(
        (batch, max_pages),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    pages = []
    for batch_idx in range(batch):
        seqlen = int(global_seqlens[batch_idx].item())
        tokens = global_kv[
            batch_idx, cp_rank:seqlen:cp_size
        ].contiguous()
        num_pages = (tokens.shape[0] + PAGE_SIZE - 1) // PAGE_SIZE
        for page_idx in range(num_pages):
            page = torch.zeros(
                PAGE_SIZE,
                1,
                HEAD_DIM,
                dtype=global_kv.dtype,
                device="cuda",
            )
            begin = page_idx * PAGE_SIZE
            end = min(begin + PAGE_SIZE, tokens.shape[0])
            page[: end - begin] = tokens[begin:end]
            block_table[batch_idx, page_idx] = len(pages)
            pages.append(page)
    if not pages:
        pages.append(
            torch.zeros(
                PAGE_SIZE,
                1,
                HEAD_DIM,
                dtype=global_kv.dtype,
                device="cuda",
            )
        )
    cache = torch.stack(pages)
    causal_seqlens = flash_mla.make_cyclic_local_causal_seqlens(
        global_seqlens,
        q_len,
        cp_size,
        cp_rank,
    )
    metadata, num_splits = flash_mla.get_mla_metadata_dense_fp8(
        local_seqlens,
        q_len * num_heads,
        1,
    )
    return RankInputs(
        cache,
        block_table,
        local_seqlens,
        causal_seqlens,
        metadata,
        num_splits,
    )


def _time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


@torch.inference_mode()
def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("dense FP8 DCP requires SM90")

    torch.manual_seed(args.seed)
    q = (
        torch.randn(
            args.batch,
            args.q_len,
            args.num_heads,
            HEAD_DIM,
            device="cuda",
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    global_kv = (
        torch.randn(
            args.batch,
            args.kv_len,
            1,
            HEAD_DIM,
            device="cuda",
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    global_seqlens = torch.full(
        (args.batch,),
        args.kv_len,
        dtype=torch.int32,
        device="cuda",
    )
    softmax_scale = 1 / math.sqrt(HEAD_DIM)

    print(
        f"GPU={torch.cuda.get_device_name()} B={args.batch} "
        f"Q={args.q_len} H={args.num_heads} KV={args.kv_len}"
    )
    print(
        "cp  rank_ms                         critical_ms  "
        "ideal_speedup  merge_ms  ideal_TFLOPS"
    )
    baseline_ms = None
    for cp_size in args.cp_sizes:
        ranks: List[RankInputs] = [
            _pack_rank(
                global_kv,
                global_seqlens,
                args.q_len,
                args.num_heads,
                cp_size,
                rank,
            )
            for rank in range(cp_size)
        ]

        def run_rank(rank: int):
            inputs = ranks[rank]
            return flash_mla.flash_mla_dense_fp8(
                q,
                inputs.cache,
                inputs.block_table,
                inputs.seqlens,
                HEAD_DIM_V,
                inputs.metadata,
                inputs.num_splits,
                softmax_scale=softmax_scale,
                causal=True,
                causal_seqlens_k=inputs.causal_seqlens,
            )

        rank_ms = [
            _time_cuda(
                lambda rank=rank: run_rank(rank),
                args.warmup,
                args.iterations,
            )
            for rank in range(cp_size)
        ]
        local_states = [run_rank(rank) for rank in range(cp_size)]
        outputs = [state[0] for state in local_states]
        lses = [state[1] for state in local_states]
        merge_ms = _time_cuda(
            lambda: flash_mla.merge_local_attention(outputs, lses),
            args.warmup,
            args.iterations,
        )

        critical_ms = max(rank_ms)
        if baseline_ms is None:
            baseline_ms = critical_ms
        ideal_speedup = baseline_ms / critical_ms
        global_flops = (
            args.batch
            * args.q_len
            * args.num_heads
            * 2
            * args.kv_len
            * (HEAD_DIM + HEAD_DIM_V)
        )
        ideal_tflops = global_flops / (critical_ms / 1000) / 1e12
        rank_text = ",".join(f"{value:.3f}" for value in rank_ms)
        print(
            f"{cp_size:<3} {rank_text:<31} {critical_ms:>11.3f}  "
            f"{ideal_speedup:>13.2f}  {merge_ms:>8.3f}  "
            f"{ideal_tflops:>12.1f}"
        )

    print(
        "\ncritical_ms models concurrent ranks with max(rank_ms). "
        "Q distribution and cross-rank communication are excluded; "
        "merge_ms is a local PyTorch merge measurement."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--cp-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.cp_sizes = sorted(
        {int(value) for value in args.cp_sizes.split(",")}
    )
    if args.q_len not in (1, 2, 4, 8):
        raise ValueError("initial benchmark scope supports q_len in 1,2,4,8")
    if any(cp_size <= 0 for cp_size in args.cp_sizes):
        raise ValueError("cp sizes must be positive")
    if 1 not in args.cp_sizes:
        raise ValueError("cp sizes must include 1 as the speedup baseline")
    if args.kv_len < args.q_len:
        raise ValueError("kv_len must be at least q_len")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
