"""Evaluate the R14 fixed-chunk/fixed-tree DCP prototype.

The timing is an intentionally unfused PyTorch upper bound.  It validates the
numerical contract and must not be compared as if it were a production kernel.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcp_eval.accuracy import (  # noqa: E402
    compute_accuracy_metrics,
    global_attention_reference,
    make_fp8_inputs,
)
from dcp_eval.canonical import canonical_chunk_attention  # noqa: E402
from dcp_eval.performance import environment_metadata, write_json  # noqa: E402


def _parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",")})
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive ints")
    return result


def _time_cuda(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    q, global_kv, global_seqlens = make_fp8_inputs(
        global_seqlens=[args.kv_len] * args.batch,
        q_len=args.q_len,
        num_heads=args.heads,
        seed=args.seed,
    )
    q_lens = torch.full(
        (args.batch,), args.q_len, dtype=torch.int32, device="cuda"
    )
    common = {
        "chunk_size": args.chunk_size,
        "softmax_scale": 1 / math.sqrt(q.shape[-1]),
        "descale_q": args.descale_q,
        "descale_k": args.descale_k,
        "head_dim_v": args.head_dim_v,
    }
    reference_output, reference_lse = global_attention_reference(
        q,
        global_kv,
        global_seqlens,
        q_lens,
        softmax_scale=common["softmax_scale"],
        descale_q=args.descale_q,
        descale_k=args.descale_k,
        head_dim_v=args.head_dim_v,
    )
    baseline_output = None
    baseline_lse = None
    records: list[dict[str, object]] = []
    for cp_size in args.cp_sizes:
        def run():
            return canonical_chunk_attention(
                q,
                global_kv,
                global_seqlens,
                q_lens,
                cp_size=cp_size,
                **common,
            )

        latency_ms = _time_cuda(run, args.warmup, args.iterations)
        output, lse = run()
        if baseline_output is None:
            baseline_output = output
            baseline_lse = lse
        invariant = torch.equal(output, baseline_output) and torch.equal(
            lse, baseline_lse
        )
        metrics = compute_accuracy_metrics(
            output, lse, reference_output, reference_lse, q_lens
        )
        record = {
            "cp_size": cp_size,
            "batch": args.batch,
            "q_len": args.q_len,
            "heads": args.heads,
            "head_dim_v": args.head_dim_v,
            "kv_len": args.kv_len,
            "chunk_size": args.chunk_size,
            "num_chunks": math.ceil(args.kv_len / args.chunk_size),
            "cp_invariant": invariant,
            "prototype_latency_ms": latency_ms,
            **asdict(metrics),
        }
        records.append(record)
        print(
            f"CP={cp_size} invariant={invariant} "
            f"latency={latency_ms:.4f} ms "
            f"out_p99={metrics.output_abs_p99:.6g} "
            f"nrmse={metrics.output_nrmse:.6g}"
        )

    payload = {
        "schema_version": 1,
        "kind": "dcp_canonical_prototype",
        "command": " ".join(sys.argv),
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "warning": "unfused PyTorch prototype; not a production kernel timing",
        },
        "environment": environment_metadata(REPO_ROOT),
        "results": records,
    }
    if args.output:
        write_json(args.output, payload)
        print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--head-dim-v", type=int, default=512)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--cp-sizes", type=_parse_ints, default=_parse_ints("1,2,4,8"))
    parser.add_argument("--descale-q", type=float, default=0.75)
    parser.add_argument("--descale-k", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if min(args.batch, args.q_len, args.heads, args.head_dim_v, args.kv_len) <= 0:
        parser.error("shape arguments must be positive")
    if args.chunk_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("chunk/measurement arguments are invalid")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
