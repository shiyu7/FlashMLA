"""Run the dense FP8 cyclic-DCP semantic accuracy matrix."""

from __future__ import annotations

import argparse
import json
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
    make_fp8_inputs,
    run_cyclic_case,
)
from dcp_eval.performance import environment_metadata, write_json  # noqa: E402


def _parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",")})
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive ints")
    return result


def _profile(args: argparse.Namespace):
    if args.profile == "smoke":
        seeds = [2026]
        distributions = ["normal"]
        descales = [(0.75, 1.25)]
    else:
        seeds = list(range(2026, 2046))
        distributions = ["normal", "uniform", "near_zero", "signed"]
        descales = [(0.25, 4.0), (0.5, 2.0), (1.0, 1.0), (2.0, 0.5)]
    return seeds, distributions, descales


def _is_close(actual, reference, *, rtol: float, atol: float) -> bool:
    return bool(torch.isclose(actual, reference, rtol=rtol, atol=atol).all())


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("dense FP8 DCP requires SM90")

    seeds, distributions, descales = _profile(args)
    records: list[dict[str, object]] = []
    failures = 0
    case_index = 0
    for cp_size in args.cp_sizes:
        for q_len in args.q_lens:
            for distribution in distributions:
                for seed in seeds:
                    descale_q, descale_k = descales[case_index % len(descales)]
                    case_index += 1
                    q, global_kv, global_seqlens = make_fp8_inputs(
                        global_seqlens=[q_len, 64 + q_len, 129 + q_len],
                        q_len=q_len,
                        num_heads=args.heads,
                        distribution=distribution,
                        seed=seed,
                    )
                    q_lens = torch.tensor(
                        [q_len, max(1, q_len // 2), 0],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    output, lse, output_ref, lse_ref = run_cyclic_case(
                        q,
                        global_kv,
                        global_seqlens,
                        q_lens,
                        cp_size=cp_size,
                        softmax_scale=1 / math.sqrt(q.shape[-1]),
                        descale_q=descale_q,
                        descale_k=descale_k,
                        permute_pages=args.permute_pages,
                        page_seed=seed,
                        local_output_dtype=args.local_output_dtype,
                    )
                    metrics = compute_accuracy_metrics(
                        output, lse, output_ref, lse_ref, q_lens
                    )
                    output_close = _is_close(
                        output.float(), output_ref, rtol=0.08, atol=0.02
                    )
                    lse_close = _is_close(
                        lse, lse_ref, rtol=0.01, atol=0.02
                    )
                    suffix_neutral = True
                    for batch_idx, logical_q in enumerate(q_lens.tolist()):
                        suffix_neutral &= (
                            int(torch.count_nonzero(output[batch_idx, logical_q:]))
                            == 0
                        )
                        suffix_neutral &= bool(
                            torch.isneginf(lse[batch_idx, logical_q:]).all()
                        )
                    passed = (
                        output_close
                        and lse_close
                        and suffix_neutral
                        and metrics.output_nan_count == 0
                        and metrics.lse_nan_count == 0
                        and metrics.lse_inf_mask_mismatches == 0
                    )
                    failures += int(not passed)
                    record = {
                        "cp_size": cp_size,
                        "q_len": q_len,
                        "batch": 3,
                        "heads": args.heads,
                        "global_seqlens": global_seqlens.cpu().tolist(),
                        "q_lens": q_lens.cpu().tolist(),
                        "distribution": distribution,
                        "seed": seed,
                        "descale_q": descale_q,
                        "descale_k": descale_k,
                        "permuted_pages": args.permute_pages,
                        "local_output_dtype": args.local_output_dtype,
                        "passed": passed,
                        "output_close": output_close,
                        "lse_close": lse_close,
                        "suffix_neutral": suffix_neutral,
                        **asdict(metrics),
                    }
                    records.append(record)
                    state = "PASS" if passed else "FAIL"
                    print(
                        f"{state} CP={cp_size} Q={q_len} dist={distribution} "
                        f"local_out={args.local_output_dtype} "
                        f"seed={seed} out_p99={metrics.output_abs_p99:.6g} "
                        f"lse_max={metrics.lse_abs_max:.6g}"
                    )

    payload = {
        "schema_version": 1,
        "kind": "dcp_accuracy",
        "profile": args.profile,
        "command": " ".join(sys.argv),
        "environment": environment_metadata(REPO_ROOT),
        "summary": {
            "cases": len(records),
            "passed": len(records) - failures,
            "failed": failures,
        },
        "results": records,
    }
    if args.output:
        write_json(args.output, payload)
        print(f"Wrote {args.output}")
    elif failures:
        print(json.dumps(payload["summary"], sort_keys=True))
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "nightly"), default="smoke")
    parser.add_argument(
        "--cp-sizes", type=_parse_ints, default=_parse_ints("1,2,4,8")
    )
    parser.add_argument(
        "--q-lens", type=_parse_ints, default=_parse_ints("1,2,4,8")
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--permute-pages", action="store_true")
    parser.add_argument(
        "--local-output-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if any(q_len not in (1, 2, 4, 8) for q_len in args.q_lens):
        parser.error("q-lens must be drawn from 1,2,4,8")
    if args.heads <= 0:
        parser.error("heads must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(evaluate(parse_args()))
