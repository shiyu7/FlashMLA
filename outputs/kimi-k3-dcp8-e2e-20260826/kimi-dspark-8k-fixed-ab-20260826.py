#!/usr/bin/env python3
"""Run one comparable 8K/1.5K point with the supplied benchmark engine."""

import argparse
import importlib.util
import json
from pathlib import Path


BENCHMARK_SOURCE = Path("/workspace/kimi-dcp/sglang_auto_bench.py")


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("sglang_auto_bench", BENCHMARK_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark source: {BENCHMARK_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    benchmark = load_benchmark_module()
    ttft, tpot, e2e, total_thr, out_thr, req_thr, raw = benchmark.run_benchmark(
        input_len=8000,
        output_len=1500,
        req_rate=1.0,
        max_conn=8,
        num_prompts_multiplier=2,
        decode_only=False,
        prefill_only=False,
        warmup_prompts=0,
        dataset_name="random",
    )
    result = {
        "label": args.label,
        "input_len": 8000,
        "output_len": 1500,
        "request_rate": 1.0,
        "max_concurrency": 8,
        "num_prompts": 16,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": e2e,
        "total_token_throughput": total_thr,
        "output_token_throughput": out_thr,
        "request_throughput": req_thr,
    }
    Path("result-8k-fixed.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path("bench-serving-raw.log").write_text(raw or "", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
