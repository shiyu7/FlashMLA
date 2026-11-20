#!/usr/bin/env python3
"""Run only the 8K Kimi-K3 case using the supplied auto-benchmark engine."""

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
    benchmark = load_benchmark_module()
    result = benchmark.find_optimal_throughput(
        input_len=8000,
        output_len=1500,
        slo_ttft=5000,
        slo_tpot=20,
        # Preserve the supplied script's 8K search granularity exactly.
        dp=8,
        dataset_name="random",
    )
    Path("result-8k.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
