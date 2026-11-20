"""Compare two DCP benchmark JSON files and enforce regression gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CASE_FIELDS = (
    "world_size",
    "batch",
    "q_len",
    "heads",
    "head_dim_v",
    "kv_len",
    "q_profile",
    "mode",
    "cuda_graph",
)


def _load(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "dcp_performance":
        raise ValueError(f"{path} is not a DCP performance result")
    return payload


def _key(record: dict[str, object]) -> tuple[object, ...]:
    return tuple(record.get(field) for field in CASE_FIELDS)


def _label(key: tuple[object, ...]) -> str:
    return " ".join(f"{field}={value}" for field, value in zip(CASE_FIELDS, key))


def _environment_signature(payload: dict[str, object]) -> tuple[object, ...]:
    environment = payload.get("environment", {})
    return (
        environment.get("gpu_name"),
        environment.get("compute_capability"),
        environment.get("torch_version"),
        environment.get("torch_cuda_version"),
        environment.get("world_size"),
    )


def compare(args: argparse.Namespace) -> int:
    baseline_payload = _load(args.baseline)
    candidate_payload = _load(args.candidate)
    if (
        not args.allow_environment_mismatch
        and _environment_signature(baseline_payload)
        != _environment_signature(candidate_payload)
    ):
        print("FAIL benchmark environments differ")
        print(" baseline:", _environment_signature(baseline_payload))
        print(" candidate:", _environment_signature(candidate_payload))
        return 1

    baseline = {_key(record): record for record in baseline_payload["results"]}
    candidate = {_key(record): record for record in candidate_payload["results"]}
    missing = sorted(set(baseline) - set(candidate), key=str)
    extra = sorted(set(candidate) - set(baseline), key=str)
    failures: list[str] = []
    if missing:
        failures.append(f"candidate is missing {len(missing)} baseline cases")
    if extra and args.require_exact_cases:
        failures.append(f"candidate has {len(extra)} unexpected cases")

    ratios: list[float] = []
    for key in sorted(set(baseline) & set(candidate), key=str):
        before = baseline[key]
        after = candidate[key]
        before_p50 = float(before["p50_ms"])
        after_p50 = float(after["p50_ms"])
        before_p99 = float(before["p99_ms"])
        after_p99 = float(after["p99_ms"])
        candidate_cv = float(after["cv"])
        ratios.append(after_p50 / before_p50)

        if candidate_cv > args.max_cv:
            failures.append(
                f"noisy CV {candidate_cv:.2%} > {args.max_cv:.2%}: {_label(key)}"
            )
        p50_budget_ms = max(
            before_p50 * args.max_p50_regression, args.absolute_budget_us / 1000
        )
        if after_p50 - before_p50 > p50_budget_ms:
            failures.append(
                f"p50 {before_p50:.6f}->{after_p50:.6f} ms: {_label(key)}"
            )
        p99_budget_ms = max(
            before_p99 * args.max_p99_regression, args.absolute_budget_us / 1000
        )
        if after_p99 - before_p99 > p99_budget_ms:
            failures.append(
                f"p99 {before_p99:.6f}->{after_p99:.6f} ms: {_label(key)}"
            )

    if not ratios:
        failures.append("baseline and candidate have no common cases")
        geomean_ratio = math.inf
    else:
        geomean_ratio = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        if geomean_ratio > 1 + args.max_geomean_regression:
            failures.append(
                f"p50 geomean regression {(geomean_ratio - 1):.2%} > "
                f"{args.max_geomean_regression:.2%}"
            )

    print(
        f"Compared {len(ratios)} cases; p50 geomean ratio={geomean_ratio:.5f}; "
        f"failures={len(failures)}"
    )
    for failure in failures:
        print("FAIL", failure)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--max-cv", type=float, default=0.02)
    parser.add_argument("--max-p50-regression", type=float, default=0.03)
    parser.add_argument("--max-p99-regression", type=float, default=0.05)
    parser.add_argument("--max-geomean-regression", type=float, default=0.02)
    parser.add_argument("--absolute-budget-us", type=float, default=1.0)
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--require-exact-cases", action="store_true")
    args = parser.parse_args()
    for name in (
        "max_cv",
        "max_p50_regression",
        "max_p99_regression",
        "max_geomean_regression",
        "absolute_budget_us",
    ):
        if getattr(args, name) < 0:
            parser.error(f"{name.replace('_', '-')} must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(compare(parse_args()))
