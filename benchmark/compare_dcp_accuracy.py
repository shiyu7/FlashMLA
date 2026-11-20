"""Compare DCP accuracy JSON with a frozen golden baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASE_FIELDS = (
    "cp_size",
    "q_len",
    "batch",
    "heads",
    "distribution",
    "seed",
    "descale_q",
    "descale_k",
    "permuted_pages",
)
ERROR_FIELDS = (
    "output_abs_p99",
    "output_abs_max",
    "output_nrmse",
    "lse_abs_p99",
    "lse_abs_max",
)


def _load(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "dcp_accuracy":
        raise ValueError(f"{path} is not a DCP accuracy result")
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
    )


def compare(args: argparse.Namespace) -> int:
    baseline_payload = _load(args.baseline)
    candidate_payload = _load(args.candidate)
    if (
        not args.allow_environment_mismatch
        and _environment_signature(baseline_payload)
        != _environment_signature(candidate_payload)
    ):
        print("FAIL accuracy environments differ")
        return 1

    baseline = {_key(record): record for record in baseline_payload["results"]}
    candidate = {_key(record): record for record in candidate_payload["results"]}
    failures: list[str] = []
    missing = sorted(set(baseline) - set(candidate), key=str)
    extra = sorted(set(candidate) - set(baseline), key=str)
    if missing:
        failures.append(f"candidate is missing {len(missing)} baseline cases")
    if extra and args.require_exact_cases:
        failures.append(f"candidate has {len(extra)} unexpected cases")

    common = sorted(set(baseline) & set(candidate), key=str)
    for key in common:
        before = baseline[key]
        after = candidate[key]
        if not after.get("passed", False):
            failures.append(f"hard accuracy gate failed: {_label(key)}")
        for field in ERROR_FIELDS:
            before_value = float(before[field])
            after_value = float(after[field])
            budget = before_value * args.max_error_regression + args.absolute_floor
            if after_value - before_value > budget:
                failures.append(
                    f"{field} {before_value:.6g}->{after_value:.6g}: {_label(key)}"
                )
        before_cosine = float(before["output_cosine_p01"])
        after_cosine = float(after["output_cosine_p01"])
        if before_cosine - after_cosine > args.max_cosine_drop:
            failures.append(
                f"output_cosine_p01 {before_cosine:.8f}->{after_cosine:.8f}: "
                f"{_label(key)}"
            )
        for field in (
            "lse_inf_mask_mismatches",
            "output_nan_count",
            "lse_nan_count",
        ):
            if int(after[field]) != 0:
                failures.append(f"{field}={after[field]}: {_label(key)}")

    if not common:
        failures.append("baseline and candidate have no common cases")
    print(f"Compared {len(common)} cases; failures={len(failures)}")
    for failure in failures:
        print("FAIL", failure)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--max-error-regression", type=float, default=0.05)
    parser.add_argument("--absolute-floor", type=float, default=1e-7)
    parser.add_argument("--max-cosine-drop", type=float, default=1e-4)
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--require-exact-cases", action="store_true")
    args = parser.parse_args()
    if (
        args.max_error_regression < 0
        or args.absolute_floor < 0
        or args.max_cosine_drop < 0
    ):
        parser.error("all gate values must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(compare(parse_args()))
