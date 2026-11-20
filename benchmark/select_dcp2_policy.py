"""Build a conservative exact-shape DCP2 peer/NCCL policy from repeated runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SHAPE_FIELDS = (
    "world_size",
    "batch",
    "q_len",
    "heads",
    "head_dim_v",
    "kv_len",
    "q_profile",
    "cuda_graph",
)


def _load(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "dcp_performance":
        raise ValueError(f"{path} is not a DCP performance result")
    if payload.get("environment", {}).get("world_size") != 2:
        raise ValueError(f"{path} is not a DCP2 result")
    return payload


def _shape(record: dict[str, object]) -> tuple[object, ...]:
    return tuple(record.get(field) for field in SHAPE_FIELDS)


def select(args: argparse.Namespace) -> int:
    if len(args.runs) < args.min_runs:
        raise ValueError(
            f"need at least {args.min_runs} repeated runs; got {len(args.runs)}"
        )
    peer_mode = f"peer_{args.stage}"
    nccl_mode = f"nccl_{args.stage}"
    runs = [_load(path) for path in args.runs]
    signatures = {
        (
            run["environment"].get("gpu_name"),
            str(run["environment"].get("compute_capability")),
            run["environment"].get("torch_version"),
            run["environment"].get("torch_cuda_version"),
        )
        for run in runs
    }
    if len(signatures) != 1 and not args.allow_environment_mismatch:
        raise ValueError("repeated runs use different benchmark environments")

    per_run: list[dict[tuple[object, ...], dict[str, dict[str, object]]]] = []
    for payload in runs:
        indexed: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
        for record in payload["results"]:
            if record["mode"] in (peer_mode, nccl_mode):
                indexed.setdefault(_shape(record), {})[record["mode"]] = record
        per_run.append(indexed)

    common_shapes = set(per_run[0])
    for indexed in per_run[1:]:
        common_shapes &= set(indexed)
    decisions: list[dict[str, object]] = []
    incomplete = 0
    for shape in sorted(common_shapes, key=str):
        observations = []
        for indexed in per_run:
            modes = indexed[shape]
            if peer_mode not in modes or nccl_mode not in modes:
                incomplete += 1
                observations = []
                break
            peer = modes[peer_mode]
            nccl = modes[nccl_mode]
            peer_p50 = float(peer["p50_ms"])
            nccl_p50 = float(nccl["p50_ms"])
            peer_p99 = float(peer["p99_ms"])
            nccl_p99 = float(nccl["p99_ms"])
            wins = (
                peer_p50 <= nccl_p50 * (1 - args.min_p50_improvement)
                and peer_p99 <= nccl_p99 * (1 + args.max_p99_regression)
            )
            observations.append(
                {
                    "peer_p50_ms": peer_p50,
                    "nccl_p50_ms": nccl_p50,
                    "peer_p99_ms": peer_p99,
                    "nccl_p99_ms": nccl_p99,
                    "p50_speedup": nccl_p50 / peer_p50,
                    "wins": wins,
                }
            )
        if not observations:
            continue
        win_count = sum(int(item["wins"]) for item in observations)
        decision = {
            **dict(zip(SHAPE_FIELDS, shape)),
            "stage": args.stage,
            "runs": len(observations),
            "peer_wins": win_count,
            "median_p50_speedup": statistics.median(
                item["p50_speedup"] for item in observations
            ),
            "selected": win_count >= args.min_wins,
            "observations": observations,
        }
        decisions.append(decision)
        state = "peer" if decision["selected"] else "nccl"
        shape_text = " ".join(
            f"{field}={value}" for field, value in zip(SHAPE_FIELDS, shape)
        )
        print(
            f"{state} wins={win_count}/{len(observations)} "
            f"median_speedup={decision['median_p50_speedup']:.3f} {shape_text}"
        )

    if not decisions:
        raise ValueError("runs have no common shapes containing peer and NCCL modes")
    payload = {
        "schema_version": 1,
        "kind": "dcp2_policy",
        "stage": args.stage,
        "criteria": {
            "min_runs": args.min_runs,
            "min_wins": args.min_wins,
            "min_p50_improvement": args.min_p50_improvement,
            "max_p99_regression": args.max_p99_regression,
        },
        "incomplete_observations": incomplete,
        "decisions": decisions,
    }
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--stage", choices=("merge", "operator"), default="operator")
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument("--min-wins", type=int, default=4)
    parser.add_argument("--min-p50-improvement", type=float, default=0.10)
    parser.add_argument("--max-p99-regression", type=float, default=0.0)
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.min_runs <= 0 or args.min_wins <= 0:
        parser.error("min-runs and min-wins must be positive")
    if args.min_wins > len(args.runs):
        parser.error("min-wins cannot exceed the supplied run count")
    if args.min_p50_improvement < 0 or args.max_p99_regression < 0:
        parser.error("improvement/regression limits must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(select(parse_args()))
