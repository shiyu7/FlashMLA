#!/usr/bin/env python3
"""Aggregate DSPARK acceptance across all DP ranks for each benchmark window.

SGLang's serving benchmark currently prints ``Accept length`` from the first
entry of ``server_info.internal_states``.  That entry is one DP scheduler and
its order is not stable.  This analyzer instead matches the R13 evaluation
protocol: select every ``Decode batch`` sample in the benchmark's UTC window
and weight its acceptance fields by ``#running-req`` across all four DP logs.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "service-logs"
OUTPUT = ROOT / "acceptance_summary.csv"
RANK_OUTPUT = ROOT / "acceptance_by_rank.csv"

DECODE_RE = re.compile(
    r"^\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .*?\] "
    r"Decode batch, #running-req: (?P<running>\d+),.*?"
    r"accept len: (?P<length>[0-9.]+), "
    r"accept rate: (?P<rate>[0-9.]+),"
)


def parse_utc(path: Path) -> datetime:
    value = path.read_text(encoding="utf-8").strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def log_paths(result_name: str) -> list[Path]:
    if result_name.startswith("dcp8-canonical-"):
        pattern = "dcp8-canonical-r14p10-smoke4-rank*.log"
    elif result_name.startswith("tp32dp4-"):
        pattern = "tp32dp4-dspark-r14p10-control-run38-rank*.log"
    else:
        raise ValueError(f"unknown result condition: {result_name}")
    paths = sorted(LOGS.glob(pattern))
    if len(paths) != 4:
        raise RuntimeError(f"expected four service logs for {result_name}, got {paths}")
    return paths


def summarize_samples(
    revision_id: str,
    rank: str,
    samples: list[tuple[int, float, float]],
) -> dict[str, object]:
    active_requests = sum(running for running, _, _ in samples)
    weighted_length = sum(
        running * length for running, length, _ in samples
    ) / active_requests
    weighted_rate = sum(running * rate for running, _, rate in samples) / active_requests
    return {
        "revision_id": revision_id,
        "rank": rank,
        "log_samples": len(samples),
        "active_request_weight": active_requests,
        "accept_len_active_req_weighted": weighted_length,
        "accept_rate_active_req_weighted_percent": 100.0 * weighted_rate,
    }


def analyze(result_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    start = parse_utc(result_dir / "started-at-utc.txt")
    finish = parse_utc(result_dir / "finished-at-utc.txt")
    samples: list[tuple[int, float, float]] = []
    samples_per_rank: list[int] = []
    rank_rows: list[dict[str, object]] = []
    revision_id = result_dir.name.removesuffix("-8k-fixed")

    for rank, path in enumerate(log_paths(result_dir.name)):
        current_samples: list[tuple[int, float, float]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = DECODE_RE.match(line)
            if match is None:
                continue
            timestamp = datetime.strptime(
                match.group("time"), "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            if not start <= timestamp <= finish:
                continue
            running = int(match.group("running"))
            current_samples.append(
                (running, float(match.group("length")), float(match.group("rate")))
            )
        samples.extend(current_samples)
        samples_per_rank.append(len(current_samples))
        rank_rows.append(summarize_samples(revision_id, str(rank), current_samples))

    if not samples:
        raise RuntimeError(f"no Decode batch samples found for {result_dir.name}")
    active_requests = sum(running for running, _, _ in samples)
    weighted_length = sum(
        running * length for running, length, _ in samples
    ) / active_requests
    weighted_rate = sum(running * rate for running, _, rate in samples) / active_requests
    lengths = [length for _, length, _ in samples]
    return ({
        "revision_id": revision_id,
        "started_at_utc": start.isoformat(),
        "finished_at_utc": finish.isoformat(),
        "log_samples": len(samples),
        "samples_per_rank": "/".join(map(str, samples_per_rank)),
        "active_request_weight": active_requests,
        "accept_len_mean": mean(lengths),
        "accept_len_median": median(lengths),
        "accept_len_active_req_weighted": weighted_length,
        "accept_rate_active_req_weighted_percent": 100.0 * weighted_rate,
        "accept_rate_derived_from_len_percent": 100.0 * (weighted_length - 1.0) / 7.0,
    }, rank_rows)


def main() -> None:
    analyzed = [analyze(path) for path in sorted(RESULTS.iterdir()) if path.is_dir()]
    rows = [row for row, _ in analyzed]
    rank_rows = [rank_row for _, per_rank in analyzed for rank_row in per_rank]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with RANK_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rank_rows[0]))
        writer.writeheader()
        writer.writerows(rank_rows)
    for row in rows:
        print(
            f"{row['revision_id']}: weighted len="
            f"{row['accept_len_active_req_weighted']:.4f}, weighted rate="
            f"{row['accept_rate_active_req_weighted_percent']:.2f}%, samples="
            f"{row['log_samples']} ({row['samples_per_rank']})"
        )


if __name__ == "__main__":
    main()
