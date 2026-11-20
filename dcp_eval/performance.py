"""Timing and provenance helpers shared by DCP performance tools."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class LatencyStats:
    samples: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    cv: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("latency sample list is empty")
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def summarize_latency(samples_ms: list[float]) -> LatencyStats:
    values = sorted(float(value) for value in samples_ms)
    if not values:
        raise ValueError("latency sample list is empty")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return LatencyStats(
        samples=len(values),
        p50_ms=_percentile(values, 0.50),
        p90_ms=_percentile(values, 0.90),
        p99_ms=_percentile(values, 0.99),
        mean_ms=mean,
        std_ms=std,
        cv=std / mean if mean else 0.0,
        min_ms=values[0],
        max_ms=values[-1],
    )


def distributed_cuda_samples(
    function: Callable[[], object],
    *,
    warmup: int,
    samples: int,
    calls_per_sample: int,
    group: dist.ProcessGroup,
) -> tuple[list[float], list[list[float]]]:
    """Measure chunk-amortized CUDA latency and return rank-critical samples.

    CUDA events are outside individual operator calls.  Every sample measures
    ``calls_per_sample`` calls and is divided by that count, avoiding one event
    pair per very short kernel.  The first result is the elementwise maximum
    across ranks; the second contains each rank's local samples.
    """
    if warmup < 0 or samples <= 0 or calls_per_sample <= 0:
        raise ValueError("invalid warmup/samples/calls_per_sample")
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for sample_idx in range(samples):
        starts[sample_idx].record()
        for _ in range(calls_per_sample):
            function()
        ends[sample_idx].record()
    ends[-1].synchronize()
    local_values = torch.tensor(
        [
            starts[idx].elapsed_time(ends[idx]) / calls_per_sample
            for idx in range(samples)
        ],
        dtype=torch.float64,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    per_rank = [torch.empty_like(local_values) for _ in range(dist.get_world_size(group))]
    dist.all_gather(per_rank, local_values, group=group)
    per_rank_cpu = [values.cpu().tolist() for values in per_rank]
    critical = torch.stack(per_rank, dim=0).amax(dim=0).cpu().tolist()
    return critical, per_rank_cpu


def _run_text(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(repo_root: str | Path) -> dict[str, object]:
    """Capture enough provenance to reject incomparable benchmark runs."""
    root = Path(repo_root).resolve()
    git_sha = _run_text(["git", "rev-parse", "HEAD"], root)
    status = _run_text(["git", "status", "--porcelain"], root) or ""
    diff = _run_text(["git", "diff", "--binary", "HEAD"], root) or ""
    dirty_hash = hashlib.sha256((status + diff).encode()).hexdigest() if status else None
    nvidia_smi = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,pstate,clocks.sm,clocks.mem,power.draw",
            "--format=csv,noheader,nounits",
        ],
        root,
    )
    return {
        "git_sha": git_sha,
        "git_dirty": bool(status),
        "git_dirty_hash": dirty_hash,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "nvidia_smi": nvidia_smi.splitlines() if nvidia_smi else None,
    }


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
