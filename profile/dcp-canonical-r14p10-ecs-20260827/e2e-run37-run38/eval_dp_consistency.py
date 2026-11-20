#!/usr/bin/env python3
"""Send an identical deterministic 8K request to every DP replica."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, payload: object, timeout: float) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def flush_cache(
    base_url: str, timeout: float, retries: int, retry_interval: float
) -> str:
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            f"{base_url}/flush_cache", data=b"", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code != 400 or attempt == retries:
                raise RuntimeError(
                    f"flush_cache failed after attempt {attempt}: "
                    f"HTTP {error.code}: {body}"
                ) from error
            print(
                f"flush_cache attempt {attempt} found pending work; "
                f"retrying in {retry_interval:.1f}s",
                flush=True,
            )
            time.sleep(retry_interval)
    raise AssertionError("unreachable")


def output_hash(output_ids: list[int]) -> str:
    payload = ",".join(map(str, output_ids)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--input-len", type=int, default=8000)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--token-id", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--flush-retries", type=int, default=12)
    parser.add_argument("--flush-retry-interval", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt = [args.token_id] * args.input_len
    rows: list[dict[str, object]] = []
    for cycle in range(1, args.cycles + 1):
        flush_response = flush_cache(
            args.base_url,
            args.timeout,
            args.flush_retries,
            args.flush_retry_interval,
        )
        for request_index in range(args.dp_size):
            started = time.perf_counter()
            response = post_json(
                f"{args.base_url}/generate",
                {
                    "input_ids": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.output_len,
                        "ignore_eos": True,
                    },
                },
                args.timeout,
            )
            elapsed = time.perf_counter() - started
            meta = response["meta_info"]
            output_ids = response["output_ids"]
            row = {
                "cycle": cycle,
                "request_index": request_index,
                "flush_response": flush_response if request_index == 0 else None,
                "dp_rank": meta.get("dp_rank"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "spec_accept_length": meta.get("spec_accept_length"),
                "spec_accept_rate": meta.get("spec_accept_rate"),
                "spec_verify_ct": meta.get("spec_verify_ct"),
                "elapsed_seconds": elapsed,
                "output_ids_sha256": output_hash(output_ids),
                "output_ids": output_ids,
            }
            rows.append(row)
            accept_len = row["spec_accept_length"]
            accept_rate = row["spec_accept_rate"]
            accept_len_text = (
                f"{accept_len:.4f}" if accept_len is not None else "n/a"
            )
            accept_rate_text = (
                f"{100.0 * accept_rate:.2f}%"
                if accept_rate is not None
                else "n/a"
            )
            print(
                f"cycle={cycle} request={request_index} dp={row['dp_rank']} "
                f"accept_len={accept_len_text} "
                f"accept_rate={accept_rate_text} "
                f"sha256={row['output_ids_sha256'][:12]} elapsed={elapsed:.2f}s",
                flush=True,
            )

    payload = {
        "config": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "cycles": args.cycles,
            "dp_size": args.dp_size,
            "token_id": args.token_id,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
