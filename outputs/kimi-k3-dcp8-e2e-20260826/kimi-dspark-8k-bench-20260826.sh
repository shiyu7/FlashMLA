#!/usr/bin/env bash
set -uo pipefail

mode=${1:?usage: kimi-dspark-8k-bench-20260826.sh dcp8-dspark|tp32dp4-dspark}
case "${mode}" in
  dcp8-dspark|tp32dp4-dspark) ;;
  *) echo "unknown mode: ${mode}" >&2; exit 2 ;;
esac

run_dir=/workspace/kimi-dcp/results/${mode}-8k
mkdir -p "${run_dir}"
cd "${run_dir}"

date -u +%Y-%m-%dT%H:%M:%SZ > started-at-utc.txt
sha256sum \
  /workspace/kimi-dcp/sglang_auto_bench.py \
  /workspace/kimi-dcp/kimi-dspark-8k-bench-20260826.py \
  > benchmark-scripts.sha256
python3 /workspace/kimi-dcp/kimi-dspark-8k-bench-20260826.py \
  > benchmark-stdout.log 2>&1
rc=$?
echo "${rc}" > benchmark.exit
date -u +%Y-%m-%dT%H:%M:%SZ > finished-at-utc.txt
exit "${rc}"
