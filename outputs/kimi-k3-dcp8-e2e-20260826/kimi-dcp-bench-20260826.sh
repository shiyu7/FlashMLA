#!/usr/bin/env bash
set -uo pipefail

mode=${1:?usage: kimi-dcp-bench-20260826.sh dcp8|tp32dp4}
case "${mode}" in
  dcp8|tp32dp4) ;;
  *) echo "unknown mode: ${mode}" >&2; exit 2 ;;
esac

run_dir=/workspace/kimi-dcp/results/${mode}
mkdir -p "${run_dir}"
cd "${run_dir}"

date -u +%Y-%m-%dT%H:%M:%SZ > started-at-utc.txt
sha256sum /workspace/kimi-dcp/sglang_auto_bench.py > benchmark-script.sha256
python3 /workspace/kimi-dcp/sglang_auto_bench.py --dataset-name random \
  > benchmark-stdout.log 2>&1
rc=$?
echo "${rc}" > benchmark.exit
date -u +%Y-%m-%dT%H:%M:%SZ > finished-at-utc.txt
exit "${rc}"
