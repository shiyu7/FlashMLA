#!/usr/bin/env bash
set -uo pipefail

label=${1:?usage: kimi-dspark-8k-fixed-ab-20260826.sh LABEL}
run_dir=/workspace/kimi-dcp/results/${label}-8k-fixed
mkdir -p "${run_dir}"
cd "${run_dir}"

date -u +%Y-%m-%dT%H:%M:%SZ > started-at-utc.txt
sha256sum \
  /workspace/kimi-dcp/sglang_auto_bench.py \
  /workspace/kimi-dcp/kimi-dspark-8k-fixed-ab-20260826.py \
  > benchmark-scripts.sha256
python3 /workspace/kimi-dcp/kimi-dspark-8k-fixed-ab-20260826.py \
  --label "${label}" > benchmark-stdout.log 2>&1
rc=$?
echo "${rc}" > benchmark.exit
date -u +%Y-%m-%dT%H:%M:%SZ > finished-at-utc.txt
exit "${rc}"
