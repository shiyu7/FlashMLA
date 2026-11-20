# DCP accuracy and performance evaluation

## Purpose

This document defines the evaluation contract for tuning dense FP8 DCP on
SM90.  It covers the rank-local FlashMLA kernel, the DCP2 NVLink peer merge,
the DCP2/4/8 NCCL fallback, and the composed operator.  A benchmark result is
actionable only when it identifies the layer being measured and records enough
environment metadata to reproduce it.

The evaluation has four goals:

1. Detect semantic errors in cyclic sharding, causal MTP boundaries, ragged
   query validity, page tables, and empty rank-local shards.
2. Separate rank-local kernel error from cross-rank merge error.
3. Separate local-kernel, merge-only, operator, and runtime latency so a
   regression can be attributed to one component.
4. Produce machine-readable accuracy and performance records suitable for a
   baseline/candidate gate on H20/H100/H800 systems.

Benchmarks are not correctness tests.  A candidate must pass the accuracy
suite before its performance result is considered.

## Evaluation layers

| Layer | Accuracy reference | Timed region |
| --- | --- | --- |
| Local kernel | FP32 attention over one rank's visible KV | Dense FP8 kernel with metadata and output storage prepared |
| Merge | FP64 merge of rank-local output and natural-log LSE | DCP2 prepare+peer merge or NCCL LSE correction+reduce-scatter |
| Operator | Unsharded FP32 global attention | Local kernel plus merge, with inputs resident on each GPU |
| Runtime | Unsharded FP32 global attention | Operator plus deployment-owned Q distribution and metadata updates |

The repository implements the first three layers.  Runtime measurement remains
in the serving system because Q distribution and scheduling are deployment
contracts rather than FlashMLA APIs.

## Accuracy oracles

### Global attention oracle

`dcp_eval.global_attention_reference` dequantizes FP8 Q and KV, applies the
scalar descales, evaluates causal attention in FP32, and returns native
`BQHD/BQH` output and natural-log LSE.  It uses the logical `q_lens[b]` trailing
query window, not the physical padded `Qmax`, and emits exact zero/negative
infinity for invalid suffixes.

This oracle checks the whole semantic chain, including cyclic ownership and
local causal boundaries.

### Merge oracle

`dcp_eval.fp64_merge_reference` merges independently produced local states:

```text
m   = max_r(lse_r)
w_r = exp(lse_r - m)
out = sum_r(w_r * out_r) / sum_r(w_r)
lse = m + log(sum_r(exp(lse_r - m)))
```

The implementation expresses the normalized form through global LSE and uses
FP64 accumulation.  Fully empty rows produce exact zero output and `-inf` LSE.
Peer-vs-NCCL comparison alone is insufficient because both implementations
could share a semantic error; both are also compared with this oracle.

## Accuracy matrix

The suite uses a small pairwise matrix for changes and a broader matrix for
nightly validation.  It deliberately avoids a full Cartesian product.

| Dimension | Change/smoke | Nightly |
| --- | --- | --- |
| DCP size | 1, 2 | 1, 2, 4, 8 |
| Physical Q | 1, 2, 4, 8 | 1, 2, 4, 8 |
| Batch | 1, 3 | 1, 2, 4, 8, 16, 32 |
| Query heads | 16, 128 | 16, 32, 64, 128 |
| KV heads | 1 | 1 and supported production GQA ratios |
| Global KV length | Q, 63, 64, 65, 129 | plus 127, 128, 4K, 8K, 32K |
| Logical query profile | full; `[Q, max(1,Q/2), 0]` | random and 25/50/75% occupancy |
| FP8 values | normal | normal, uniform, near-zero, signed high-dynamic-range |
| Descales | 0.75/1.25 | values from 0.25, 0.5, 1, 2, 4 |
| Page table | contiguous | permuted physical page IDs and partial final pages |
| Seeds | 1 | at least 20 |

Required invariants are checked independently of floating-point tolerances:

- Invalid query suffixes are exactly zero output and negative-infinity LSE.
- A fully empty local or global row has the same exact neutral state.
- No valid output or LSE contains NaN.
- Repacking identical logical KV into permuted physical pages does not change
  the result.
- CP1/2/4/8 reshards of one global input agree with the same global oracle.
- Allocating and preallocated output APIs agree exactly.
- Eager and CUDA Graph replay agree within the merge tolerance.
- DCP2 peer, NCCL, and the FP64 merge oracle agree.

The long-running graph test must replay at least 10,000 times while changing
input state and `q_lens`.  It is a nightly stress test, not a per-change test.

## Accuracy metrics and gates

Each case records output absolute p50/p99/max error, p99 relative error, NRMSE,
row-cosine p01/min, LSE absolute p50/p99/max error, Inf-mask mismatches, and NaN
counts.  Relative error is diagnostic only near zero; absolute error, NRMSE,
and cosine prevent a misleading relative-error gate.

The initial hard compatibility gates match the existing DCP tests:

| Comparison | Output | LSE |
| --- | --- | --- |
| End-to-end vs global FP32 | `rtol=0.08, atol=0.02` | `rtol=0.01, atol=0.02` |
| DCP2 peer vs NCCL/FP64 merge | `rtol=0.02, atol=0.02` | `rtol=2e-5, atol=2e-5` |

After collecting at least 20 seeds on the target build, freeze a golden JSON
baseline.  In addition to the hard caps, p99, max, and NRMSE must not worsen by
more than 5% from that baseline.  A failed case must retain its full shape,
seed, rank, distribution, descales, and worst row coordinate.

Compare a new accuracy run with the frozen baseline using:

```bash
python benchmark/compare_dcp_accuracy.py \
  --baseline /tmp/dcp-accuracy-baseline.json \
  --candidate /tmp/dcp-accuracy-candidate.json
```

Run the single-GPU semantic sweep with:

```bash
python benchmark/eval_dcp_accuracy.py --profile smoke \
  --output /tmp/dcp-accuracy.json

python benchmark/eval_dcp_accuracy.py --profile nightly \
  --output /tmp/dcp-accuracy-nightly.json
```

Run real distributed merge correctness with one command per world size:

```bash
torchrun --standalone --nproc-per-node=2 -m pytest -q \
  tests/test_dcp_nccl_merge.py
torchrun --standalone --nproc-per-node=4 -m pytest -q \
  tests/test_dcp_nccl_merge.py
torchrun --standalone --nproc-per-node=8 -m pytest -q \
  tests/test_dcp_nccl_merge.py
```

The DCP2-specific peer and graph tests remain:

```bash
torchrun --standalone --nproc-per-node=2 -m pytest -q \
  tests/test_dcp2_peer_merge.py
```

Nightly endurance expands the same graph tests without changing their default
per-change runtime.  Inputs and ragged `q_lens` change on every replay; NCCL is
used as a periodic reference instead of adding communication to every replay:

```bash
FLASH_MLA_DCP2_RAGGED_GRAPH_REPLAYS=10000 \
FLASH_MLA_DCP2_EPOCH_GRAPH_REPLAYS=10000 \
FLASH_MLA_DCP2_GRAPH_CHECK_INTERVAL=1000 \
torchrun --standalone --nproc-per-node=2 -m pytest -q \
  tests/test_dcp2_peer_merge.py \
  -k 'ragged_q_lens_survives_graph_replay or epoch_protocol'
```

## Performance matrix

Performance is measured on real ranks inside one 8-GPU Pod.  The single-GPU
sequential rank simulator remains useful for kernel shape exploration but is
not an end-to-end DCP latency result.

The canonical tuning matrix is:

- `B={1,8,32}`
- `Q={1,4,8}`
- `H={64,128}`
- `KV={128,4096,32768}`
- `CP={1,2,4,8}`
- logical Q occupancy `{full, ragged}`
- full-attention `Dv=512`; synthetic merge crossover may additionally use
  `Dv={128,256,512}`

The change/smoke profile is a pairwise subset.  The nightly performance job
expands the production-important axes but still does not form the entire
Cartesian product.

### Timing method

The timed functions use precomputed scheduler metadata, persistent input
tensors, and preallocated output tensors.  Setup, tensor creation, page
packing, workspace creation, and CUDA Graph capture are outside the replay
measurement.

Defaults are:

- 100 untimed warm-up calls.
- 100 samples with 10 calls per sample, for 1,000 measured calls.
- CUDA events around each 10-call chunk, then latency amortized per call.
- Elementwise maximum sample across ranks as DCP critical-path latency.
- p50, p90, p99, mean, standard deviation, coefficient of variation, min, and
  max in the output record.

Increase calls per sample until the total measured window is at least one
second for sub-microsecond or low-microsecond paths.  CUDA Graph capture time
is recorded separately from replay and must never be averaged into replay.

The operator reports valid query tokens/s and effective TFLOPS.  Effective
work uses the actual causal visible-pair count for every logical query:

```text
2 * H * (Dq + Dv) * sum_bq(visible_k[b, q])
```

It must not use `B * physical_Q * KV` for ragged requests.

Run a real distributed suite with:

```bash
torchrun --standalone --nproc-per-node=2 \
  benchmark/bench_dcp_suite.py --profile smoke --cuda-graph \
  --output /tmp/dcp2-candidate.json

torchrun --standalone --nproc-per-node=8 \
  benchmark/bench_dcp_suite.py --profile smoke --cuda-graph \
  --output /tmp/dcp8-candidate.json
```

Available timing modes are `local`, `nccl_merge`, `nccl_operator`, and, for
DCP2, `peer_merge` and `peer_operator`.

## Performance gates

Baseline and candidate must run on the same Pod and GPU ranks in an interleaved
`baseline, candidate, candidate, baseline` order.  A result collected while
another inference workload is active is diagnostic only and must not become a
golden baseline.

Initial gates are:

- Reject a candidate case whose CV exceeds 2%.
- Critical-shape p50 may not regress by more than `max(3%, 1 us)`.
- Critical-shape p99 may not regress by more than `max(5%, 1 us)`.
- Equal-weight geometric-mean p50 may not regress by more than 2%.
- Select DCP2 peer over NCCL only when it wins at least four of five repeated
  rounds, improves p50 by at least 10%, and does not worsen p99.

Compare two machine-readable runs with:

```bash
python benchmark/compare_dcp_results.py \
  --baseline /tmp/dcp2-baseline.json \
  --candidate /tmp/dcp2-candidate.json
```

The peer/NCCL policy is calibrated by exact `(B,Q,H,Dv)` shape.  A single
payload-byte threshold is permitted only as a conservative deployment
approximation derived from that shape table.

After collecting five interleaved runs, generate the conservative exact-shape
selection table with:

```bash
python benchmark/select_dcp2_policy.py --stage operator \
  --runs /tmp/dcp2-run-{1,2,3,4,5}.json \
  --output /tmp/dcp2-policy.json
```

The default selector requires at least four wins, at least 10% p50
improvement, and no p99 regression.

## Environment and two-Pod protocol

Every JSON result records git SHA, dirty-worktree hash, PyTorch and CUDA
versions, GPU model and compute capability, world size, GPU UUIDs, driver,
P-state, clocks, and power when `nvidia-smi` exposes them.  The command line,
profile, warm-up, sample count, calls per sample, and CUDA Graph mode are also
stored.

Use the two 8-GPU development Pods as follows:

- Pod A runs the accuracy matrix, CP1/2/4/8 distributed correctness, and graph
  endurance.
- Pod B is reserved for performance A/B runs and targeted Nsight work.
- Repeat the canonical smoke matrix on both Pods to estimate node variance,
  but never subtract a baseline from one Pod from a candidate on the other.
- DCP2 peer evaluation uses two GPUs inside one Pod.  Two Pods are not a DCP16
  peer group.

Report the exact GPU, CUDA version, command, build flags, and pass/fail/skip
result.  On CUDA 12.8 or Hopper-only validation, state the
`FLASH_MLA_DISABLE_SM100=1` build restriction explicitly.

## Run cadence

| Cadence | Required work |
| --- | --- |
| Per change | CPU oracle tests, smoke global accuracy, existing dense DCP tests, DCP2 peer correctness when relevant |
| Nightly | 20-seed accuracy matrix, CP2/4/8 NCCL oracle tests, 10,000 graph replays, canonical performance matrix |
| Before merge/release | Same-Pod baseline/candidate A/B, both-Pod reproducibility check, targeted Nsight on regressions |
