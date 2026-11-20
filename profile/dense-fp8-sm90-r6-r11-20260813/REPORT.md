# SM90 dense FP8 DCP optimization report: R6--R11

## Outcome

This round continued from the accepted R5 kernel and kept four cumulative
changes:

1. R6 replaces the large-launch split combine with a warp-owned row mapping
   and 128-bit `float4` accumulator loads, while preserving the old mapping for
   launches with at most 512 output rows.
2. R7 software-pipelines split accumulator loads in the combine kernel.
3. R8 overlaps the main attention producer and combine consumer with Hopper
   Programmatic Dependent Launch (PDL), again retaining the exact non-PDL path
   for at most 512 rows.
4. R11 selects eight rows per combine CTA only for the measured 1024-row
   crossover and keeps four rows per CTA for larger launches.

R9 and R10 were measured and rejected. The final accepted source is present in
the local worktree; it is not committed. The base commit remains
`3a0b48b8e7715ebc0e03aa09a1700868ccc97916`.

Against R5, final R11 local p50 geomean is:

| Node | R5 | R11 | Cumulative speedup | R5 peer operator | R11 peer operator | Cumulative peer-operator speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| node-decode | 25.1123 us | 23.2985 us | **1.07785x (+7.785%)** | 46.5923 us | 44.7626 us | **1.04087x (+4.087%)** |
| node-prefill | 25.1641 us | 23.3553 us | **1.07745x (+7.745%)** | 46.6755 us | 44.9014 us | **1.03951x (+3.951%)** |

The R11-only A/B gain against an immediately adjacent R8 run is +0.169% local
and +0.121% peer operator on node-decode, and +0.230% local and +0.058% peer
operator on node-prefill. Its targeted 1024-row subset reproduced more clearly:
**+0.550%** and **+0.533%** local geomean on the two independent nodes, with all
four target shapes positive on both.

## Environment and measurement contract

| Item | node-decode | node-prefill |
| --- | --- | --- |
| K8s Pod | `dev-sq-roleset-4bwl5-decode-858c8b85bf-0` | `dev-sq-roleset-4bwl5-prefill-d7c77b4b-0` |
| Physical node | `192.168.122.78` | `192.168.1.179` |
| GPU | 8 x NVIDIA H20; GPU 0 and 1 used | 8 x NVIDIA H20; GPU 0 and 1 used |
| Architecture | SM90 | SM90 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 | 2.11.0+cu130 / 13.0 |
| NVCC / driver | 13.0 / 535.161.08 | 13.0 / 535.161.08 |

The performance gate uses CP=2, B={1,8}, Q={1,4,8}, H=128, Dv=512, and
KV={128,4096}: 12 full-Q shapes. Each case uses a CUDA Graph, 100 warm-ups,
100 samples, 10 calls per sample, and the maximum of the two rank latencies as
the critical path. Summary values are p50 geometric means.

Build command:

```bash
FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=4 MAX_JOBS=8 \
  python3 setup.py build_ext --build-lib /tmp/flashmla-REV-site
```

The reused incremental build directory still contained previously compiled
SM100 objects and linked them into the extension. The dense FP8 source modified
in this round was explicitly recompiled with
`-gencode arch=compute_90a,code=sm_90a`; therefore this report claims SM90
validation only, not a clean SM90-only wheel build.

## Revision table

The complete machine-readable table, including intermediate R6 candidates, is
[`iterations.csv`](./iterations.csv). `performance` below is local p50 geomean;
the speedup is against the immediately previous accepted revision, using an
adjacent baseline for R10/R11.

| revision_id | commit_info | Main change | node-decode performance | node-prefill performance | Result |
| --- | --- | --- | ---: | ---: | --- |
| R5 | binary `b3a3818e...d401` | Accepted barrier baseline | 25.1123 us | 25.1641 us | baseline |
| R6 | binary `5c6894de...892e` | Hybrid warp-owned combine + `float4` loads | 24.1220 us (**1.04105x**) | 24.2350 us (**1.03833x**) | **accepted** |
| R7 | binary `a278e07f...0969` | Prefetch and pipeline the next split | 23.7689 us (**1.01486x**) | 23.9006 us (**1.01399x**) | **accepted** |
| R8 | binary `5ccffc06...bf4e` | PDL only above 512 rows | 23.3606 us (**1.01748x**) | 23.3975 us (**1.02150x**) | **accepted** |
| R9 | binary `e7588eb4...d084` | Eight rows/CTA for every grouped launch | 23.4079 us (**0.99798x**) | not run | rejected |
| R10 | binary `83a53402...28f` | Private split LSE in base-2 space | 23.3685 us (**0.99957x** in repeat A/B) | not run | rejected |
| R11 | header `6eac2fed...b05`; binary `ba1ec1e8...708` | Eight rows/CTA only at 1024 total rows | 23.2985 us (**1.00169x**) | 23.3553 us (**1.00230x**) | **accepted** |

## R6: combine remapping and vector loads

The R5 combine launch had one 128-thread CTA for every output row. NCU on the
B8/Q8/H128/KV2048 world-size-1 workload measured a 10.880 us combine kernel,
48 registers/thread, 10.5 waves/SM, 98,304 global-load sectors/requests in the
selected metric, and no spill. The launch exposed four output floats to each
thread but repeated narrow access across one CTA per row.

R6 assigns one warp to one output row and groups four rows in a CTA. Each lane
owns four consecutive FP32 accumulator values and loads them as a `float4`.
The original mapping remains the fallback for `total_rows <= 512`, because the
large per-thread accumulation of the grouped kernel regressed B1/Q1 long-prefix
cases. Candidate geomeans were:

| Candidate | local p50 geomean | Observation |
| --- | ---: | --- |
| grouped 8 rows/CTA everywhere | 24.2674 us | B1/Q1/KV4096 -7.86% |
| grouped 4 rows/CTA everywhere | 24.2481 us | B1/Q1/KV4096 about -5.0% |
| grouped 1 row/CTA everywhere | 24.5626 us | B1/Q1/KV4096 about -4.1% |
| hybrid fallback + 4 rows/CTA | **24.1220 us** | range -0.14% to +16.69%; accepted |

The accepted combine NCU duration was 6.912 us (**1.574x** versus R5), while
the selected global-load metric fell 4x from 98,304 to 24,576. It used 109
registers/thread but had zero spill.

An intentionally invalid diagnostic removed the FP8 V transpose to estimate
its upper bound. It reached 24.6656 us, only 1.0181x over R5 and about 0.09% on
B8/Q8/KV4096. This showed the transpose was mostly hidden and prevented a broad,
high-risk transpose rewrite.

## R7: software-pipelined output accumulation

R7 preloads split 0's output accumulator before the LSE reduction. While the
current split is multiplied and accumulated, it loads the next split into the
same register buffer. This mirrors the producer/consumer organization of the
repository's newer `csrc/smxx/decode/combine/combine.cu` implementation.

The compiler reduced grouped-kernel register use from 109--110 to 66--68 with
zero spill. On the same B8/Q8 NCU workload, combine duration fell from 6.912 to
5.600 us, active warps rose from 3.489 to 5.678, eligible warps from 0.486 to
0.973, and issue active from 35.315% to 49.330%.

## R8: guarded Hopper PDL overlap

R8 launches the main attention kernel and grouped combine with
`cudaLaunchKernelEx` and programmatic stream serialization. The producer calls
`cudaTriggerProgrammaticLaunchCompletion()` before its final tile writes; the
consumer calls `cudaGridDependencySynchronize()` before every path that reads
accumulators or writes final output.

The first draft placed the consumer dependency wait after the ragged-row early
path and failed 4 of 21 tests because final output could be written before the
producer dependency was satisfied. Moving the wait ahead of all output paths
restored correctness. A safe all-shape PDL version still regressed the
B1/Q1/KV4096 high-sample case from about 15.4 to 18.4 us, so the final R8 uses
PDL only when `total_rows > 512`; small launches compile and run the exact R7
non-PDL path.

The accepted hybrid improved local geomean by 1.748% / 2.150% and peer operator
by 1.548% / 1.646% on node-decode / node-prefill.

## R9 and R10: rejected experiments

R9 changed the grouped launch from four to eight rows per CTA for every large
shape. It helped the four 1024-row cases by roughly 0.4%--0.9%, but regressed
B8/Q4/KV128 by 2.33% and B8/Q8/KV128 by 1.70%; full local geomean fell 0.202%.
The source was reverted before R10.

R10 stored private split LSE accumulators in base-2 space, used `exp2f/log2f`
in combine, and converted only the public final LSE to natural-log space. Its
CP={1,2,4,8}, Q={1,2,4,8} accuracy matrix passed 16/16. Performance did not
survive repeat A/B: the first run showed +0.154% local, while the adjacent repeat
showed -0.043% local and -0.011% peer operator. It was classified as noise and
fully reverted; final `softmax.h` retains the R8 natural-log contract.

## R11: measured 1024-row crossover

R11 converts R9's useful subset into a host-side policy with no device hot-path
branch: use an eight-row (256-thread) grouped CTA only when
`total_rows == 1024`; use the four-row grouped CTA for all other large launches
and the original fallback through 512 rows.

Adjacent R8 -> R11 A/B by output-row group:

| Node | 128 rows | 512 rows | 1024 rows (changed) | 4096 rows | 8192 rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| node-decode | -0.234% | -0.245% | **+0.550%** | +0.110% | +0.287% |
| node-prefill | +0.307% | -0.037% | **+0.533%** | +0.142% | -0.095% |

Unchanged groups serve as the measurement-noise control. The changed 1024-row
group reproduced on both physical nodes, and all four constituent shapes were
positive on each node.

Full NCU on B8/Q1/H128/KV4096 world-size-1 confirms the intended launch change:

| Metric | R8 rows4 | R11 rows8 | Change |
| --- | ---: | ---: | ---: |
| Grid / block | 256 / 128 | 128 / 256 | half CTAs, twice warps/CTA |
| Duration | 7.360 us | 7.328 us | **1.00437x** |
| Registers / spill requests | 66 / 0 | 66 / 0 | unchanged |
| Waves/SM | 0.47 | 0.55 | +17.0% |
| Active warps | 3.1433 | 3.1899 | +1.48% |
| Eligible warps | 0.2118 | 0.2146 | +1.31% |
| Global-load metric | 334,848 | 334,848 | unchanged |

Raw NCU reports and the machine-readable summary are in
[`reports/node-decode`](./reports/node-decode) and
[`ncu_metrics.csv`](./ncu_metrics.csv).

## Correctness and binary audit

Final R11 validation:

- node-decode: `pytest -q tests/test_dcp_eval_oracles.py
  tests/test_dense_fp8_dcp.py`: **21 passed in 1.88 s**.
- node-prefill: the same command: **21 passed in 1.94 s**.
- Both nodes also ran the two-GPU distributed dense FP8 DCP test; each rank
  reported **14 passed**.
- node-decode `eval_dcp_accuracy.py --profile smoke --permute-pages`:
  **16/16 PASS** for CP={1,2,4,8}, Q={1,2,4,8}; maximum output absolute p99
  approximately 0.002393 and maximum LSE absolute error approximately 3.99e-6.
- ptxas: grouped rows4/rows8 kernels use 66--68 registers, 24-byte stack frame,
  zero spill stores/loads; main attention variants use 251 registers, 12
  barriers, and zero spills.
- Final decode extension SHA256:
  `ba1ec1e8e9e5cefa7366dedadad2eb6e0055976c455253f26ea9fe2ffc07e708`.
- Final prefill extension SHA256:
  `4d910f0225cf6797e37dc284c672cd9e2d31efa2775d9f595acac55a22f0b2a1`.
- Final local kernel header SHA256:
  `6eac2fed9801559209766ba1c7112be5505764b446cdf85be1b2adb5ced9cb05`.
- `git diff --check` passes; vendored CUTLASS was not modified.

## Data index

- [`iterations.csv`](./iterations.csv): revision, commit/binary identity, change,
  performance, and accept/reject decision.
- [`ncu_metrics.csv`](./ncu_metrics.csv): selected R5/R6/R7/R8/R11 NCU metrics.
- [`reports/node-decode`](./reports/node-decode): all decode benchmark JSON,
  repeat A/B data, accuracy JSON, and NCU reports.
- [`reports/node-prefill`](./reports/node-prefill): independent-node benchmark
  JSON and repeat A/B data.

The iteration policy followed the installed KernelAgent optimization skill's
single-variable profile -> verify -> accept/reject loop. CUDA profiling guidance
was used for non-interactive NCU capture, and Hopper-specific KernelWiki guidance
informed the warp-owned combine and guarded PDL design. Acceptance decisions are
based on the repository's real two-node H20 measurements above.
