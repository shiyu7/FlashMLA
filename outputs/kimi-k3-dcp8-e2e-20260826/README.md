# Kimi-K3 DCP8/DP4 four-node validation (2026-08-26)

## Scope

- Build the latest local SM90 DCP FlashMLA extension in the requested
  `sglang:kimi-0810` container.
- Launch Kimi-K3 on four 8-GPU H20 nodes with TP32, DCP8, DP4, and EP32.
- Exercise the external dense-FP8 DCP FlashMLA path under CUDA Graph capture.
- Run the supplied `sglang_auto_bench.py` unchanged against a unified service.

## Topology

| node rank | public address | private address | GPU | model host path |
|---:|---|---|---|---|
| 0 | `101.126.12.182` | `192.168.44.92` | 8 x H20 (SM90) | `/data01/models/Kimi-K3` |
| 1 | `115.191.2.23` | `192.168.44.90` | 8 x H20 (SM90) | `/data01/models/Kimi-K3` |
| 2 | `115.191.51.207` | `192.168.44.91` | 8 x H20 (SM90) | `/data00/Kimi-K3` |
| 3 | `115.191.21.96` | `192.168.44.93` | 8 x H20 (SM90) | `/data00/Kimi-K3` |

All four model copies contain 96 weight shards and were hash-checked. The
container path is `/data/models/models/Kimi-K3` on every node.

## Reproducible artifacts

| artifact | revision / SHA256 |
|---|---|
| container image | `iaas-gpu-cn-beijing.cr.volces.com/sunqi/sglang:kimi-0810`, digest `sha256:7c4f…` |
| FlashMLA-DCP source | branch `codex/dcp2-nvlink-merge`, commit `3a0b48b8e7715ebc0e03aa09a1700868ccc97916` |
| FlashMLA-DCP source archive | `71efcc3800d4d893c8a117b78549ef49e3a212831b3ade9df43e86c8b46f002b` |
| SGLang Kimi-K3 base | commit `f748ae35a26fbe1be98db09967ffb828658b821a` |
| final deployed K3+DCP overlay archive | `sglang-kimi-k3-f748-dcp-run15.tar.gz`, `ecfba1d57f28fa3d0df53e907793f7e665306ee32304b3ac32c93acabcd25f27` |
| normalized launch script | `kimi-dcp-serve-20260826.sh`, `5cee3614ffbcf5d8b7405eb4c84f99b406e71745caa2c5a066a85d302ef867b3` |
| benchmark wrapper | `kimi-dcp-bench-20260826.sh`, `9ab58d6413a039fdfac309890c4a7f43c6ae440c56a5b90f1720de496246f868` |
| fixed 8K A/B wrapper | `kimi-dspark-8k-fixed-ab-20260826.py`, `88d0490865fd0e49f87e8cfbc03838736d8c89a7552bcd9bbed1c6759b54bd4f` |
| supplied benchmark script | `6d36d7d14637bc46257dc54b6c70630f4de3a3bb17652020a53f3658a511d27e` |

The container is named `kimi-dcp-20260826`; host work products are under
`/data00/kimi-dcp`, mounted at `/workspace/kimi-dcp`.

## Build and narrow validation

The extension was built on every node with:

```bash
FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=8 MAX_JOBS=32 \
  python3 setup.py build_ext --inplace
```

Validated on CUDA/H20:

- Python import reports architecture `(9, 0)` and exposes the dense-FP8 DCP API.
- Narrow DCP pytest suite: 21 tests passed.
- 8-GPU distributed DCP suite: 8 ranks passed.
- 32-GPU, four-node NCCL/RDMA smoke test: passed with `mlx5_1` through `mlx5_4`.

## Integration changes

| file | purpose | SHA256 |
|---|---|---|
| `sglang/srt/layers/attention/flashmla_backend.py` | cyclic DCP metadata/page tables, external dense-FP8 launch, split MLA Q/K compatibility, Q=8 target verify, base-2 LSE handoff | `de517c9a9c84e439a6a0eecaef61f3510ebaa6c0fa6b9fadaefaf9b09a72e3df` |
| `sglang/srt/mem_cache/memory_pool.py` | map logical DCP cache locations to rank-local storage | `63216356be33a8e3e1052a8e126614d13ac316826e9f52f329644f90ff89fc24` |
| `sglang/srt/arg_groups/overrides.py` | enable Kimi-K3 DCP FlashMLA decode while preserving explicit Triton prefill | `cfd5de8039e77f8171952cfc5ab9fc9fa448aef1d121f57fb2bff236a4706b88` |
| `sglang/srt/models/deepseek_common/utils.py` | route FlashMLA through the standard DCP `out/lse` A2A merge | `0bd5410fad950ca04185faf8394ca9674c5f30984bf84fca6c890aaf3aa0774d` |
| `sglang/srt/layers/moe/moe_runner/marlin.py` | experimental Marlin/DeepEP bridge; present in overlay but inactive in run14 | `6ec5d9134d036ae854e59c56c7d31538329b66dc69cf7895617005042af78c3b` |
| `sglang/srt/layers/quantization/mxfp4.py` | experimental MXFP4 bridge companion; present but inactive in run14 | `556d5e39eaa5c47dce8b8f6ec072a3c1f9fd09699ed46afc542e78fc5239c55b` |

The service uses `--disable-custom-all-reduce`: node rank 2 had four odd GPU
ranks diverge in `custom_all_reduce.register_graph_buffers()` during graph
capture. This flag removes that unrelated local custom-allreduce path while
retaining NCCL, DCP A2A, and the external FlashMLA kernel.

The reference `--mem-fraction-static 0.95` left insufficient graph-capture
headroom on H20. The validated configuration uses `0.85`. Run14 omitted
ReplaySSM; run18/run19 enable the branch's current
`--enable-linear-replayssm-spec` spelling together with DSPARK.

## Deployment and E2E result

Run14 is a unified four-node service (`disaggregation_mode=null`) with the
following resolved attention paths:

- prefill/extend: Triton MLA;
- decode: external dense-FP8 DCP FlashMLA;
- DCP merge: SGLang standard `out/lse` A2A merge;
- MoE: Marlin with `moe_a2a_backend=none` (no peer operator acceleration).

Startup validation passed:

- four nodes loaded 66.11 GB of weights per GPU in 65--66 seconds;
- Mamba cache: 5.61 GB; FP8 KV cache: 571,264 local tokens / 7.36 GB;
- decode CUDA Graph capture: 13.1 seconds, 3.52--3.57 GB, 8.20--8.25 GB free;
- native K3 VLM warmup completed and `/health` returned HTTP 200;
- a real 8-token completion returned HTTP 200 in 3.64 seconds.

The supplied benchmark script was run unchanged (SHA256 above). Its first
completed point was:

| input/output | RR | MC | TTFT | TPOT | E2E | total throughput | SLO |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8000/1500 | 1.0 | 8 | 39,848.12 ms | 85.33 ms | 167,763.35 ms | 447.84 tok/s | fail (5 s / 20 ms) |
| 8000/1500 | 0.8 | 8 | 26,584.24 ms | 88.76 ms | 159,636.72 ms | 466.80 tok/s | fail (5 s / 20 ms) |
| 8000/1500 | 0.6 | 8 | 27,622.16 ms | 89.99 ms | 162,519.60 ms | 457.87 tok/s | fail (5 s / 20 ms) |
| 8000/1500 | 0.4 | 8 | 26,626.35 ms | 91.52 ms | 163,820.65 ms | 449.79 tok/s | fail (5 s / 20 ms) |
| 8000/1500 (Phase 2) | 0.2 | 8 | 23,283.16 ms | 97.37 ms | not reported in phase summary | 422.47 tok/s | saturated at 0.04 req/s |

The earlier adaptive benchmark ran under `/data00/kimi-dcp/results/dcp8`; its
script start time was `2026-08-26T12:56:34Z`. The high TPOT is consistent with the unaccelerated
peer/MoE path; DCP FlashMLA decode itself is stable under CUDA Graph and shows
about 26.6--27.3 output tok/s per DP at two concurrent sequences.

The completed 8k point logs, benchmark stdout snapshot, smoke response, and
all four service-log snapshots are stored under `logs/` beside this document.

## DSPARK 8K fixed-point DCP A/B (run18/run19)

To obtain a fast, directly comparable signal before running another adaptive
search, run18 and run19 used exactly one point from the supplied benchmark
engine:

- random input/output length: `8000/1500`;
- request rate: `1.0 req/s`;
- maximum client concurrency: `8`;
- prompts: `16`, seed `5`;
- common serving topology: TP32, DP4, EP32, Marlin, DSPARK block size 7,
  ReplaySSM, FP8 KV, and the built-in TP MoE path;
- the only requested topology change was `dcp_size=8` versus `dcp_size=1`.

DeepEP is installed in the container, but cannot be included in this DSPARK
A/B. The deployed SGLang revision rejects the combination during argument
validation with: `DSpark with dp attention only supports the built-in TP MoE
(moe_a2a_backend='none')`. Here `none` means no external token-dispatcher
plugin; it does not disable expert routing or the built-in TP dispatch/combine
path.

Both runs completed all 16 requests and generated the same 24,000 output
tokens:

| mode | duration | mean TTFT | mean TPOT | mean E2E | total tok/s | output tok/s | DSPARK accept length |
|---|---:|---:|---:|---:|---:|---:|---:|
| DCP8, run18 | 472.84 s | 32,986.28 ms | 128.86 ms | 226,150.02 ms | 321.46 | 50.76 | 1.11 |
| DCP1, run19 | 155.65 s | 14,200.70 ms | 39.83 ms | 73,912.38 ms | 976.56 | 154.19 | 6.27 |

At this fixed point, DCP1 delivered `3.04x` the total/output throughput of
DCP8. Relative to DCP8, it reduced mean TTFT by 56.9%, TPOT by 69.1%, and E2E
latency by 67.3%. Neither mode met the 5 s TTFT / 20 ms TPOT SLO.

The dominant diagnostic signal is not ordinary DCP communication overhead:
DSPARK accept length fell from 6.27 without DCP to 1.11 with DCP. Runtime logs
showed DCP8 accept rates around 0--6%, while DCP1 stayed around 73--77%. This
strongly points to a DCP-specific target-verify numerical or semantic mismatch
in the new Q=8 path; most draft tokens are rejected, erasing DSPARK's benefit.
It should be treated as a correctness/integration blocker before further DCP
performance tuning.

CUDA Graph startup also recorded a memory cost for the DCP target-verify path:
approximately 4.15--4.20 GB for DCP8 versus 3.22--3.27 GB for DCP1, leaving
roughly 1.1 GB less free memory after target capture.

Local evidence is preserved in:

- `logs/dcp8-dspark-run18-result-8k-fixed.json` and
  `logs/dcp1-dspark-run19-result-8k-fixed.json`;
- `logs/dcp8-dspark-run18-bench-serving-raw.log` and
  `logs/dcp1-dspark-run19-bench-serving-raw.log`;
- `logs/dcp8-dspark-run18-serve-rank0.log` and
  `logs/dcp1-dspark-run19-serve-rank0.log`.

Run18 finished at `15:31:31Z`. Its rank0 service log contains Gloo
`Connection closed by peer` traces at `15:32:13`, after the successful result
was written, when the four-node service was deliberately terminated to free
the GPUs and launch the DCP1 control. Those shutdown traces are not benchmark
failures.

## DSPARK DCP precision fix and matched rerun (run31--run34)

### Root cause

The accept collapse was not caused by the SM90 FlashMLA target-verify kernel.
The dense Qwen3 DSPARK draft has 64 query heads and 16 KV heads and runs TP8,
so every TP rank owns different KV heads. `draft_tp_context()` patched the
target TP group to the draft TP group, but left the target's live DCP fields
unchanged. Under DCP8, the draft Triton backend therefore initialized with
`attn_dcp_size=8`: it treated the eight ranks as sequence shards of the same KV
heads, all-gathered query heads, and LSE-merged results backed by different
TP-sharded KV heads. DCP1 never entered this invalid path.

The fix makes the replicated draft context explicitly non-DCP while retaining
its TP8 group. This matches the existing draft KV configurator, which already
allocates a replicated pool large enough to consume the target allocator's
widened virtual DCP locations. A companion fix maps speculative accepted-token
KV relocation from widened logical DCP locations to the owning rank's compact
local rows; applying this relocation fix alone in run21 did not recover
acceptance, but it is required once draft tokens are accepted again.

Final source changes are saved in the local SGLang worktree:

- `python/sglang/srt/speculative/spec_utils.py`: force
  `dcp_enabled=False`, `dcp_size=1`, `dcp_rank=0`,
  `attn_dcp_size=1`, and `attn_dcp_rank=0` inside `draft_tp_context()`;
- `python/sglang/srt/mem_cache/memory_pool.py`: owner-filter and translate
  logical source/target locations in `MLATokenToKVPool.move_kv_cache()`;
- `test/registered/unit/test_runtime_context.py`: verify the draft context
  masks target DCP and restores all fields on exit.

The cleaned deployment archive is
`sglang-kimi-k3-draft-dcp-fix-20260827.tar.gz`, SHA256
`2aba72b8757c3ac7c7b70d30cbb963fbbf076078b6708c1ecc98b03d072a69ff`.
It was extracted on all four nodes. Diagnostic-only local-reference and
communication-backend switches used during isolation are not present in this
archive.

### Isolation evidence

- The 8-GPU H20 distributed operator test passed all four cases on every rank:
  CP8, Q={1,2,4,8}, including ragged and empty local rows.
- A real 8K/Q8 target-verify capture reconstructed each rank's runtime page
  table and compared local FlashMLA against PyTorch. Across 32 processes,
  output max absolute error was at most 0.0057 and LSE max absolute error was
  at most 0.0313.
- Changing DCP merge from A2A to AG/RS, disabling replicated Q projection,
  disabling CUDA Graphs, and reducing input length from 8K to 1K did not
  recover acceptance. TokenSpeed/CuTeDSL MLA could not serve as an H20 oracle
  because this revision restricts them to Blackwell.
- After masking DCP only in the draft context, the 1K/512 smoke test reached
  accept length **7.81**, TPOT **11.07 ms**, and output throughput
  **293.61 tok/s**.

### Matched 8K/1500 rerun

Run32 and run34 used the same fixed script and workload as run18/run19:
16 prompts, seed 5, request rate 1.0, max concurrency 8, TP32/DP4/EP32,
DSPARK block size 7, ReplaySSM, Marlin, FP8 KV, and 24,000 generated tokens.

| mode | mean TTFT | mean TPOT | mean E2E | total tok/s | output tok/s | accept length |
|---|---:|---:|---:|---:|---:|---:|
| DCP8 before fix, run18 | 32,986.28 ms | 128.86 ms | 226,150.02 ms | 321.46 | 50.76 | 1.11 |
| DCP8 fixed, run32 | **9,789.76 ms** | **39.77 ms** | **69,407.08 ms** | **951.88** | **150.30** | **4.74** |
| DCP1 matched control, run34 | 11,704.78 ms | 39.43 ms | 70,816.79 ms | 1,013.58 | 160.04 | 6.17 |

Relative to the broken DCP8 run, the fix improved total/output throughput by
**2.96x**, increased accept length by **4.27x**, reduced TTFT by **70.3%**,
reduced TPOT by **69.1%**, and reduced E2E latency by **69.3%**. Against the
matched DCP1 control, fixed DCP8 had 16.4% lower TTFT, 0.86% higher TPOT, 1.99%
lower E2E latency, and 6.09% lower throughput. Its exact-match accept length
remains 23.2% lower (4.74 versus 6.17), but the catastrophic collapse is gone
and end-to-end decode performance is near parity.

Neither group meets the original 5 s TTFT / 20 ms TPOT SLO at this workload.
The remaining gap is no longer dominated by draft corruption; future tuning
can treat DCP communication, target numerical-order effects, and the full K3
MoE/ReplaySSM path as performance work rather than a blocking precision bug.

Local raw artifacts:

- `logs/dcp8-draft-nodcp-fix-run32-8k-fixed/`;
- `logs/dcp1-draft-nodcp-control-run34-8k-fixed/`;
- `logs/dcp8-draft-nodcp-fix-run31-smoke-1k.log` and
  `logs/dcp8-draft-nodcp-fix-run31-smoke-1k-512.log`;
- `logs/dcp8-draft-nodcp-fix-run31-rank{0,1,2,3}.log` and
  `logs/dcp1-draft-nodcp-control-run33-rank{0,1,2,3}.log`.

## R13/R14 follow-up artifacts (2026-08-27)

R13 adds an opt-in `SGLANG_DCP_FP32_LOCAL_OUTPUT=1` path. Rank-local FlashMLA
output remains FP32 through the SGLang DCP merge and is cast once at the
attention ABI boundary. The four-node launch should otherwise remain byte-for-byte
equivalent to run32, and the test must use paired `0/1` runs rather than compare
against an older service.

Saved local deployment artifacts:

| artifact | SHA256 | purpose |
| --- | --- | --- |
| `sglang-kimi-k3-r13-fp32-local-20260827.tar.gz` | `e1fe2122e910d7af9fe507e79220784de9deaca0e636aafbd1b83d25262e9aaf` | two-file R13 delta on top of the run15/R12 overlay |
| `flashmla-dcp-r13-r14-src-20260827.tar.gz` | `aa1d46e0df2cbc9f75c1417527d7cf7f2666fe3a91a1445927f4860eef60cc93` | FlashMLA source without CUTLASS/build/profile artifacts |

H20 component results are recorded in
[`profile/dense-fp8-sm90-r13-20260827`](../../profile/dense-fp8-sm90-r13-20260827/REPORT.md).
R13 improves mean NRMSE by only 0.394%; the production SGLang component model
estimates Q1/Q8 attention latency regressions of 1.95%/2.96%.

### R13 four-node fixed 8K result (run35/run36)

The target image was rebuilt on all four ECS nodes with
`FLASH_MLA_DISABLE_SM100=1`; PyTorch 2.11.0+cu130, CUDA 13.0, and 8 x H20 per
node were used. The cloud image passed 14/14 narrow tests and the eight-GPU
distributed test passed 4/4 on every rank.

Both services used TP32/DP4/EP32/DCP8, DSPARK block 7, ReplaySSM, Marlin, FP8
KV, 8K input, 1500 output, 16 prompts, request rate 1, and maximum concurrency
8. The binary and overlay were identical; only
`SGLANG_DCP_FP32_LOCAL_OUTPUT=0/1` changed. Three repetitions were collected
per service, with `/flush_cache` between repetitions.

| path | rep | TTFT | TPOT | E2E | output tok/s | accept rate / weighted accept len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 local, run35 | 1 | 10,269.69 ms | 37.94 ms | 67,139.45 ms | 163.11 | 71.23% / 5.983 |
| BF16 local, run35 | 2 | 9,732.26 ms | 39.59 ms | 69,070.76 ms | 152.82 | 67.12% / 5.699 |
| BF16 local, run35 | 3 | 9,814.25 ms | 34.22 ms | 61,112.70 ms | 184.91 | 77.96% / 6.459 |
| FP32 local, run36 | 1 | 9,775.09 ms | 42.33 ms | 73,225.29 ms | 146.44 | 60.79% / 5.257 |
| FP32 local, run36 | 2 | 9,725.13 ms | 38.45 ms | 67,359.65 ms | 156.40 | 70.66% / 5.945 |
| FP32 local, run36 | 3 | 9,766.81 ms | 35.10 ms | 62,384.27 ms | 181.08 | 76.28% / 6.344 |

| three-run summary | BF16 mean / median | FP32 mean / median | FP32 relative mean / median |
| --- | ---: | ---: | ---: |
| TTFT | 9,938.73 / 9,814.25 ms | 9,755.68 / 9,766.81 ms | -1.84% / -0.48% |
| TPOT | 37.25 / 37.94 ms | 38.63 / 38.45 ms | **+3.70% / +1.34%** |
| E2E | 65,774.30 / 67,139.45 ms | 67,656.40 / 67,359.65 ms | **+2.86% / +0.33%** |
| output throughput | 166.95 / 163.11 tok/s | 161.31 / 156.40 tok/s | **-3.38% / -4.11%** |
| active-request-weighted accept rate | 72.10% / 71.23% | 69.24% / 70.66% | **-2.86 / -0.57 pp** (relative **-3.97% / -0.80%**) |
| active-request-weighted accept len | 6.047 / 5.983 | 5.849 / 5.945 | **-3.28% / -0.63%** |

The system-level variance is material, and the two conditions were separate
service launches rather than interleaved trials. Nevertheless, FP32 local did
not produce a consistent acceptance improvement, while its performance
direction matches the component estimate. R13 is therefore
**performance-rejected** and remains opt-in only.

Raw evidence is preserved in `logs/dcp8-r13-fp32local-{off,on}-run*-*`, with
the machine-readable aggregate in
`profile/dense-fp8-sm90-r13-20260827/e2e_results.csv`. The Gloo/TCPStore
warnings after run35 occurred during the intentional container restart, 40
seconds after rep3 finished, not inside a measurement window.

R14-P0 proves a DCP-size-invariant fixed-chunk/fixed-tree algorithm, but its
unfused oracle is about 77x slower than R13 and is not included in the service
overlay. See
[`profile/dcp-canonical-r14-20260827`](../../profile/dcp-canonical-r14-20260827/REPORT.md).

After collecting and downloading all artifacts, the four `kimi-dcp-20260826`
containers were restarted into their `sleep infinity` entrypoint. All 32 H20s
returned to 0--3 MiB used memory. No password or credential was copied into
the artifacts.
