# SM90 dense FP8 local attention：R3--R4 性能优化记录

## 结论

保留 **R4**：仅在 `batch_size == 1` 时把 dense FP8 metadata 的
`num_sm_parts` 从向下取整改为向上取整，使单请求长 KV 的最后一个 SM wave
覆盖完整 H20 的 78 个 SM；批量请求继续使用原 floor 调度，避免制造额外
partial output 和 combine 流量。

两台独立 NVIDIA H20 节点的 12 个 CUDA Graph shape 上，R4 相对 R1 local
attention baseline 的 p50 几何平均分别加速 **1.017x**（node-a）和
**1.019x**（node-b），自动门禁均为 **0 failures**。单请求长 KV 的收益更明显：
B1/Q4/KV4096 分别加速 **1.101x** 和 **1.092x**，B1/Q8/KV4096 分别加速
**1.147x** 和 **1.140x**。B8 子集保持在基线噪声范围。

R3 的无条件 ceil 方案被拒绝：虽然 node-b 的 B1 子集加速 **1.042x**，B8
子集却回退 6.65%，其中 B8/Q8/KV4096 从 205.894 us 回退到 270.571 us。

## 范围与环境

- 目标是 `csrc/extension/sm90/dense_fp8/` 的本地 FP8 attention 调度；R1 的
  DCP2 peer merge 保持不变，也没有修改 vendored CUTLASS。
- 基础提交：`3a0b48b8e7715ebc0e03aa09a1700868ccc97916`；R3/R4 都是该提交之上的
  未提交 working-tree revision。R4 C++ 文件 diff SHA256 为
  `3eda44c8f4918a67ab94c9578c4ac4568836811ae7171a12f1df0be1864db47e`。
- node-a：Pod `dev-sq-roleset-m88mv-decode-64f4b88598-0`，物理节点
  `192.168.122.78`。
- node-b：Pod `dev-sq-roleset-m88mv-prefill-94dd4bf87-0`，物理节点
  `192.168.1.179`。
- 两节点均为 8 x NVIDIA H20（SM90、NV18），PyTorch 2.11.0+cu130、CUDA/NVCC
  13.0、driver 535.161.08；性能 A/B 使用 GPU 0、1，distributed correctness
  使用 node-b 的 8 张 GPU。
- R4 隔离扩展在两节点的 SHA256 均为
  `4986ce0f43ad27a487b6692172870ff15800e376505cf85820d18f9293612b4d`。
- 构建设置：`FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=4 MAX_JOBS=8`。本报告只
  声明 SM90 验证。

两个 Pod 上都有既有模型服务和显存占用。测量前 GPU utilization 接近 0%，
baseline/candidate 在同一 Pod 内相邻执行，因此数据用于相对 A/B，不作为独占
节点的绝对 golden baseline。

## Baseline profile 与诊断

长 shape 为 CP=2、B8/Q8/H128/Dv512/KV4096；每个 rank 的本地 KV shard 为
2048。Nsight Systems 的 eager 结果显示：

- 主 `flash_fwd_splitkv_mla_kernel` 中位数 195.297 us；
- combine kernel 中位数 9.632 us；
- 主 attention 约占 local operator 的 95%，应优先优化主 kernel；
- 短 shape B1/Q1/KV128 的主 kernel 与 combine 中位数分别为 7.680 us 和
  1.472 us。

Nsight Compute 2025.3.1 使用等价单 GPU B8/Q8/KV2048 workload，完整 39-pass
profile 得到：

| 指标 | R1 baseline |
| --- | ---: |
| grid / block | `(16, 1, 4)` / `(256, 1, 1)` |
| CTA / H20 SM / waves per SM | 64 / 78 / 0.82 |
| registers / dynamic shared memory | 251 / 148.48 KB |
| theoretical / achieved occupancy | 12.50% / 12.45% |
| SM / DRAM throughput | 63.41% / 1.63% |
| one-or-more eligible / no eligible | 16.71% / 83.29% |
| barrier stall per issue-active ratio | 7.490847 |
| local spill requests | 0 |

每个 CTA 同时受寄存器和 shared memory 限制，只能每 SM 驻留 1 block。64 CTA
因此留下 14 个 SM 空闲；DRAM throughput 只有 1.63%，说明当前长 shape 不是
简单的 HBM 带宽饱和。`cuobjdump` 进一步确认主 kernel 为 REG=251、STACK=0、
LOCAL=0；SASS 无 `LDL`/`STL`，PTX 的 bulk KV load 使用 16-byte
`cp.async.cg.shared.global.L2::128B`，output 使用 128-bit store。基于这些证据，
首个单变量实验选择补齐 scheduler 的最后 SM wave，而不是改 GMMA 或访存宽度。

## Revision 迭代

完整机器可读数据见 [iterations.csv](./iterations.csv)。`performance` 为 12 个
shape 的 local operator p50 几何平均，括号内为相对同节点 R1 的比例和加速。

| revision_id | commit_info | 修改点 | node-a performance | node-b performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R1 | base `3a0b48b8`; binary `eff5e2f1...51b9` | floor `num_sm_parts`；保留已接受的 R1 merge | 25.803 us (1.000 / 1.000x) | 26.038 us (1.000 / 1.000x) | baseline |
| R3 | binary `1902ec3a...27159` | 所有 batch 无条件 ceil，64 CTA -> 80 CTA | 未运行 | 26.345 us (1.01182 / 0.988x) | **rejected** |
| R4 | source diff `3eda44c8...b47e`; binary `4986ce0f...2b4d` | 仅 B1 ceil；B>1 保留 floor | 25.371 us (0.98325 / **1.017x**) | 25.540 us (0.98089 / **1.019x**) | **accepted** |

### 关键 shape p50

| 节点 | shape | R1 | R4 | speedup |
| --- | --- | ---: | ---: | ---: |
| node-a | B1/Q4/KV4096 | 25.707 us | 23.350 us | **1.101x** |
| node-a | B1/Q8/KV4096 | 41.693 us | 36.365 us | **1.147x** |
| node-a | B8/Q8/KV4096 | 205.830 us | 205.770 us | 1.000x |
| node-b | B1/Q4/KV4096 | 26.050 us | 23.866 us | **1.092x** |
| node-b | B1/Q8/KV4096 | 41.738 us | 36.605 us | **1.140x** |
| node-b | B8/Q8/KV4096 | 205.894 us | 206.099 us | 0.999x |

### R3 为什么拒绝

R1 在 node-b 的 B8/Q8/KV2048-local metadata 中，每个 batch 只有一个
partial。R3 把 `num_sm_parts` 从 4 增至 5 后，部分 batch 被拆成两个 partial；
B8/Q4 从 9 增至 10 个 part 后，每个 batch 最多产生两个 partial。额外 partial
需要写 FP32 accumulator 并由 combine 重新读取，抵消了填满 SM 的收益。

补测 B2/B4 也证明不能用宽松的“小 batch”阈值：

| shape（KV4096 global） | R1 p50 | R3 p50 | 变化 |
| --- | ---: | ---: | ---: |
| B2/Q4 | 38.486 us | 36.019 us | +6.85% |
| B2/Q8 | 69.976 us | 71.971 us | -2.85% |
| B4/Q4 | 63.752 us | 66.475 us | -4.27% |
| B4/Q8 | 105.466 us | 157.347 us | -49.19% |

因此 R4 只优化单请求 latency：B1 本来就需要 split reduction，补齐最后 wave
的收益大于额外调度成本；B>1 则完全保留已验证的 floor 行为。

## 正确性与性能门槛

- node-a、node-b 各自：
  `pytest -q tests/test_dcp_eval_oracles.py tests/test_dense_fp8_dcp.py`，均
  **21 passed**。
- node-b 八卡：
  `torchrun --standalone --nproc-per-node=8 -m pytest -q
  tests/test_dense_fp8_dcp_distributed.py`，8 个 rank 均 **4 passed**；覆盖
  Q={1,2,4,8}、ragged `q_lens`、cyclic causal layout 和 NCCL merge 对全局参考。
- 性能矩阵：CP=2、B={1,8}、Q={1,4,8}、H=128、Dv=512、
  KV={128,4096}，full-Q；CUDA Graph，100 warm-up，100 samples，每 sample
  10 calls，两个 rank 取 critical path。
- `compare_dcp_results.py`：node-a `p50 geomean ratio=0.98325; failures=0`；
  node-b `p50 geomean ratio=0.98089; failures=0`。

## 数据索引

- node-a：[R1 baseline](./reports/node-a/r4-base.json)、
  [R4 candidate](./reports/node-a/r4-candidate.json)
- node-b：[R1 baseline](./reports/node-b/r3-base.json)、
  [R3 candidate](./reports/node-b/r3-candidate.json)、
  [R3 B2/B4 baseline](./reports/node-b/r3-extra-base.json)、
  [R3 B2/B4 candidate](./reports/node-b/r3-extra-candidate.json)、
  [R4 candidate](./reports/node-b/r4-candidate.json)
- Nsight Systems：[long baseline](./reports/node-b/r3-base-long.nsys-rep)、
  [short baseline](./reports/node-b/r3-base-short.nsys-rep)
- Nsight Compute：[full metrics](./reports/node-b/r3-base-long-full.ncu-rep)、
  [source metrics](./reports/node-b/r3-base-long-source.ncu-rep)
- 静态证据：[PTX](./reports/node-b/r3-base-main.ptx)、
  [SASS](./reports/node-b/r3-base-main.sass)

本轮使用 KernelAgent `ka-kernel-opt` 的 profile/diagnose/单变量
accept-reject 流程、`cuda` 的 Nsys/NCU/PTX/SASS 交叉验证方式，以及
KernelWiki 的 SM90 attention 调度资料。最终 accept/reject 仍以本仓库两台真实
H20 节点的正确性和相邻 A/B 数据为准。
