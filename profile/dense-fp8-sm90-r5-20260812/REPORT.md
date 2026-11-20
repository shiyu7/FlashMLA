# SM90 dense FP8 local attention：R5 性能优化记录

## 结论

保留 **R5**：删除 dense FP8 attention 两个 warp group 在 PV GEMM 前的两处
冗余 CTA `__syncthreads()`。两台独立 NVIDIA H20 节点的 12 个 CUDA Graph
shape 上，local p50 几何平均相对 R4 分别加速 **1.0225x** 和 **1.0251x**；
包含 local attention 与 merge 的 operator p50 几何平均分别加速 **1.0093x**
和 **1.0101x**。两个节点的全部 12 个 local shape 都更快。

完整 NCU 等价 workload 中，主 kernel 从 217.248 us 降至 209.984 us
（**1.0346x**），barrier stall 降低 **7.64%**，eligible warps 增加
**9.57%**；寄存器、occupancy 和 shared bank conflict 基本不变。因此本轮收益
可归因于 barrier elimination，而不是资源占用或 bank conflict 改善。

## 环境与恢复方式

| 项目 | node-decode | node-prefill |
| --- | --- | --- |
| Pod | `dev-sq-roleset-4bwl5-decode-858c8b85bf-0` | `dev-sq-roleset-4bwl5-prefill-d7c77b4b-0` |
| 物理节点 | `192.168.122.78` | `192.168.1.179` |
| GPU | 8 x NVIDIA H20；测试使用 GPU 0、1 | 8 x NVIDIA H20；测试使用 GPU 0、1 |
| 架构 | SM90 / CC 9.0 | SM90 / CC 9.0 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 | 2.11.0+cu130 / 13.0 |
| NVCC / driver | 13.0 / 535.161.08 | 13.0 / 535.161.08 |

基础提交为 `3a0b48b8e7715ebc0e03aa09a1700868ccc97916`，R4/R5 都是其上
现有 dirty worktree 的隔离快照。恢复时没有停止 Pod 中的模型服务；测量前使用的
GPU utilization 为 0，R4/R5 在同一 Pod 内相邻执行，所以数字用于相对 A/B，
不作为独占节点的绝对 golden baseline。

构建命令：

```bash
FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=4 MAX_JOBS=8 \
  python3 setup.py build_ext --build-lib /tmp/flashmla-r5-barrier-site-20260812
```

R4 扩展 SHA256 为
`27ddafadc510078f6e665ca9dfa7e90f3302626b82be807af5d435b04788dcf9`，
R5 为
`b3a3818e8d416f94bc51f1ef681468bb0e83d0d9c4c00ab456a8af2b0579d401`。
R5 ptxas 为 251 registers、12 barriers、0 spill，与 R4 资源占用相同。
虽然设置了禁用 SM100，`setup.py build_ext` 仍枚举了部分 SM100 source；实际本轮
code generation 均为 `sm_90a`，本报告只声明 SM90 验证。

## Revision 迭代

机器可读版本见 [iterations.csv](./iterations.csv)。`performance` 采用 12 个
local shape 的 p50 几何平均；operator 汇总包含 `nccl_operator` 和
`peer_operator` 共 24 个 shape-mode。

| revision_id | commit_info | 修改点 | node-decode performance | node-prefill performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R4 | base `3a0b48b8`; binary `27ddafad...cf9` | R4 scheduler，作为本轮 baseline | 25.677 us / 1.000x | 25.796 us / 1.000x | baseline |
| R5-VT | rejected patch `45947204...e43a` | conflict-free FP8 V transpose，并在 epilogue 恢复列排列 | 26.828 us / **0.9571x** | 未运行 | rejected：local 回退 4.29% |
| R5 | delta `15998b02...70f1`; binary `b3a3818e...d401` | 删除 PV GEMM 前两个冗余 FP8 CTA barrier | 25.112 us / **1.0225x** | 25.164 us / **1.0251x** | **accepted** |

### 关键 shape

| 节点 | shape | R4 p50 | R5 p50 | speedup |
| --- | --- | ---: | ---: | ---: |
| node-decode | B1/Q1/KV128 | 10.560 us | 10.205 us | **1.0348x** |
| node-decode | B1/Q8/KV4096 | 36.499 us | 35.602 us | **1.0252x** |
| node-decode | B8/Q4/KV4096 | 105.397 us | 102.166 us | **1.0316x** |
| node-decode | B8/Q8/KV4096 | 205.843 us | 199.315 us | **1.0328x** |
| node-prefill | B1/Q1/KV128 | 10.626 us | 10.182 us | **1.0435x** |
| node-prefill | B1/Q8/KV4096 | 36.534 us | 35.514 us | **1.0287x** |
| node-prefill | B8/Q4/KV4096 | 105.461 us | 102.274 us | **1.0312x** |
| node-prefill | B8/Q8/KV4096 | 205.805 us | 199.485 us | **1.0317x** |

## R5 修改与同步正确性

唯一的 accepted R5 source delta 位于
`csrc/extension/sm90/dense_fp8/flash_fwd_mla_kernel.h`：

1. WG0 等待 `TransVReady` 后，删除 PV GEMM 前的 CTA barrier。
2. WG1 等待 `SReady`、读完 P/scale 并 rescale 后，删除 PV GEMM 前的 CTA
   barrier。

循环头已有 CTA barrier，保证上一轮 PV 消费完 Vt 后下一轮才覆盖 shared
memory；当前轮的 `TransVReady` 保证 V transpose 完成，`SReady` 保证 P/scale
可见。原先紧跟在 named barrier 后、PV GEMM 前的 CTA barrier 没有再增加新的
producer/consumer 顺序。下一轮循环头 barrier 继续承担跨迭代保护。

R4→R5 的 unified diff SHA256 为
`15998b02d53d2cc8bc19e1c2cdf1874685a2b0be2b66b4c80abe1adf613d70f1`，
恰好只删除两行 `__syncthreads()`。

## NCU 证据

使用 node-decode 上等价单 GPU B8/Q8/H128/KV2048-local workload 完整 profile。
原始报告为 [R4](./reports/node-decode/r4-long-full.ncu-rep) 和
[R5](./reports/node-decode/r5-long-full.ncu-rep)，机器可读摘要见
[ncu_metrics.csv](./ncu_metrics.csv)。

| 指标 | R4 | R5 | 变化 |
| --- | ---: | ---: | ---: |
| 主 kernel duration | 217.248 us | 209.984 us | **-3.34%** |
| barrier stall / issue-active | 7.493000 | 6.920193 | **-7.64%** |
| eligible warps | 0.185236 | 0.202959 | **+9.57%** |
| issue active | 16.717812% | 17.261864% | **+3.25%** |
| tensor cycles active / elapsed | 15.889254% | 16.440845% | **+3.47%** |
| shared bank conflicts | 1,655,583 | 1,650,497 | -0.31% |
| registers / waves per SM | 251 / 0.82 | 251 / 0.82 | 不变 |

R4 source attribution 还显示 16 条热点 `STSM.16.M88.4`，每条 8-way，合计
1,048,576 excessive wavefronts，主要来自 FP8 V transpose。这个观察驱动了
R5-VT 实验，但其额外 epilogue layout/copy 令 ptxas registers 从 251 增到 255，
最终整体变慢。因此 shared conflict 仍是后续方向，但需要低成本、原位可逆的
layout 方案，不能用当前 epilogue 补偿方式。

## 正确性与测量方法

- node-decode：`test_dcp_eval_oracles.py` + `test_dense_fp8_dcp.py`，
  **21 passed in 1.94s**。
- node-prefill：同一命令，**21 passed in 1.90s**。
- 两节点都完成 CP=2、B={1,8}、Q={1,4,8}、H=128、Dv=512、
  KV={128,4096} 的 12-shape CUDA Graph smoke；100 warm-up、100 samples、
  每 sample 10 calls、两 rank 取 critical path。
- node-decode 的 R5 local 12/12 更快，speedup 范围 1.0084x--1.0348x；
  node-prefill 为 12/12 更快，范围 1.0098x--1.0435x。
- `git diff --check` 通过；没有修改 vendored CUTLASS。

## 数据索引

- node-decode：[R4 JSON](./reports/node-decode/r4.json)、
  [R5 JSON](./reports/node-decode/r5.json)、
  [R5-VT rejected JSON](./reports/node-decode/r5-vtranspose-rejected.json)
- node-prefill：[R4 JSON](./reports/node-prefill/r4.json)、
  [R5 JSON](./reports/node-prefill/r5.json)
- 被拒绝候选源码：[rejected-r5-vtranspose.patch](./rejected-r5-vtranspose.patch)

本轮继续使用 KernelAgent `ka-kernel-opt` 的 profile/diagnose/单变量
accept-reject 流程、`cuda` 的 NCU/SASS 交叉验证方式，以及 KernelWiki 的 SM90
warp-specialized attention 资料。最终结论以两台真实 H20 节点的正确性、相邻
A/B 数据和完整 NCU 为准。
