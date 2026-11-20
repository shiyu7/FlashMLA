# SM90 DCP 性能优化记录

## 调试环境硬约束（必须遵守）

**DCP 开发、编译、精度验证、性能测试和端到端实验必须且只能使用下面四台
ECS 云服务器。禁止使用 K8s Pod 代替这些 ECS 生成正式数据或接受结论。**

SSH 登录必须使用公网 IP，并显式指定密钥
`/Users/bytedance/perfTest.pem`；`192.168.44.x` 私网 IP 只用于 ECS 节点间的
NCCL/GLOO、`--dist-init-addr` 和其他分布式通信，不从本地 Mac 直接登录。

| 本轮 `node_rank` | 公网 IP（SSH 登录） | 私网 IP（节点间通信） |
| ---: | --- | --- |
| 0 | `101.126.12.182` | `192.168.44.92` |
| 1 | `115.191.51.207` | `192.168.44.91` |
| 2 | `115.191.21.96` | `192.168.44.93` |
| 3 | `115.191.2.23` | `192.168.44.90` |

登录模板：

```bash
ssh -i /Users/bytedance/perfTest.pem root@<公网IP>
```

从本节加入起，K8s 上已有的 R14-P1--P6 数字仅视为算法原型/预筛选证据，
**不计入正式 revision 接受结果**。相关 revision 必须在上述 ECS 上重新编译并
按同一 warm-up、精度和性能协议复测后，才能标记为 ECS accepted。

## 结论

本轮保留 R1：把 DCP2 owner merge 从 256-thread、BF16x2 窄访存改为
64-thread、`uint4` 128-bit 向量化访存。两台独立 NVIDIA H20 节点的 12 个
CUDA Graph shape 上，`peer_merge` p50 几何平均分别加速 **1.578x** 和
**1.589x**，包含本地 FlashMLA 的 `peer_operator` 分别加速 **1.262x** 和
**1.262x**。B8/Q8/H128/Dv512/KV4096 的 `peer_merge` 从约 120 us 降至
约 46 us，双节点加速分别为 **2.617x** 和 **2.605x**。

R2 的单 warp 方案在大 shape 继续快约 2%，但 B1/Q1 回退 7.8%--9.0%，
全矩阵几何平均相对 R1 回退 1.87%，因此拒绝。

后续 R3--R4 转向本地 dense FP8 attention。最终保留 R4：仅对
`batch_size == 1` 使用向上取整的 `num_sm_parts` 以补齐 H20 的最后 SM wave，
批量请求保持原 floor 调度。两节点 12-shape local p50 几何平均分别加速
**1.017x** 和 **1.019x**；B1/Q8/KV4096 分别加速 **1.147x** 和
**1.140x**。详细 profile、拒绝实验和原始数据见
[`dense-fp8-sm90-r3-20260810/REPORT.md`](../profile/dense-fp8-sm90-r3-20260810/REPORT.md)。

在重新恢复的两台 H20 节点上继续得到并保留 **R5**：删除 local attention
两个 warp group 在 PV GEMM 前的两处冗余 FP8 CTA barrier。相对 R4，12-shape
local p50 几何平均分别加速 **1.0225x** 和 **1.0251x**，且两个节点均为
12/12 shape 正收益。完整 NCU 的主 kernel 加速 **1.0346x**，barrier stall
降低 **7.64%**，eligible warps 增加 **9.57%**。报告和原始数据见
[`dense-fp8-sm90-r5-20260812/REPORT.md`](../profile/dense-fp8-sm90-r5-20260812/REPORT.md)。

R6--R11 继续优化完整 local attention 的 split combine 与跨 kernel 调度，最终
保留 R6、R7、R8、R11：warp-owned row combine、split output 预取流水、仅对大
launch 启用 Hopper PDL，以及 1024-row 的 8-row CTA crossover。最终 R11 相对
R5 的 local p50 几何平均在两节点分别加速 **1.07785x** 和 **1.07745x**，
peer operator 分别加速 **1.04087x** 和 **1.03951x**。R9 全局 8-row CTA 与
R10 base-2 private LSE 均已回退。完整记录见
[`dense-fp8-sm90-r6-r11-20260813/REPORT.md`](../profile/dense-fp8-sm90-r6-r11-20260813/REPORT.md)。

四节点 Kimi-K3/DSPARK 端到端验证继续保留 **R12**：dense draft 的 TP rank
拥有不同 KV heads，不能继承 MLA target 的 DCP sequence-shard 语义。R12 在
`draft_tp_context()` 内把 draft 显式隔离为 DCP1，并补齐 accepted KV relocation
的 logical-to-local DCP 映射。固定 8K/1500 workload 上，DCP8 accept length 从
1.11 恢复到 4.74，输出吞吐从 50.76 提升到 150.30 tok/s（**2.96x**），TPOT
从 128.86 降至 39.77 ms。匹配 DCP1 对照为 accept 6.17、160.04 tok/s、
39.43 ms。完整根因、排除实验与原始证据见
[`kimi-k3-dcp8-e2e-20260826/README.md`](../outputs/kimi-k3-dcp8-e2e-20260826/README.md)。

R13 继续验证“剩余差距来自 rank-local 舍入”的假设：新增 FP32 local output，
让 attention state 经过全局 LSE merge 后才做一次 BF16 cast。generic
FlashMLA NCCL operator 在 Q1/Q8 分别加速 **1.0267x/1.0447x**，但 Kimi-K3
实际 SGLang merge 路径的组件合成估算分别回退 **1.95%/2.96%**；16-case
平均 NRMSE 只改善 **0.394%**。四节点固定 8K、3 x 3 E2E 最终显示 FP32
local 的平均 TPOT **+3.70%**、output throughput **-3.38%**，按 active request
加权的 accept rate 从 72.10% 降至 69.24%（**-2.86 pp，相对 -3.97%**），
accept length **-3.28%**。它没有恢复 DSPARK acceptance，性能方向也与
component model 一致，因此 **R13 performance rejected**，仅保留 opt-in
实验路径。
完整数据见
[`dense-fp8-sm90-r13-20260827/REPORT.md`](../profile/dense-fp8-sm90-r13-20260827/REPORT.md)。

R14-P0 随后验证真正的 DCP-size 不变量：固定 1024-token global chunks，并用
固定 adjacent balanced tree 合并 FP32 O/LSE。B1/Q8/H128/Dv512/KV8192 上，
CP=1/2/4/8 output/LSE 均 **bitwise identical**；但未融合 PyTorch 原型约
7.46 ms，相对 R13 的 0.0966 ms 慢约 77x，因此只保留为 correctness oracle，
性能方案拒绝。设计与证据见
[`dcp-canonical-r14-20260827/REPORT.md`](../profile/dcp-canonical-r14-20260827/REPORT.md)。

R14-P1 已把逐 chunk Python 原型融合进 SM90 kernel：一次 launch 生成 8 个
FP32 leaf，再用固定 adjacent tree 合并。H20、B1/Q8/H128/Dv512/KV8192
三轮均值为 **0.11135 ms**，相对 BF16 regular 的 0.12753 ms 加速
**12.69%**；`tests/test_dense_fp8_dcp.py` 为 **15 passed**，重复执行
output/LSE bitwise identical。P1 operator 接受，但 distributed tree 与 E2E
尚未完成。完整数据见
[`dcp-canonical-r14p1-20260827/REPORT.md`](../profile/dcp-canonical-r14p1-20260827/REPORT.md)。

R14-P2--P4 在 K8s 预筛选中完成了真实 8 GPU distributed tree。P2 的 leaf export 在模拟
CP1/2/4/8 ownership 后逐 leaf、output、LSE 均 bitwise identical；P3 的 NCCL
P2P fixed tree 虽保持 bitwise，却以 **0.19946 ms** 相对 CP1 回退
**78.31%**。P4 改为 symmetric-memory direct peer read 和 device epoch，三轮
均值 **0.10550 ms**，相对同轮 CP1 fused 0.11166 ms 加速 **5.52%**，三轮
max abs 均为 0，因此通过 K8s operator 预筛选；按当前环境硬约束不作为正式
接受结论。完整数据见
[`dcp-canonical-r14p2-p4-20260827/REPORT.md`](../profile/dcp-canonical-r14p2-p4-20260827/REPORT.md)。

R14-P5 把 root-owned tree 改为每 rank 连续 H/8 head shard 的 exact
reduce-scatter；R14-P6 再让 local attention 直接写 caller-owned symmetric FP32
leaf，删除 leaf output 的额外 materialization。P6 已在两台独立 ECS H20 节点
`.90`/`.92` 上重新同步源码、独立编译和正式复测。B1/Q8/H128/Dv512/KV8192、
CUDA Graph、100 warm-up + 1000 iterations 的 6 次结果中，critical latency
均值为 **0.095784 ms**，同轮 CP1 fused 均值为 **0.112692 ms**，平均提升
**15.00%**；两台 ECS 各自的三轮范围分别为 **14.97%--15.04%** 和
**14.70%--15.22%**。6/6 output/LSE 均与 CP1 **bitwise equal**，两台节点的
`tests/test_dense_fp8_dcp.py` 均为 **16 passed**，因此 **R14-P6 ECS operator
accepted**。这仍不是 DSPARK speculative acceptance rate：P6 尚未接入四节点
Kimi-K3/SGLang，E2E acceptance 必须单独测试。完整 ECS 报告和原始 JSON 见
[`dcp-canonical-r14p6-ecs-20260827/REPORT.md`](../profile/dcp-canonical-r14p6-ecs-20260827/REPORT.md)。

R14-P7 进一步解决了生产接入前的布局不一致：SGLang allocator 按 global token
`i % 8` 做 token-cyclic ownership，而 P6 benchmark 假设每 rank 已有连续
1024-token chunk。P7 先把本地 paged FP8 KV stage 到 symmetric buffer，再由
8 个 peer 重建固定 global chunk，最后复用 P6 exact peer-RS。两台指定 ECS
`.90`/`.92` 的 6 次正式结果为 **0.123786 ms**，同轮 CP1 fused 为
**0.112967 ms**，代价 **+9.58%**；6/6 output/LSE bitwise、max abs 0，两节点
各 **16 passed**。但 `.92` 的现有生产 DCP8/NCCL operator p50 为
**0.099192 ms**，P7 约慢 **24.92%**，所以 P7 只通过 correctness gate，保留为
四节点 E2E 候选，尚未通过 performance/acceptance gate。P8 cooperative pack
和 P9 per-consumer read epoch 分别比 P7 smoke 慢 **1.12%/1.17%**，均已拒绝并
回退。完整数据、profile 和三份源码归档见
[`dcp-canonical-r14p7-ecs-20260827/REPORT.md`](../profile/dcp-canonical-r14p7-ecs-20260827/REPORT.md)。

R14-P10 删除 peer-RS 后的 full-H -> H/8 copy，并完成四 ECS 的 SGLang v3
接入。固定 8K/1500、16 prompts、RR=1、MC=8 的三轮 A/B 中，canonical DCP8
的 mean/median TPOT 为 **36.94/35.54 ms**，输出吞吐为
**168.82/177.38 tok/s**；同代码 DCP1 为 **38.82/38.74 ms** 和
**166.83/167.89 tok/s**。性能方向为正，但 DCP8 acceptance rate 的
active-request 加权 mean/median 为 **73.86%/78.10%**，DCP1 为
**80.09%/80.17%**。P10 相对 R13 DCP8 的 acceptance mean/median 已提高
1.76/6.87 pp，但相对 DCP1 仍差 6.23/2.07 pp，第一轮也只有 65.09%。因此
P10 是 **operator accepted / E2E candidate**，尚未最终发布接受。完整六组
JSON/JSONL、八份四节点日志、逐 DP 加权分析器和汇总 CSV 见
[`dcp-canonical-r14p10-ecs-20260827/REPORT.md`](../profile/dcp-canonical-r14p10-ecs-20260827/REPORT.md)。

R17--R22 随后完成精度闭环。实际首步 tensor 的 CP1 GPU reconstruction 证明
R14 canonical DCP8 FlashMLA output bitwise、max abs 0，剩余问题不在 SM90
merge；真正根因是 SGLang prefill 写 KV 时，Triton backend 和 MLA KV pool
各做了一次 owner filter/`loc//8`。修复前 8K 请求只有 **553/8001** 个 global
RoPE KV row 非零，R22 改为 MLA logical loc 只由 pool 映射一次后恢复为
**8001/8001**。target-only Graph 连续 **12/12 exact**，DSPARK smoke
**8/8 exact**。正式三轮 8K/1500 的 TPOT mean/median 为
**32.60/32.46 ms**，output throughput 为 **193.63/194.79 tok/s**，
active-request 加权 acceptance rate 为 **80.54%/80.69%**，accept length 为
**6.639/6.650**。相对旧 DCP8 run37，mean TPOT 改善 **11.74%**、输出吞吐
提高 **14.69%**、acceptance 提高 **6.68 pp**；相对 DCP1 control，mean TPOT
改善 **16.01%**、吞吐提高 **16.06%**，acceptance 还高 **0.45 pp**。
因此 R22 的 **精度修复与相对性能 accepted**。但 TTFT mean 为 10.01 s、TPOT
mean 为 32.60 ms，仍未达到 5 s / 20 ms 绝对 SLO；整套部署保持 SLO pending。
完整 trace、revision 表和机器可读结果见同一
[ECS 报告](../profile/dcp-canonical-r14p10-ecs-20260827/REPORT.md)。
面向背景、怀疑方向、验证链路与 A/B 收益归因的专题复盘见
[《Kimi-K3 DCP8 接受率下降问题复盘》](kimi_k3_dcp_acceptance_drop_postmortem.md)。

## R0--R11 总表

性能列沿用每一阶段的 accept/reject 主指标：R0--R2 是
`peer_merge / peer_operator`，R3--R4 只有 `local`，R5--R11 是
`local / peer_operator`。R4 和 R8 的重复 baseline 使用相邻 A/B，因此不能把
不同阶段的绝对 latency 强行连成一条曲线。完整机器可读版本与证据链接见
[`dcp-sm90-r0-r11-summary-20260813`](../profile/dcp-sm90-r0-r11-summary-20260813/REPORT.md)。

| revision | 优化范围 | 核心修改 | node-a performance | node-b performance | 相对结果 | 状态 | R11 保留 |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| R0 | peer merge | 256 threads；BF16x2、32-bit load/store | 31.713 / 60.449 us | 31.787 / 60.521 us | baseline | baseline | 否 |
| R1 | peer merge | 64 threads；`uint4` 128-bit；4 组 BF16x2；fallback | 20.097 / 47.895 us | 20.003 / 47.962 us | vs R0：**1.578x / 1.262x**；**1.589x / 1.262x** | **accepted** | **是** |
| R2 | peer merge | 32 threads；双 `uint4`；warp barrier | 未运行 | 20.378 / 48.350 us | vs R1：**0.9816x / 0.9920x** | rejected | 否 |
| R3 | local scheduler | 所有 batch 无条件 ceil `num_sm_parts` | 未运行 | 26.345 us | vs R1-local：**0.9883x** | rejected | 否 |
| R4 | local scheduler | 仅 B1 ceil；B>1 保留 floor | 25.371 us | 25.540 us | vs R1-local：**1.0170x / 1.0195x** | **accepted** | **是** |
| R5 | local mainloop | 删除 PV GEMM 前两个冗余 FP8 CTA barrier | 25.112 / 46.592 us | 25.164 / 46.675 us | vs R4-rebaseline：local **1.0225x / 1.0251x** | **accepted** | **是** |
| R6 | split combine | <=512 rows 保留旧路径；其余 warp-owned row + `float4` | 24.122 / 45.774 us | 24.235 / 45.937 us | vs R5：local **1.0411x / 1.0383x** | **accepted** | **是** |
| R7 | split combine | split 0 预取；下一 split 软件流水 | 23.769 / 45.528 us | 23.901 / 45.683 us | vs R6：local **1.0149x / 1.0140x** | **accepted** | **是** |
| R8 | producer→consumer | 仅大于 512 rows 启用 Hopper PDL | 23.361 / 44.834 us | 23.398 / 44.943 us | vs R7：local **1.0175x / 1.0215x** | **accepted** | **是** |
| R9 | combine launch | 全部 grouped launch 使用 8 rows/CTA | 23.408 / 44.851 us | 未运行 | vs R8：local **0.9980x** | rejected | 否 |
| R10 | split LSE | private LSE 使用 base-2，最终转回自然对数 | 23.369 / 44.853 us | 未运行 | vs 相邻 R8：local **0.9996x** | rejected | 否 |
| R11 | combine crossover | 仅 1024 total rows 使用 8 rows/CTA | 23.299 / 44.763 us | 23.355 / 44.901 us | vs 相邻 R8：local **1.0017x / 1.0023x** | **accepted** | **是** |

## 目标与边界

- R0--R2 优化 SM90 上 DCP size=2 的 peer attention merge；R3--R11 优化
  dense FP8 本地 attention 的调度、同步、split combine 与 producer/consumer
  overlap。不改变 DCP4/DCP8 NCCL 路径，也不改 vendored CUTLASS。
- 保留现有 epoch、system-scope acquire/release atomic、CUDA Graph 和 peer
  buffer 生命周期协议。
- 保持公开约束 `head_dim_v` 为任意正偶数：`Dv % 8 == 0` 使用 128-bit
  路径，否则回退到原 BF16x2 路径。
- 用 NCCL merge 作为数值 oracle；性能只在正确性通过后比较。

## 环境

| 项目 | node-a | node-b |
| --- | --- | --- |
| K8s Pod | `dev-sq-roleset-m88mv-decode-64f4b88598-0` | `dev-sq-roleset-m88mv-prefill-94dd4bf87-0` |
| 物理节点 | `192.168.122.78` | `192.168.1.179` |
| GPU | 8 x NVIDIA H20，测试使用 GPU 0、1 | 8 x NVIDIA H20，测试使用 GPU 0、1 |
| 架构/拓扑 | SM90，NV18 全互联 | SM90，NV18 全互联 |
| PyTorch/CUDA | 2.11.0+cu130 / 13.0 | 2.11.0+cu130 / 13.0 |
| NVCC/driver | 13.0 / 535.161.08 | 13.0 / 535.161.08 |

基础提交为 `3a0b48b8e7715ebc0e03aa09a1700868ccc97916`，实验基于其上的现有
dirty worktree。DCP2 translation unit 以 `-O3 --use_fast_math -lineinfo
--source-in-ptx -gencode arch=compute_90a,code=sm_90a` 编译；构建命令设置
`FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=4 MAX_JOBS=8`。wheel 是增量构建，
构建目录中保留了此前的其他架构 object，但本次 DCP2 object 明确重新编译为
SM90a；本报告只声明 SM90 验证。

两个 Pod 都有正在运行的 SGLang 服务，每卡已有约 68--87 GiB 显存占用。
测量期间观察到 GPU utilization 接近 0%，且 R0/R1 在同一 Pod 内相邻执行；
因此结果适合作为相对 A/B 和复验数据，不作为独占节点 golden baseline。

R6--R11 使用同两台物理节点上重新创建的 Pod：
`dev-sq-roleset-4bwl5-decode-858c8b85bf-0` 和
`dev-sq-roleset-4bwl5-prefill-d7c77b4b-0`。GPU、CUDA、driver 和测量协议不变。

## 测量方法

性能矩阵固定为 CP=2、B={1,8}、Q={1,4,8}、H=128、Dv=512、
KV={128,4096}、full-Q，共 12 个 shape。每个 case 使用：

- CUDA Graph；
- 100 次 warm-up；
- 100 个样本，每个样本 10 次调用；
- 对两个 rank 取 critical-path latency；
- 同时记录 p50、p99、CV；表中汇总值为 12 个 shape 的 p50 几何平均。

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 benchmark/bench_dcp_suite.py \
  --profile smoke --cuda-graph --output /tmp/dcp2-REV-perf-NODE.json
```

Nsight Systems 诊断另用 B8/Q8/H128/Dv512/KV128、eager launch、5 次
warm-up、10 x 10 次采样，以免 CUDA Graph 折叠 kernel instance。Nsight
Compute 2025.3.1 能连接两个 rank，但对包含跨 GPU epoch 自旋的 kernel 做
replay 时在 0% 以 `UnknownError` 退出；因此没有伪造 NCU metric，瓶颈判断改用
Nsight Systems、ptxas resource usage 和 SASS 访存指令交叉验证。

## Revision 迭代表

下表 R0--R2 的 performance 数字均为 `peer_merge p50 geomean / peer_operator p50
geomean`；括号内是相对同节点 R0 的加速。

| revision_id | commit_info | 修改点 | node-a performance | node-b performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R0 | base `3a0b48b8`; wheel `c5c5e9d3...d300` | 256 threads；每线程 BF16x2，32-bit load/store | 31.713 / 60.449 us (1.000x / 1.000x) | 31.787 / 60.521 us (1.000x / 1.000x) | baseline |
| R1 | kernel diff `71e0a756...c93`; wheel `076786cf...98f9` | 64 threads；每线程 `uint4`/8 BF16；4 组 BF16x2 展开；非 8 倍数回退 | 20.097 / 47.895 us (**1.578x / 1.262x**) | 20.003 / 47.962 us (**1.589x / 1.262x**) | **accepted** |
| R2 | wheel `4b45c945...b3cb` | 32 threads；block barrier 改 warp barrier；每线程两个 `uint4` | 未运行 | 20.378 / 48.350 us (1.560x / 1.252x) | rejected：比 R1 慢 1.87% / 0.81% |

### Dense FP8 local attention（R3--R4）

下表 performance 为 12 个 local shape 的 p50 几何平均；括号内为相对同节点
R1 local baseline 的延迟比例和加速。

| revision_id | commit_info | 修改点 | node-a performance | node-b performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R1 | base `3a0b48b8`; binary `eff5e2f1...51b9` | 原 floor scheduler | 25.803 us (1.000 / 1.000x) | 26.038 us (1.000 / 1.000x) | baseline |
| R3 | binary `1902ec3a...27159` | 无条件 ceil `num_sm_parts` | 未运行 | 26.345 us (1.01182 / 0.988x) | rejected：B8/Q8/KV4096 回退 31.4% |
| R4 | source diff `3eda44c8...b47e`; binary `4986ce0f...2b4d` | 仅 B1 ceil；B>1 保持 floor | 25.371 us (0.98325 / **1.017x**) | 25.540 us (0.98089 / **1.019x**) | **accepted** |

机器可读表见
[`iterations.csv`](../profile/dense-fp8-sm90-r3-20260810/iterations.csv)。

### Dense FP8 local attention（R5）

R5 使用重新创建的 Pod；下表 performance 为 12 个 local shape 的 p50 几何
平均，括号内为相对同节点 R4 的加速。R5-VT 是先尝试、后拒绝的单变量实验；
accepted R5 只包含 barrier 修改。

| revision_id | commit_info | 修改点 | node-decode performance | node-prefill performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R4 | base `3a0b48b8`; binary `27ddafad...cf9` | R4 scheduler baseline | 25.677 us (1.000x) | 25.796 us (1.000x) | baseline |
| R5-VT | rejected patch `45947204...e43a` | conflict-free FP8 V transpose + epilogue unpermute | 26.828 us (**0.9571x**) | 未运行 | rejected：local 回退 4.29% |
| R5 | delta `15998b02...70f1`; binary `b3a3818e...d401` | 删除 PV GEMM 前两个冗余 FP8 CTA barrier | 25.112 us (**1.0225x**) | 25.164 us (**1.0251x**) | **accepted** |

机器可读表、NCU 指标、原始 JSON、两份 NCU report 和 rejected patch 见
[`dense-fp8-sm90-r5-20260812`](../profile/dense-fp8-sm90-r5-20260812/REPORT.md)。

### Dense FP8 local attention（R6--R11）

下表 performance 为 12 个 local shape 的 p50 几何平均；括号内为相对前一个
accepted revision 的加速。R10/R11 使用紧邻 A/B 的 R8 baseline 计算括号值。

| revision_id | commit_info | 修改点 | node-decode performance | node-prefill performance | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| R5 | binary `b3a3818e...d401` | R5 barrier baseline | 25.1123 us | 25.1641 us | baseline |
| R6 | binary `5c6894de...892e` | <=512 rows 保留旧 combine；其余 warp-owned row + `float4` load | 24.1220 us (**1.04105x**) | 24.2350 us (**1.03833x**) | **accepted** |
| R7 | binary `a278e07f...0969` | 预取 split 0，并流水加载下一 split 的 `oaccum` | 23.7689 us (**1.01486x**) | 23.9006 us (**1.01399x**) | **accepted** |
| R8 | binary `5ccffc06...bf4e` | 仅 `total_rows > 512` 启用 Hopper PDL | 23.3606 us (**1.01748x**) | 23.3975 us (**1.02150x**) | **accepted** |
| R9 | binary `e7588eb4...d084` | 所有 grouped launch 使用 8 rows/CTA | 23.4079 us (**0.99798x**) | 未运行 | rejected |
| R10 | binary `83a53402...28f` | private split LSE 保持 base-2 | 23.3685 us (**0.99957x** repeat A/B) | 未运行 | rejected |
| R11 | header `6eac2fed...b05` | 仅 1024 total rows 使用 8 rows/CTA | 23.2985 us (**1.00169x**) | 23.3553 us (**1.00230x**) | **accepted** |

R11 的目标 1024-row subset 在两节点分别复现 **+0.550%**、**+0.533%**；
NCU combine kernel 为 66 registers、0 spill，7.360 -> 7.328 us。最终两节点
各 21/21 回归通过，CP={1,2,4,8}、Q={1,2,4,8} 的扩展精度矩阵 16/16
通过。完整候选表、原始 JSON 和 NCU report 见
[`dense-fp8-sm90-r6-r11-20260813`](../profile/dense-fp8-sm90-r6-r11-20260813/REPORT.md)。

### 完整 Kimi-K3/DSPARK 集成（R12）

本阶段的 performance 为四节点 32 x H20、TP32/DP4/EP32、8K 输入、1500
输出、16 prompts、RR=1、MC=8 的端到端结果。`commit_info` 如实记录为本地
SGLang diff；本轮未创建提交。

| revision_id | commit_info | 修改点 | performance | 结论 |
| --- | --- | --- | --- | --- |
| R11-E2E | run18；SGLang deployed overlay before draft fix | dense DSPARK draft 错误继承 target DCP8 | accept 1.11；TPOT 128.86 ms；output 50.76 tok/s | precision blocker |
| R12 | uncommitted SGLang diff；`spec_utils.py` SHA256 `210d7159...9b9ce`；`memory_pool.py` SHA256 `809c881c...47bc`；archive `2aba72b8...a69ff` | draft TP context 强制 DCP1；accepted MLA KV owner-filter + logical-to-local relocation | accept **4.74**；TPOT **39.77 ms**；output **150.30 tok/s**；vs R11-E2E **2.96x** | **accepted** |
| R12-control | run34；同一代码，DCP1 | 匹配 TP32/DP4 对照 | accept 6.17；TPOT 39.43 ms；output 160.04 tok/s | fixed DCP8 output throughput 为对照的 93.91% |

完整机器可读版本见
[`iterations.csv`](../profile/dcp2-sm90-20260808/iterations.csv)。

### FP32 rank-local state（R13）

本阶段是 8 x H20 单节点真实 DCP8/NCCL 算子 A/B。配置为 B8/H128/Dv512/
KV8192，CUDA Graph，100 warm-up、100 samples、每 sample 10 calls。
`commit_info` 如实记录为本地未提交 diff。

| revision_id | commit_info | 修改点 | Q1 performance | Q8 performance | 精度 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| R12-micro | base `3a0b48b8`; BF16 local output | 每 rank 在 merge 前舍入 BF16 | local/merge/operator 0.025430/0.072990/0.099397 ms | 0.104291/0.186590/0.289936 ms | 16/16 PASS；mean NRMSE 0.00409152 | baseline |
| R13 | binary `b86346f7...3bab`; dense API `cf7c10ea...e5c0`; epilogue `1778358d...e5c0` | local output FP32；全局 merge 后单次 BF16 cast | generic 0.026547/0.069368/**0.096811 ms (1.0267x)** | generic 0.107011/0.167973/**0.277533 ms (1.0447x)** | 16/16 PASS；mean NRMSE 0.00407540（**-0.394%**） | **performance rejected after E2E** |

R13 local kernel 因 32-bit 写回慢 2.61%--4.39%。generic merge 避免 BF16
扩展后 operator 为正收益；SGLang Triton merge 计入最终 cast 后却慢
0.65%--3.28%，合成生产 attention 约回退 1.95%--2.96%。两条数据不可混用。
完整报告和 JSON 见
[`dense-fp8-sm90-r13-20260827`](../profile/dense-fp8-sm90-r13-20260827/REPORT.md)。

四节点 E2E 使用同一 R13 binary/overlay，只切换环境变量；每个 rep 前均清空
radix/mamba cache。`commit_info` 仍是未提交本地 diff，部署归档为 FlashMLA
`aa1d46e0...cc93`、SGLang `e1fe2122...aaf`。

| revision_id | commit_info | 修改点 | 三轮 performance（mean / median） | accept length | 结论 |
| --- | --- | --- | --- | --- | --- |
| R13-control-run35 | 同一 R13 binary；`SGLANG_DCP_FP32_LOCAL_OUTPUT=0` | BF16 rank-local state | TPOT 37.25 / 37.94 ms；output 166.95 / 163.11 tok/s | rate 72.10% / 71.23%；weighted length 6.047 / 5.983 | baseline |
| R13-run36 | 同一 R13 binary；`SGLANG_DCP_FP32_LOCAL_OUTPUT=1` | FP32 state；merge 后 cast | TPOT 38.63 / 38.45 ms；output 161.31 / 156.40 tok/s | rate 69.24% / 70.66%；weighted length 5.849 / 5.945 | **rejected** |

FP32 local 相对 BF16 local 的均值变化为 TTFT -1.84%、TPOT **+3.70%**、
E2E **+2.86%**、output throughput **-3.38%**。三轮系统噪声较大，因此不把
约 3% 差异冒充高置信度因果效应；但 accept 指标没有一致正收益，已经足够拒绝
“rank-local BF16 舍入是剩余 acceptance gap 主因”的假设。六组 JSON、四节点
日志与机器可读 CSV 见上述 R13 报告。

### Canonical DCP（R14-P0--P10）

P0--P4 的数字来自 K8s 原型筛选，只保留为历史证据；P6 的数字来自指定 ECS，
P7 也在指定 ECS 完成布局兼容性和正式复测。P8/P9 只在 ECS `.92` 做单变量
smoke 筛选并拒绝。P10 先在 `.92` 做 consumer-ready 单变量筛选，并已完成
正式四节点 E2E。

| revision_id | commit_info | 修改点 | correctness | performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R13 | binary `b86346f7...3bab` | FP32 local state + NCCL merge | 精度通过，但不同 CP size 仍有不同归约图 | B1/Q8/KV8192 operator 0.0966 ms | baseline |
| R14-P0 | uncommitted `dcp_eval/canonical.py` oracle | fixed 1024-token global chunks + fixed adjacent tree | CP=1/2/4/8 output/LSE **bitwise identical**；p99 2.33e-9；NRMSE 5.06e-7 | 7.46 ms，约 **77x slower** | correctness proof；**performance rejected** |
| R14-P1 | uncommitted；kernel `bfa7ac98...6953`；metadata `7f13737c...9b63` | fused 8 leaf + fixed local tree；FP32 output | dense FP8 tests **15 passed**；repeat bitwise identical | B1/Q8/KV8192 **0.11135 ms**；vs BF16 **-12.69%** | **operator accepted**；进入 distributed tree |
| R14-P2 | uncommitted；leaf ABI `f027e16b...bf2f` | export fixed FP32 leaves；restore global chunk order | CP1/2/4/8 leaf/output/LSE **bitwise identical**；dense FP8 **16 passed** | leaf-only **0.08287 ms** | **accepted** |
| R14-P3 | uncommitted；pair/tree bench `284507d3...1835` | NCCL P2P 三层 fixed tree | 3/3 output/LSE bitwise；max abs 0 | **0.19946 ms**；vs CP1 **+78.31%** | **rejected** |
| R14-P4 | uncommitted；peer kernel `dbe90312...0bde` | symmetric memory + per-row epoch + direct NVLink read | 3/3 output/LSE bitwise；max abs 0 | **0.10550 ms**；CP1 0.11166 ms；**-5.52%** | K8s 预筛选通过；非 ECS 正式接受 |
| R14-P5 | uncommitted；peer-RS kernel `aa3a4e3a...60a` | head-sharded exact reduce-scatter；每 rank ownership 为连续 H/8 | K8s 预筛选 bitwise | 已由 P6 覆盖，不单独生成 ECS 性能结论 | subsumed by P6 |
| R14-P6 | uncommitted；base `3a0b48b8`；archive `1739d613...b9a`；bench `143b203f...dd7` | attention 直接写 symmetric FP32 leaf；peer-RS fixed tree；CUDA Graph | 两 ECS 6/6 output/LSE bitwise；两节点各 **16 passed** | **0.095784 ms**；CP1 0.112692 ms；平均 **-15.00%** | **ECS operator accepted**；DSPARK acceptance 待 E2E |
| R14-P7 | uncommitted；archive `b24a2d12...90668` | token-cyclic paged KV -> symmetric staging -> fixed global chunk；复用 P6 exact peer-RS | 两 ECS 6/6 output/LSE bitwise；max abs 0；两节点各 **16 passed** | **0.123786 ms**；CP1 0.112967 ms；**+9.58%**；vs current DCP8 约 **+24.92%** | **correctness accepted；E2E candidate** |
| R14-P8 | rejected archive `285c7014...928` | cooperative persistent kernel 融合 6 个 pack/sync stage | smoke bitwise；max abs 0 | **0.126314 ms**；vs P7 smoke **+1.12%** | **rejected** |
| R14-P9 | rejected archive `11b9b7c9...9cb` | per-consumer read epoch 替代 aggregated remote read-count atomic | smoke bitwise；max abs 0 | **0.126375 ms**；vs P7 smoke **+1.17%** | **rejected** |
| R14-P10 | uncommitted；archive `55ae0bb8...20fc` | peer-RS 直接写 compact BQHD/BHQ shard，删除下游 full-H -> H/8 copy | 四 ECS 各自的 8-rank workspace 全部通过，覆盖 KV8192/9692；output/LSE bitwise，max abs 0 | C1024/KV8192 **0.125432 ms**，vs P7+copy 0.128682 ms **-2.53%**；C1216/KV8192 **0.134721 ms**，vs 0.137522 ms **-2.04%**；C1216/KV9500 **0.134879 ms**，vs 0.137453 ms **-1.87%** | **consumer-ready operator accepted** |
| R14-P10-E2E-run37 | wheel `ea3f5027...9c57`；SGLang overlay `df34f3a5...212f` | TP32/DCP8/DP4/EP32；C1216；DSPARK；三轮 8K/1500 | 48/48 请求成功；active-request 加权 acceptance 为 65.09%/78.37%/78.10%，mean/median 73.86%/78.10% | TPOT mean/median **36.94/35.54 ms**；output **168.82/177.38 tok/s**；DCP1 为 38.82/38.74 ms、166.83/167.89 tok/s | **performance positive；E2E candidate** |

R14 的生产实现不能 all-gather 全部 leaves；P5/P6 已完成 head-sharded direct-peer
fixed tree；P7 已补齐 SGLang token-cyclic KV layout adapter，P10 则把最终结果直接
落到每 rank 的 compact H/8 shard。1024-token chunk 的总容量只有 8192，不能覆盖
8K prompt 后的 decode；生产候选改用 1216-token chunk，总容量 9728，可覆盖
8192+1500，并保留 36-token 余量。四台指定 ECS 的 8K 性能和 DSPARK
acceptance 已完成：P10 相对 R13 DCP8 的性能和 acceptance 都有改善，但相对
DCP1 仍有 6.23 pp mean gap，且第一轮低值构成新的 blocker。
P10 原始 JSON、容量实验、8-GPU 回归与恢复归档见
[`dcp-canonical-r14p10-ecs-20260827`](../profile/dcp-canonical-r14p10-ecs-20260827/REPORT.md)。
完整设计见
[`dcp_canonical_merge.md`](dcp_canonical_merge.md)。

### Accepted-KV 跨 owner 搬运诊断（R15）

R15 定位到 R12 的 SGLang accepted-KV relocation 只处理
`src_owner == tgt_owner == dcp_rank`。对 `topk>1` speculative tree compaction，
accepted source slot 压缩到 committed target slot 时 owner 可以变化；R15 把
所有 MLA layer source rows 合并成 raw-byte buffer，DCP group 一次 all-gather
后由 destination owner 写回，以保持 FP8 bitwise。随后 caller trace 证明当前
DSPARK 使用 `speculative_eagle_topk=1`，该 relocation 调用点根本不执行；所以
它是有效的 topk>1 低层修复，但不是当前 K3 精度漂移的生产根因。

| revision_id | commit_info | 修改点 | correctness / acceptance | performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R15-unit | uncommitted SGLang file `8b4157bb...d86`；patch `35029b9a...ad1` | cross-owner raw-FP8 all-gather relocation | 8 ranks x 3 layers，8 cross-owner + 8 same-owner pairs，failure 0 | 未计时 | low-level accepted |
| R15-smoke9 | 同文件；canonical P10；8K deterministic | 2 cycles x DP4 | **7/8 exact**；accept rate mean/median **88.94%/100%**，min 11.55% | 非正式 smoke | variance；不能归因给 R15 |
| R15-smoke10 | 同服务 post-flush | 3 cycles x DP4 | **10/12 exact**；accept rate mean/median **87.07%/100%**，min 15.57% | 非正式 smoke | rejected |
| R15-ring32-bs12 | diagnostic script `7e8132cb...25b` | ring32；Graph max-bs 12 | **8/12 exact**；失败 rate 12.70%--20.68% | 非正式 smoke | ring-length 假设 rejected |
| R15-sync | file `78a0e3a1...bbe3` | relocation 后显式 CUDA sync | **7/12 exact**；失败 rate 14.63%--43.54% | 非正式 smoke | stream-visibility 假设 rejected |
| R15-locdebug | file `d0cede03...7ffe` | relocation caller trace | **7/8 exact**；四节点日志 0 trace | 非正式 smoke | 证明 topk1 不调用 R15 |
| target-only-DCP8 | 关闭 DSPARK；canonical Graph | 8K/128，1 cycle x DP4 | **0/4 exact；4 hashes**；first bad 19/105/4/13 | 29.27--32 s/request | target DCP8 根因 |

两轮合计 **17/20 exact（85%）**、请求级 acceptance mean/median
**87.82%/100%**，但 locdebug 证明该代码没有执行，数字只反映运行方差。关闭
DSPARK 后 target-only DCP8 仍 0/4 exact，剩余根因在 target KV/attention 集成。
本轮没有运行 R15 正式 8K/1500 性能，不能填写 R15 TPOT/throughput。ring32
最初的 shape mismatch 是 backend 有效 max-bs=14 却捕获 bs16；降为 bs12 后服务
就绪但精度仍失败，并非 ring=16 被静态硬编码。完整 JSON、四节点日志和 patch 见
[`dcp-canonical-r14p10-ecs-20260827`](../profile/dcp-canonical-r14p10-ecs-20260827/REPORT.md)。

### Kimi hybrid DCP allocator 原子页（R16）

普通 MLA DCP allocator 已用 `size*dcp_size`、`page_size*dcp_size` 保持 logical
page 原子性；Kimi-K3 hybrid SWA allocator 却仍按物理 64-token page 分配，同时
KV writer 执行 `loc//8`、FlashMLA block table 按 `PAGED_SIZE=512` 采样。初始连续
分配可能偶然成立，radix/repeated request 产生页复用或碎片后，一个 512-token
logical page 会由八个独立小页拼接，block table 仍错误假设其 local rows 连续。

| revision_id | commit_info | 修改点 | correctness / acceptance | performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R16-unit | uncommitted SGLang；patch `f94bf55d...3ac0` | hybrid full/SWA allocator 的 size/page 扩展为 DCP logical geometry；resize 同步保持；新增 regression | rank0 容器 **2 passed**；四 ECS SHA 一致 | 未计时 | 独立修复 accepted |
| R16-smoke17 | 同 patch；canonical Graph；target-only 8K/128 | 四 DP 单轮 | **0/4 exact；4 hashes**；first bad 43/24/18/82 | 28.96--31.23 s/request | 当前 K3 路径 rejected |

生产启动日志为 `hybrid_ssm=True, hybrid_swa=False`，证明当前 K3 full-attention
KV 没有走 R16 修改的 SWA allocator；补丁保留为独立修复，但对当前 blocker
rejected。关闭 CUDA Graph 的 target-only 对照还暴露出 canonical workspace 仅在
`init_cuda_graph_state()` 分配，首个 eager decode 会直接报 workspace 未初始化。
这是独立生命周期缺陷，顺延到 R17 解耦 workspace 初始化并完成 eager/Graph 对照。

### Prefill KV 单次 DCP 映射（R17--R22）

| revision_id | commit_info | 修改点 | correctness / acceptance | performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R17 | uncommitted SGLang | canonical workspace 从 Graph-only 初始化解耦 | eager 可生成；跨 DP 仍不一致 | smoke only | lifecycle accepted；blocker rejected |
| R18 | smoke20；no-Q-rep | 关闭 query projection replication | **0/4 exact；4 hashes** | 29.60--31.43 s/request | rejected |
| R19 | eager actual-tensor trace | DCP8 实际 Q/KV/table 在 GPU 重建 CP1 | canonical DCP8 output vs CP1 **bitwise；max abs 0** | diagnostic | 排除 SM90 merge 偏差 |
| R20 | DCP1 actual-tensor trace | DCP8/DCP1 进入 attention 前逐 tensor A/B | DCP1 4/4 exact；修复前 KV cosine 0.266、relL2 0.964 | diagnostic | pre-attention KV blocker |
| R21 | default-Q-rep trace | 统计 logical/local KV 非零覆盖 | Q cosine 0.999952；KV 仅 **553/8001** nonzero | diagnostic | 双重 owner/filter/`loc//8` 根因 |
| R22-unit | local source `789d7912...e7dd`；test `d58aa686...8660` | MLA logical loc 只由 KV pool 映射一次；MHA 不变 | rank0 容器 **2 passed** | CPU unit | accepted |
| R22-Graph | ECS active `0af32605...1bca` | 修复 prefill KV 写入后 target-only 8K/128 | **12/12 exact；1 hash**；KV **8001/8001** nonzero | 28.75--29.20 s/request | correctness accepted |
| R22-run39 | 同 active source；R14-P10 wheel `ea3f5027...9c57` | TP32/DCP8/DP4/EP32；DSPARK；三轮 8K/1500 | acceptance **80.54%/80.69% mean/median**；length **6.639/6.650** | TPOT **32.60/32.46 ms**；output **193.63/194.79 tok/s** | **fix accepted；SLO pending** |

R22 与 run37 的正式差值为：mean/median TPOT **-11.74%/-8.67%**，output
throughput **+14.69%/+9.82%**，acceptance **+6.68/+2.59 pp**。与 DCP1 run38
相比，mean/median TPOT **-16.01%/-16.21%**，output throughput
**+16.06%/+16.02%**，acceptance **+0.45/+0.52 pp**。修改属于 SGLang
host/prefill integration；SM90 R14-P10 kernel 和 wheel 没有变化。完整诊断、
三轮 TTFT 为 9.73/9.86/10.44 s、TPOT 为 33.49/32.46/31.86 ms，均未达到
5 s / 20 ms 绝对 SLO。三轮逐项表、on-node acceptance 聚合与 JSON 见
[`dcp-canonical-r14p10-ecs-20260827`](../profile/dcp-canonical-r14p10-ecs-20260827/REPORT.md)。

### 关键 shape

| 节点 | revision | B1/Q1/KV128 peer merge p50 | B8/Q8/KV4096 peer merge p50 | B8/Q8/KV4096 peer operator p50 |
| --- | --- | ---: | ---: | ---: |
| node-a | R0 | 13.117 us | 120.432 us | 324.589 us |
| node-a | R1 | 12.653 us | 46.024 us | 249.814 us |
| node-b | R0 | 13.082 us | 120.240 us | 325.491 us |
| node-b | R1 | 12.378 us | 46.150 us | 251.269 us |
| node-b | R2 | 13.491 us | 45.226 us | 250.333 us |

R2 的 B1/Q1/KV128 `peer_merge` 相对 R1 回退 9.00%，KV4096 回退
7.80%。自动回归门禁报告 4 个失败项，因此没有用“大 shape 更快”覆盖小 shape
回退。

## Profile 证据与修改点

### R0

- ptxas：27 registers、1 barrier、0 spill、0 local memory。
- `cuobjdump` resource usage：REG=27、STACK=0、LOCAL=0。
- B8/Q8 eager Nsight Systems：merge kernel 212 instances，中位数
  116.081 us；prepare kernel 中位数 2.752 us。
- merge 占绝对热点，prepare launch 不是本轮首要目标。

### R1

- 把 prepare 和 merge 的 block size 拆成 `kPrepareThreads=256` 与
  `kMergeThreads=64`。
- 对齐且 `Dv % 8 == 0` 时，用 16-byte aligned union 一次读取/写入 8 个
  BF16；每个向量展开 4 次 BF16x2 -> FP32 merge -> BF16x2。
- SASS 确认生成 2 条 `LDG.E.128` 和 1 条 `STG.E.128`，而不是被编译器拆成
  窄访存。
- ptxas：31 registers、1 barrier、0 spill、0 local memory。
- B8/Q8 eager Nsight Systems：merge kernel 212 instances，中位数
  41.409 us；prepare kernel仍为 2.752 us。merge kernel 加速 **2.803x**。
- `Dv % 8 != 0` 保留 BF16x2 fallback，并新增 Dv=510 的两卡 NCCL oracle
  比较。

### R2（拒绝）

- 进一步缩成 32 threads，并用 `__syncwarp()` 去掉 block barrier。
- ptxas：32 registers、0 barrier、0 spill。
- 大 shape 约快 2%，但单 warp 的串行向量迭代使小 shape 调度/访存延迟变差；
  综合门禁不通过，最终源码恢复 R1。

## 正确性与回归结果

基线验证：

- `pytest -q tests/test_dcp_eval_oracles.py tests/test_dense_fp8_dcp.py`：
  **21 passed**。
- `benchmark/eval_dcp_accuracy.py --profile smoke`：CP={1,2,4,8}、
  Q={1,2,4,8} 共 **16/16 PASS**；输出 p99 最大约 0.002393，LSE 最大误差
  约 3.99e-6。
- 两卡 `tests/test_dcp_nccl_merge.py tests/test_dcp2_peer_merge.py`：两个 rank
  均 **10 passed**。
- 八卡 `tests/test_dcp_nccl_merge.py`：8 个 rank 均 **4 passed**。

R1 验证：

- node-b 两卡 `tests/test_dcp2_peer_merge.py`：两个 rank 均 **6 passed**。
- node-a 两卡 `tests/test_dcp_nccl_merge.py tests/test_dcp2_peer_merge.py`：两个
  rank 均 **10 passed**。
- 加入 Dv=510 fallback case 后，node-a 两卡
  `tests/test_dcp2_peer_merge.py`：两个 rank 均 **7 passed**。
- 两节点各自完成完整 12-shape R0/R1 CUDA Graph A/B，R1 peer mode 的 CV
  均低于 2%。
- `compare_dcp_results.py` 对 node-b R0/R1 的全部 60 个 mode-shape 记录：
  `p50 geomean ratio=0.86955; failures=0`。

## 数据与复现材料

- node-a：[R0 JSON](../profile/dcp2-sm90-20260808/reports/node-a/r0.json)、
  [R1 JSON](../profile/dcp2-sm90-20260808/reports/node-a/r1.json)
- node-b：[R0 JSON](../profile/dcp2-sm90-20260808/reports/node-b/r0.json)、
  [R1 JSON](../profile/dcp2-sm90-20260808/reports/node-b/r1.json)、
  [R2 JSON](../profile/dcp2-sm90-20260808/reports/node-b/r2.json)
- Nsight Systems：
  [R0 report](../profile/dcp2-sm90-20260808/reports/node-b/r0.nsys-rep)、
  [R1 report](../profile/dcp2-sm90-20260808/reports/node-b/r1.nsys-rep)

本轮方法选择受 KernelAgent `ka-kernel-opt` 的单变量 accept/reject 流程、
`ncu-report-skill` 的可审计 profile 规范，以及 KernelWiki/FlashInfer 的 128-bit
MergeState 向量化模式影响。最终结论仍以本仓库两台真实 H20 节点的正确性和 A/B
数据为准。
