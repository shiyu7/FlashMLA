# SM90 dense FP8 DCP R13：延迟 BF16 舍入

## 结论

R13 增加 FP32 rank-local 输出路径：attention 累加器不再在每个 DCP rank
先舍入为 BF16，而是保持 FP32 经过全局 LSE correction 和 NCCL
reduce-scatter，最后在原 attention ABI 边界只 cast 一次 BF16。

FlashMLA 自带的 generic NCCL 路径显示正收益：DCP8 完整 operator 在
B8/H128/KV8192 上，Q1 从 0.099397 ms 降至 0.096811 ms（**1.0267x**），
Q8 从 0.289936 ms 降至 0.277533 ms（**1.0447x**）。但 Kimi-K3 实际使用
SGLang Triton correction/reduce-scatter；计入 merge 后的最终 BF16 cast，FP32
state 在 rows=8/64 上分别慢 0.65%/3.28%。与 local kernel 合成估算，生产
attention operator 在 Q1/Q8 约回退 **1.95%/2.96%**。generic microbenchmark
不能替代生产路径结论。

精度改善真实但很小：16-case smoke 矩阵的平均 output NRMSE 下降
**0.394%**，p99 绝对误差下降 **0.372%**；CP8/Q8 NRMSE 下降 **0.494%**。
因此 R13 消除了一处不必要的 rank-local 舍入，但收益很小且生产路径存在约
2%--3% 性能成本，不足以单独解释 R12 端到端 DCP8 与 DCP1 的 DSPARK
accept-length 差距。

四节点 32 x H20 的固定 8K/1500、3 x 3 样本 E2E 已完成。FP32 local 相对
BF16 local 的均值为：TPOT **+3.70%**、E2E **+2.86%**、output throughput
**-3.38%**、按 active request 加权的 accept rate 从 72.10% 降至 69.24%
（**-2.86 pp，相对 -3.97%**），accept length **-3.28%**。中位数方向也没有
稳定精度收益。最终状态为 `performance-rejected`；实现作为 opt-in 实验路径
保留，不作为生产默认值。

## 修改点

- 新增 `dense_decode_fp8_fwd_fp32` C++/Python API 和
  `flash_mla_dense_fp8_fp32` 公共包装。
- 实例化 `run_mha_fwd_splitkv_mla<float_e4m3_t, float, 576>`。
- FP32 epilogue 使用 split-accumulator 已有的 128-bit generic copy atom；
  16-bit BF16 路径继续使用 Hopper `STSM_N`，不修改 vendored CUTLASS。
- SGLang 用 `SGLANG_DCP_FP32_LOCAL_OUTPUT=1` 打开该路径，并在全局 DCP merge
  之后恢复到原 BF16 attention ABI。
- 精度与性能脚本新增 `--local-output-dtype {bf16,fp32}`，可做同 shape A/B。

## 环境与构建

| 项目 | 值 |
| --- | --- |
| Pod | `dev-sq-roleset-mknrd-decode-5678dcfc86-0` |
| GPU | 8 x NVIDIA H20，SM90 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| Driver | 535.161.08 |
| 构建 | `FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=8 MAX_JOBS=32 python3 setup.py build_ext --inplace` |
| base | `3a0b48b8e7715ebc0e03aa09a1700868ccc97916` 上的现有 dirty worktree |
| binary | SHA256 `b86346f7ef36f94bc8e849d8103943a0612efc9cd2967d2843a13cc198313bab` |

四节点 E2E 使用 `.92/.90/.91/.93` 四台 ECS、每台 8 x H20，目标镜像内
PyTorch 2.11.0+cu130 / CUDA 13.0。源码归档 SHA256 为
`aa1d46e0...cc93`，SGLang overlay 为 `e1fe2122...aaf`。四台均在目标镜像
内重编译成功；云端再次验证单卡 14/14、单节点 8-GPU 每 rank 4/4 通过。

调试 Pod 复用了 FlashInfer 自带的 CUTLASS 4.5.0；本地 Git submodule 是
CUTLASS 4.3.0。R13 没有修改 CUTLASS，正式四节点验证仍应在目标镜像内重新
编译。ptxas 对 FP32 主 kernel 报告 251 registers、12 barriers、0 spill，
与同 translation unit 的 BF16 主 kernel 资源级别一致。

## 正确性与精度

| 验证 | 结果 |
| --- | --- |
| `pytest -q tests/test_dense_fp8_dcp.py` | **14 passed** |
| 8 卡 `tests/test_dense_fp8_dcp_distributed.py` | 每 rank **4 passed** |
| BF16 accuracy smoke，CP={1,2,4,8}、Q={1,2,4,8} | **16/16 PASS** |
| FP32 accuracy smoke，同矩阵 | **16/16 PASS** |

| 指标（16 case 平均） | BF16 local | FP32 local | 变化 |
| --- | ---: | ---: | ---: |
| output abs p50 | 8.22684e-5 | 8.22133e-5 | -0.067% |
| output abs p99 | 1.08275e-3 | 1.07871e-3 | **-0.372%** |
| output abs max | 2.74309e-3 | 2.72779e-3 | -0.558% |
| output NRMSE | 4.09152e-3 | 4.07540e-3 | **-0.394%** |
| LSE abs max | 3.74019e-6 | 3.74019e-6 | 0 |

“FP32 每个 case 的最大误差都必须更小”不是正确门禁：BF16 舍入偶尔会碰巧
更靠近 FP64 reference。正确性测试验证两条路径均在容差内、LSE 完全一致、
FP32 转 BF16 精确复现原 rank-local 输出；收益判断使用矩阵级分位数和 NRMSE。

## DCP8 性能

配置：B8/H128/Dv512/KV8192，CUDA Graph，100 warm-up，100 samples，
每 sample 10 calls，8 个真实 NCCL rank；表中为 critical-path p50。

| Q | 路径 | BF16 local | FP32 local | BF16 / FP32 |
| ---: | --- | ---: | ---: | ---: |
| 1 | local | 0.025430 ms | 0.026547 ms | 0.9579x |
| 1 | NCCL merge | 0.072990 ms | 0.069368 ms | **1.0522x** |
| 1 | full operator | 0.099397 ms | 0.096811 ms | **1.0267x** |
| 8 | local | 0.104291 ms | 0.107011 ms | 0.9746x |
| 8 | NCCL merge | 0.186590 ms | 0.167973 ms | **1.1108x** |
| 8 | full operator | 0.289936 ms | 0.277533 ms | **1.0447x** |

上表只覆盖 `flash_mla.nccl_merge_attention`。Kimi-K3 的实际 SGLang merge
使用 base-2 Triton correction + reduce-scatter；下面同样是 8 个真实 rank、
CUDA Graph、100 warm-up、100 x 10 calls，并包含 merge 后最终 BF16 cast。

| rows（B*Q） | BF16 local state | FP32 local state | FP32 latency 变化 |
| ---: | ---: | ---: | ---: |
| 8 | 0.047699 ms | 0.048008 ms | +0.65% |
| 64 | 0.110912 ms | 0.114552 ms | +3.28% |

将该 merge 与上表 local kernel 相加，Q1 为 0.073130 -> 0.074555 ms
（**+1.95%**），Q8 为 0.215203 -> 0.221563 ms（**+2.96%**）。这是组件
合成估算，不冒充完整模型 E2E；最终取舍仍由四节点固定 workload 决定。

## 四节点 Kimi-K3/DSPARK E2E

配置保持 R12 run32 不变：TP32/DP4/EP32/DCP8、DSPARK block 7、ReplaySSM、
Marlin、FP8 KV，8K 输入、1500 输出、16 prompts、request rate 1、最大并发
8。run35/run36 使用同一 R13 binary 与 SGLang overlay，只切换
`SGLANG_DCP_FP32_LOCAL_OUTPUT=0/1`。每个 rep 前调用 `/flush_cache`，后端确认
HybridReqToTokenPool 重置成功。

| 路径 | rep | TTFT | TPOT | E2E | output tok/s | accept rate / weighted accept len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 local，run35 | 1 | 10,269.69 ms | 37.94 ms | 67,139.45 ms | 163.11 | 71.23% / 5.983 |
| BF16 local，run35 | 2 | 9,732.26 ms | 39.59 ms | 69,070.76 ms | 152.82 | 67.12% / 5.699 |
| BF16 local，run35 | 3 | 9,814.25 ms | 34.22 ms | 61,112.70 ms | 184.91 | 77.96% / 6.459 |
| FP32 local，run36 | 1 | 9,775.09 ms | 42.33 ms | 73,225.29 ms | 146.44 | 60.79% / 5.257 |
| FP32 local，run36 | 2 | 9,725.13 ms | 38.45 ms | 67,359.65 ms | 156.40 | 70.66% / 5.945 |
| FP32 local，run36 | 3 | 9,766.81 ms | 35.10 ms | 62,384.27 ms | 181.08 | 76.28% / 6.344 |

| 三轮汇总 | BF16 local mean / median | FP32 local mean / median | FP32 相对变化 mean / median |
| --- | ---: | ---: | ---: |
| TTFT | 9,938.73 / 9,814.25 ms | 9,755.68 / 9,766.81 ms | -1.84% / -0.48% |
| TPOT | 37.25 / 37.94 ms | 38.63 / 38.45 ms | **+3.70% / +1.34%** |
| E2E | 65,774.30 / 67,139.45 ms | 67,656.40 / 67,359.65 ms | **+2.86% / +0.33%** |
| output throughput | 166.95 / 163.11 tok/s | 161.31 / 156.40 tok/s | **-3.38% / -4.11%** |
| accept rate，active-req weighted | 72.10% / 71.23% | 69.24% / 70.66% | **-2.86 / -0.57 pp**（相对 **-3.97% / -0.80%**） |
| accept len，log mean | 6.104 / 6.002 | 6.004 / 6.064 | -1.65% / +1.03% |
| accept len，active-req weighted | 6.047 / 5.983 | 5.849 / 5.945 | **-3.28% / -0.63%** |

三轮存在明显系统级波动，且 run35/run36 是先后启动的两个服务，不能把约 3%
差异解释成高置信度因果效应；但 precision 指标没有一致改善，性能方向也与生产
component model 的 +1.95%/+2.96% latency 预测一致。R13 未达到接受门槛。
run35 末尾的 Gloo/TCPStore warning 发生在第三轮结束 40 秒后，是切换 run36
时主动重启容器产生的 teardown 日志，不属于测量窗口。

## 下一步门禁

1. R13 不再扩展局部精度补丁；生产默认继续使用 BF16 local output。
2. 进入 canonical chunk state + 固定 merge tree 的 fused/streaming 实现，目标是
   保留 R14-P0 的 DCP-size-invariant 性质，同时消除 materialize leaves 的 77x
   原型开销。
3. 新实现先通过 CP={1,2,4,8} bitwise/高精度 oracle，再跑相同四节点 3 轮 E2E；
   accept-length 与 output tok/s 必须同时不回退。

## 原始证据

- `r13-accuracy-bf16.json` / `r13-accuracy-fp32.json`
- `r13-perf-dcp8-bf16.json` / `r13-perf-dcp8-fp32.json`
- `r13-sglang-merge.json` / `bench_sglang_dcp_merge.py`
- `FlashMLA-DCP-r13-build3.log`
- `ECS-rank0-r13-build.log`
- `e2e_results.csv`
- `outputs/kimi-k3-dcp8-e2e-20260826/logs/dcp8-r13-fp32local-*`
- 机器可读迭代表：`iterations.csv`
