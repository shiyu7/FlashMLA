# R13 四节点 E2E 最终结论

## 决策

**拒绝 R13 FP32 rank-local output，生产默认继续使用 BF16 rank-local
output。** FP32 路径在最关键的解码性能和 DSPARK 接受率两项上均未获得收益。

## 核心结果

环境：4 台 ECS、32 x NVIDIA H20，TP32/DP4/EP32/DCP8，DSPARK block 7，
固定 8K 输入 / 1500 输出，16 prompts，request rate 1，最大并发 8。BF16 与
FP32 各运行 3 轮，每轮前清空 radix/mamba cache。

| 核心指标 | BF16 local mean / median | FP32 local mean / median | FP32 相对变化 mean / median |
| --- | ---: | ---: | ---: |
| 输出吞吐 | 166.95 / 163.11 tok/s | 161.31 / 156.40 tok/s | **-3.38% / -4.11%** |
| TPOT | 37.25 / 37.94 ms | 38.63 / 38.45 ms | **+3.70% / +1.34%** |
| 接受率（active-request weighted） | 72.10% / 71.23% | 69.24% / 70.66% | **-2.86 / -0.57 pp**（相对 **-3.97% / -0.80%**） |
| 接受长度（active-request weighted） | 6.047 / 5.983 | 5.849 / 5.945 | **-3.28% / -0.63%** |

TTFT 均值从 9,938.73 ms 降至 9,755.68 ms（-1.84%），但这不足以抵消
TPOT、输出吞吐和接受率的回退。三轮存在系统级波动，因此不把约 3% 的差异
宣称为高置信度精确效应；但接受率没有一致改善，性能方向也为负，已足以拒绝
R13。

## Revision 记录

| revision_id | commit_info | 修改点 | performance | acceptance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R13-control-run35 | uncommitted；FlashMLA archive `aa1d46e0...cc93`；SGLang overlay `e1fe2122...aaf` | `SGLANG_DCP_FP32_LOCAL_OUTPUT=0`，BF16 local state | output 166.95 tok/s；TPOT 37.25 ms | rate 72.10%；length 6.047 | baseline |
| R13-run36 | 同一 binary/overlay | `SGLANG_DCP_FP32_LOCAL_OUTPUT=1`，FP32 local state，global merge 后 cast | output 161.31 tok/s（-3.38%）；TPOT 38.63 ms（+3.70%） | rate 69.24%（-2.86 pp）；length 5.849（-3.28%） | **performance rejected** |

## 含义与下一步

本实验否定了“rank-local BF16 舍入是剩余 DSPARK 接受率差距主要原因”的
假设。R13 作为 opt-in 实验路径保留，但不进入生产默认配置。下一步转向 fused
canonical chunk state + 固定 merge tree，在保持 R14-P0 DCP-size-invariant
数值性质的同时消除未融合原型约 77x 的开销。

原始数据见 [`e2e_results.csv`](e2e_results.csv)，完整实验过程见
[`REPORT.md`](REPORT.md)，四节点 benchmark JSON 与服务日志见
[`../../outputs/kimi-k3-dcp8-e2e-20260826/logs`](../../outputs/kimi-k3-dcp8-e2e-20260826/logs)。
