# SM90 DCP R0--R11 总览

## 一张表看完

所有 revision 都基于 `3a0b48b8e7715ebc0e03aa09a1700868ccc97916`
之上的隔离构建或 dirty worktree。性能列使用每一轮实际用于 accept/reject 的主
指标：

- R0--R2：`peer_merge / peer_operator` p50 几何平均；
- R3--R4：`local` p50 几何平均；
- R5--R11：`local / peer_operator` p50 几何平均。

因此 R2 和 R3 之间不是同一性能指标，不能把整列当成一条连续 latency 曲线。
R4 在新 Pod 上为 R5 重新测过 baseline；R10/R11 则使用紧邻的 R8 A/B。表中
`speedup` 都明确标出自己的 reference。

| revision | 优化范围 | 核心修改 | node-a performance | node-b performance | 相对结果 | 状态 | 最终 R11 保留 |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| R0 | DCP2 peer merge | 256 threads；BF16x2、32-bit load/store | 31.713 / 60.449 us | 31.787 / 60.521 us | baseline | baseline | 否 |
| R1 | DCP2 peer merge | 64 threads；`uint4` 128-bit；4 组 BF16x2；保留 fallback | 20.097 / 47.895 us | 20.003 / 47.962 us | vs R0：**1.578x / 1.262x**；**1.589x / 1.262x** | **accepted** | **是** |
| R2 | DCP2 peer merge | 32 threads；每线程两个 `uint4`；block barrier 改 warp barrier | 未运行 | 20.378 / 48.350 us | vs R1：**0.9816x / 0.9920x** | rejected | 否 |
| R3 | local scheduler | 所有 batch 无条件 ceil `num_sm_parts`，补齐最后 SM wave | 未运行 | 26.345 us | vs R1-local：**0.9883x** | rejected | 否 |
| R4 | local scheduler | 仅 B1 ceil；B>1 保留 floor 调度 | 25.371 us | 25.540 us | vs R1-local：**1.0170x / 1.0195x** | **accepted** | **是** |
| R5 | local mainloop | 删除 PV GEMM 前两个冗余 FP8 CTA barrier | 25.112 / 46.592 us | 25.164 / 46.675 us | vs R4-rebaseline：local **1.0225x / 1.0251x** | **accepted** | **是** |
| R6 | local split combine | <=512 rows 保留旧映射；大 launch 一 warp/row、4 rows/CTA、`float4` load | 24.122 / 45.774 us | 24.235 / 45.937 us | vs R5：local **1.0411x / 1.0383x** | **accepted** | **是** |
| R7 | local split combine | 预取 split 0，并在消费当前 split 时流水加载下一 split | 23.769 / 45.528 us | 23.901 / 45.683 us | vs R6：local **1.0149x / 1.0140x** | **accepted** | **是** |
| R8 | main→combine overlap | 仅 `total_rows > 512` 启用 Hopper PDL；小 launch 保持非 PDL | 23.361 / 44.834 us | 23.398 / 44.943 us | vs R7：local **1.0175x / 1.0215x** | **accepted** | **是** |
| R9 | combine launch shape | 所有 grouped combine 都使用 8 rows/CTA | 23.408 / 44.851 us | 未运行 | vs R8：local **0.9980x** | rejected | 否 |
| R10 | split LSE numerics | private split LSE 使用 base-2；最终 public LSE 转回自然对数 | 23.369 / 44.853 us | 未运行 | vs 相邻 R8：local **0.9996x** | rejected | 否 |
| R11 | combine crossover | 仅 `total_rows == 1024` 使用 8 rows/CTA；其他大 launch 保持 4 rows/CTA | 23.299 / 44.763 us | 23.355 / 44.901 us | vs 相邻 R8：local **1.0017x / 1.0023x** | **accepted** | **是** |

性能列中两个数字的含义随阶段变化：R0--R2 为
`peer_merge / peer_operator`，R5--R11 为 `local / peer_operator`。R3--R4
当轮只记录了 local 主指标。

## 最终累计结果

最终 R11 包含的 accepted 路线是：

```text
R1 peer 128-bit merge
  + R4 B1 scheduler
  + R5 redundant-barrier removal
  + R6 warp-owned vectorized combine
  + R7 split prefetch pipeline
  + R8 guarded Hopper PDL
  + R11 1024-row CTA crossover
```

两段可审计的累计收益分别是：

| 累计区间 | node-a | node-b |
| --- | ---: | ---: |
| R0→R1 peer merge | **1.578x** | **1.589x** |
| R0→R1 peer operator | **1.262x** | **1.262x** |
| R5→R11 local | **1.07785x (+7.785%)** | **1.07745x (+7.745%)** |
| R5→R11 peer operator | **1.04087x (+4.087%)** | **1.03951x (+3.951%)** |

R1→R4 和 R4→R5 跨越了独立测量批次与 Pod 重建，因此没有把 R0→R11
强行合成为单一 speedup。最终 R11 的 absolute latency 和逐段 A/B 均来自原始
JSON，而不是用不同批次的几何均值拼接。

## 数据索引

- [统一机器可读表](./iterations.csv)
- [R0--R2 peer merge 数据](../dcp2-sm90-20260808/iterations.csv)
- [R3--R4 local scheduler 数据](../dense-fp8-sm90-r3-20260810/iterations.csv)
- [R5 mainloop barrier 数据](../dense-fp8-sm90-r5-20260812/iterations.csv)
- [R6--R11 combine/PDL 数据](../dense-fp8-sm90-r6-r11-20260813/iterations.csv)

中间候选（R5-VT、R6 rows1/4/8、R6 transpose ceiling、R8 full-PDL）仍保留在
各阶段报告中，但不占用 R0--R11 主 revision 的一行。
