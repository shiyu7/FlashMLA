# DCP-size 不变量：canonical chunk + fixed merge tree

## 目标定义

“无偏差”在这里定义为：给定相同 Q/KV、相同 chunk 配置和相同 GPU kernel，
CP=1/2/4/8 的最终 attention output/LSE **bitwise identical**。它消除的是
DCP-size 引入的数值漂移，不承诺与当前未分块 CP1 FlashMLA bitwise 相同；
CP1 对照也必须走 canonical 路径。

现有 DCP 在实数域严格等价，但 token-cyclic KV 会让不同 CP size 使用不同的
局部归约边界与 merge 树。浮点加法、`exp` 和归一化并不满足结合律，所以
“把 rank-local O/LSE 最后合起来”仍会留下 CP-size 相关舍入。R13 只删除了
rank-local BF16 cast，无法改变归约图。

## 数值契约

全局 KV 按固定大小 `C` 切成连续 chunk，边界只由 global token index 决定：

```text
chunk i = [i * C, min((i + 1) * C, sequence_length))
owner(i) = i % dcp_size
```

每个 chunk 产生 FP32 状态 `(O_i, L_i)`：`O_i` 是该 chunk 内归一化输出，
`L_i` 是 natural-log LSE。两个相邻状态按固定左右顺序合并：

```text
L = logaddexp(L_left, L_right)
O = exp(L_left - L) * O_left + exp(L_right - L) * O_right
```

空状态固定为 `(0, -inf)`。所有 chunk 按 global chunk id 进入相邻 balanced
tree：先 `(0,1), (2,3), ...`，再合并相邻父节点；奇数尾节点原样上移。
chunk owner、到达顺序和 DCP size 都不能改变这棵树。

## SM90 实现草图

1. **KV layout**：从 token-cyclic 改成 chunk-cyclic；chunk 内保持连续 page，
   cache insertion、accepted-token relocation 和 block table 都按 global chunk id
   映射。
2. **leaf kernel**：复用 dense FP8 FlashMLA mainloop，但 scheduler 以固定
   global chunk 为输出单位。一个 launch 批量生成本 rank 拥有的 leaf states，
   输出 FP32 O/LSE；不在 leaf 内跨 canonical chunk combine。
3. **tree kernel**：每层只执行固定 left/right merge，FP32 输入输出，允许原位
   覆盖 surviving left node。ragged/causal 行用 `(0,-inf)` 中性态补齐。
4. **通信**：不能 all-gather 全部 leaf。每层由 right owner 把 state 发给 left
   owner，left owner merge 后进入下一层；最后再按 query-head shard 分发。单层
   NCCL group send/recv 或对称内存 peer load 都必须保持相同运算次序。
5. **内存**：流式树只保留本地存活 state 和一个 receive buffer；禁止
   `num_chunks * B * Q * H * Dv` 全量常驻。CUDA Graph 按 chunk-count bucket
   预分配，缺失叶子填中性态。

推荐初始 `C=1024`：8K 上固定 8 个 leaf，chunk 内仍含 16 个 64-token K tile；
后续只在 512/1024/2048 三点做性能与数值 crossover。chunk 大小属于模型运行
配置，不能随 CP size 改变。

## 原型结果与门禁

R14-P0 用 PyTorch materialize 8 个 leaf 并执行固定树。在 H20、B1/Q8/H128/
Dv512/KV8192 上：

- CP=1/2/4/8 output 和 LSE 全部 bitwise identical；
- 相对 unsharded FP32 reference，p99 绝对误差 2.33e-9，NRMSE 5.06e-7；
- unfused latency 约 7.46 ms；当前 R13 DCP8 FP32 operator 是 0.0966 ms，
  原型约慢 **77x**。

所以 R14-P0 只证明数值可行，性能上明确拒绝。生产候选门禁：

| 阶段 | 必须通过的门禁 |
| --- | --- |
| P1 leaf | 单 launch 批量 leaf；相对 R13 local <= 15% 回退；leaf 对 oracle |
| P2 local tree | CP size simulation bitwise identical；无 NaN/Inf mask mismatch |
| P3 distributed | CP={1,2,4,8} bitwise identical；CUDA Graph；无 host sync |
| P4 E2E | 8K DSPARK 至少 3 轮配对；accept gap 显著收敛；TPOT 回退 <= 5% |

如果 P1/P2 不能把 B1/Q8 operator 压到 0.12 ms 内，应停止该路线；两遍全局
softmax或 FP64 all-reduce 只适合作为 oracle，不进入在线服务。
