# Kimi-K3 DCP8 接受率下降问题复盘

## 摘要

在 Kimi-K3 的 TP32/DCP8/DP4/EP32、DSPARK 推理链路中，我们曾观察到一个看似
“DCP 数值漂移”的问题：底层 SM90 FlashMLA DCP operator 已通过 bitwise
正确性测试，但完整服务的 DSPARK 接受率仍从 DCP1 的约 80% 下降到 74% 左右，
个别轮次甚至只有 65%；关闭 DSPARK 后，同一个确定性请求在四个 DP rank 上仍会
产生四种输出。

最终结论是：问题不在 DCP FlashMLA 的局部 attention、canonical merge 或
peer operator 数学，而在 SGLang prefill KV 写入路径。Triton attention backend
和 MLA KV pool 各执行了一次 DCP owner filter 与 `loc // dcp_size`，造成 KV
location 被映射两次。修复前，8K 请求的 8001 个 global RoPE KV row 中只有
553 个非零；修复后恢复为 8001/8001。

修复 R22 后：

- target-only Graph 与 DCP1 参考连续 12/12 逐 token 一致；
- DSPARK smoke 8/8 一致；
- 正式三轮 8K/1500 的接受率 mean/median 为 80.54%/80.69%，追平 DCP1；
- TPOT mean/median 从旧 DCP8 的 36.94/35.54 ms 降到
  32.60/32.46 ms；
- 输出吞吐从 168.82/177.38 tok/s 提高到 193.63/194.79 tok/s。

R22 的精度修复和相对性能已经通过接受门禁，但完整部署的 TTFT mean 为
10.01 s、TPOT mean 为 32.60 ms，仍未满足 5 s/20 ms 的绝对 SLO。

## 1. 背景

### 1.1 部署形态

本次问题发生在四台 8-GPU SM90 ECS 上，总计 32 张 H20 GPU。正式部署配置为：

| 维度 | 配置 |
| --- | --- |
| Target model | Kimi-K3 |
| Tensor Parallel | TP32 |
| Decode Context Parallel | DCP8 |
| Data Parallel | DP4 |
| Expert Parallel | EP32 |
| Decode attention | FlashMLA |
| Speculative decoding | DSPARK，draft block size 7 |
| KV cache | FP8 E4M3 |
| 正式 workload | 8000 input、1500 output、16 prompts、RR=1、MC=8 |

DCP8 把一条请求的 KV sequence 按八个 attention rank 分片。每个 rank 只保存和
计算约八分之一的 KV，随后把局部 attention state 按全局 LSE 合并。R14-P10
还把 peer reduce-scatter 的 owned H/8 state 直接写入 compact output，删除了
full-H 到 H/8 的下游复制。

### 1.2 什么是 DSPARK 接受率

DSPARK draft 每轮提出最多 7 个 token，target model 对这些 token 做验证。本文的
accept length 取值范围为 1--8，其中 1 表示本轮没有接受 draft token，8 表示
7 个 draft token 全部接受。接受率可近似写成：

```text
accept_rate = (accept_length - 1) / 7
```

接受率不仅是 speculative decoding 的性能指标，也是一种系统级精度信号：如果
target 的 logits 因 KV、position、query 或 attention 状态错误而变化，draft 和
target 的 token 更容易分叉，接受率会立即下降。接受率降低后，每个 target verify
step 产出的有效 token 变少，还会增加拒绝、回滚、重新验证和状态提交开销。

### 1.3 初始现象

R14-P10 的底层 operator 已在四台 ECS 上分别完成 8-GPU workspace 回归，覆盖
KV8192/9692，output/LSE 与 CP1 reference bitwise equal。但正式 E2E 仍出现：

| 方案 | TPOT mean/median | 输出吞吐 mean/median | accept length mean/median | 接受率 mean/median |
| --- | ---: | ---: | ---: | ---: |
| R14-P10 DCP8 | 36.94/35.54 ms | 168.82/177.38 tok/s | 6.167/6.466 | 73.86%/78.10% |
| DCP1 control | 38.82/38.74 ms | 166.83/167.89 tok/s | 6.606/6.613 | 80.09%/80.17% |

DCP8 性能方向已经为正，但接受率 mean 比 DCP1 低 6.23 个百分点。更异常的是，
DCP8 三轮接受率分别为 65.09%、78.37%、78.10%，第一轮存在明显坠落。

为了把 speculative decoding 本身从问题中剥离，我们关闭 DSPARK，只运行
target-only DCP8。结果是四个 DP rank 对同一确定性 8K 请求产生四种输出 hash。
这证明剩余问题位于 target DCP8 路径，而不是单纯的 draft 质量波动。

## 2. 评测与定位方法

我们把评测拆成四层，由外向内逐层收缩：

1. **请求层**：比较不同 DP rank 的输出 token/hash，检查确定性请求是否一致；
2. **DSPARK 层**：统计 accept length/rate，判断 target 与 draft 是否重新对齐；
3. **tensor 层**：保存首个 decode step 的 Q、selected KV、block table 和 output；
4. **operator 层**：用实际 tensor 在 GPU 上重建 CP1 FlashMLA，与 canonical
   DCP8 做 bitwise 比较。

性能与接受率使用同一正式协议：每轮开始前调用 `/flush_cache`，服务日志确认
`Cache flushed successfully!`；接受率不使用 bench-serving 返回的单个 DP
状态，而是在同一 UTC 时间窗内收集四个 DP 的 `Decode batch` 样本，按
`#running-req` 加权：

```text
weighted_accept_rate =
    sum(running_requests_i * accept_rate_i) / sum(running_requests_i)
```

这样可以避免 DP 返回顺序变化导致的统计偏差。

## 3. 主要怀疑方向

| 怀疑方向 | 为什么合理 | 关键验证 | 结论 |
| --- | --- | --- | --- |
| Canonical merge 或 FP32/BF16 舍入 | DCP rank 数变化会改变归约顺序 | 实际首步输入上重建 CP1，DCP8 output bitwise、max abs 0 | 排除 |
| accepted KV 跨 owner 搬运 | speculative compaction 后 source/target owner 可能变化 | 修复并通过 raw-FP8 单测，但 caller trace 显示 DSPARK topk=1 时不执行 | 独立问题，不是当前根因 |
| DSPARK ring 长度或 ReplaySSM 状态 | 低接受率可能来自 draft 历史不足 | ring32 服务可运行后仍只有 8/12 exact | 排除 |
| CUDA Graph/stream 可见性 | 首次或跨轮波动像未同步状态 | eager、显式 synchronize、Graph A/B 均未恢复一致性 | 排除为主因 |
| Hybrid allocator 原子页 | DCP logical page 与 physical page 可能不一致 | allocator 回归通过，但生产是 `hybrid_swa=False`，修改未命中 | 独立问题，不是当前路径 |
| Q replication 错误 | 八个 rank 可能在合并不同 query | 关闭 replication 仍失败；组内 gathered Q bitwise identical | 排除 |
| Prefill KV 写入/映射 | target-only 已失败，且 Q 基本一致 | DCP8/DCP1 actual tensor A/B；检查 KV nonzero coverage | 最终根因 |

## 4. 验证过程

### 4.1 R15：先检查 speculative accepted-KV relocation

旧的 `MLATokenToKVPool.move_kv_cache()` 只处理
`src_owner == tgt_owner == dcp_rank`。对于 topk>1 tree compaction，source 和
target owner 可能不同，因此我们实现了 raw-FP8 all-gather，再由 destination
owner 写回的方案。底层测试覆盖 8 个 rank、3 个 MLA layer、cross-owner 与
same-owner location，全部通过。

但进一步的 caller trace 显示，当前 DSPARK 使用 `speculative_eagle_topk=1`，
生产请求根本不会调用这段 relocation。关闭 DSPARK 后 target-only DCP8 仍然
0/4 exact。因此，R15 是一个有效的 topk>1 独立修复，却不能解释当前接受率下降。

### 4.2 R16--R18：排除 allocator、Graph 生命周期和 Q replication

R16 修复了 hybrid SWA allocator 没有按 DCP logical page 扩展的问题，CPU
regression 2 passed；但启动日志明确显示 Kimi-K3 当前路径是
`hybrid_ssm=True, hybrid_swa=False`，所以该改动没有进入生产 KV 路径。

R17 修复了 canonical workspace 只在 `init_cuda_graph_state()` 中创建、导致
eager decode 无 workspace 的生命周期问题。修复后 eager 可以完整运行，但输出
仍然分叉，说明 workspace 生命周期不是精度主因。

R18 关闭 Q replication，直接检查 DCP gather 后的 Q。同一 DCP group 八个 rank
上的 gathered FP8 Q bitwise identical，但四个 DP 仍产生四个输出 hash。因此
“不同 rank 合并了不同 query”也被排除。

### 4.3 R19：用实际输入证明 SM90 DCP kernel 没有偏差

我们保存了首个 decode step 的实际 Q、selected KV、block table 和 output，
然后在 rank0 GPU 上用同一份 Q/KV 重建 CP1 FlashMLA。

结果为：

| 比较项 | 结果 |
| --- | --- |
| 八个 DCP rank 的 gathered Q | bitwise identical |
| canonical DCP8 output vs CP1 output | bitwise identical |
| max absolute error | 0 |
| relative L2 | 0 |

这一步是定位过程中的关键转折：只要输入相同，SM90 canonical
pack/attention/peer merge 可以给出与 CP1 完全相同的结果。因此不能继续把接受率
下降归因于“DCP merge 天生有偏差”，而应向 attention 之前的数据准备路径追查。

### 4.4 R20：DCP1/DCP8 actual tensor A/B 指向 KV

随后运行 DCP1 eager trace。DCP1 的四个 DP 输出完全一致，八个 TP rank 的 KV
也 bitwise identical。将 DCP8 与 DCP1 的实际首步 tensor 对齐后得到：

| Tensor | cosine | relative L2 | 观察 |
| --- | ---: | ---: | --- |
| Q（关闭 replication） | 0.7879 | 0.6703 | 存在差异，但不能解释全部问题 |
| KV | 0.2657 | 0.9640 | 严重异常 |
| Attention output | 0.476 | 0.8819 | 被错误 KV 直接污染 |

token-cyclic、page-cyclic、contiguous 三种候选布局都无法解释差异，说明问题不是
简单地选错了一种 block-table 重排方式。

### 4.5 R21：恢复默认 Q replication 后锁定 KV 写入缺失

恢复生产默认 Q replication 后，DCP8 与 DCP1 的 Q 已基本一致：

| Tensor | exact fraction/cosine | relative L2 |
| --- | ---: | ---: |
| real Q | 93.625% exact；cosine 0.999952 | 0.01019 |
| KV | 0.744% exact；cosine 0.26585 | 0.9640 |

进一步统计 RoPE KV 的非零位置，发现修复前 global KV 只有 553/8001 个位置
非零，nonzero runs 为 `0--551` 和 `8000`。每个 DCP local rank 只保存了 69 个
旧 token；rank0 还正确保存了新生成位置 8000。

“旧 prefill KV 大量缺失、最新 decode token 正确”这一形态把问题准确指向了
prefill KV 写入，而不是 decode FlashMLA 读取或 merge。

## 5. 最终根因

### 5.1 错误的数据流

修复前，DCP MLA prefill 的 location 经过了两次映射。

第一步，`TritonAttnBackend._set_kv_buffer()` 已经执行：

```python
loc = forward_batch.out_cache_loc // dcp_size
dcp_kv_mask = positions % dcp_size == dcp_rank
token_to_kv_pool.set_kv_buffer(..., loc, dcp_kv_mask=dcp_kv_mask)
```

第二步，hybrid KV pool 的 MLA 分支没有使用 `dcp_kv_mask`，而是把已经除过一次的
`loc` 交给 `MLATokenToKVPool`。MLA pool 又执行：

```python
valid_mask = loc % dcp_size == dcp_rank
loc = loc[valid_mask]
loc = loc // dcp_size
```

也就是说，本应只执行一次的 owner filter 和 logical-to-compact-local translation
被执行了两次。大量 prefill token 在第二次 owner filter 中被错误丢弃，剩余
location 又被第二次压缩，最终写入了错误或过少的物理 KV row。

### 5.2 为什么 operator 单测没有发现

FlashMLA workspace/operator 单测从一份已经正确构造好的 token-cyclic KV 开始，
验证的是：

```text
正确 KV -> pack -> local attention -> fixed peer merge -> compact output
```

生产问题发生在更上游：

```text
prefill hidden states -> Triton backend -> Hybrid KV pool -> MLA KV pool
                     -> 错误/缺失 KV -> FlashMLA
```

因此 operator bitwise 通过与 E2E 失败并不矛盾。给 kernel 正确输入时它完全正确；
生产路径给它的是已经损坏的 KV。

### 5.3 R22 修复

R22 明确了 location translation 的唯一所有者：

```python
if dcp_size > 1:
    if use_mla:
        # MLA pool 负责唯一一次 owner filter 和 loc // dcp_size。
        loc = loc_info
        kwargs = {}
    else:
        # MHA 保持原有 pre-translated + mask 路径。
        loc = out_cache_loc // dcp_size
        kwargs = {"dcp_kv_mask": dcp_kv_mask}
```

这不是改变 attention 数学，也不是增加一次新的转换；它删除了 MLA 路径中
重复的那一次转换，同时保持 MHA 行为不变。

## 6. 修复后的正确性验证

| 验证层次 | 结果 |
| --- | --- |
| CPU regression | MLA logical loc 原样传入 pool；MHA 旧 masked write 不变；2 passed |
| KV nonzero coverage | 553/8001 恢复为 8001/8001 |
| local shard coverage | rank0 为 1001/1001；其余 rank 为 1000/1000 |
| R22 actual-tensor output | cosine 0.999946；relative L2 0.018674 |
| target-only eager | 四个 DP 4/4 exact |
| target-only Graph | 3 cycles × DP4，12/12 逐 token exact；1 hash |
| DSPARK smoke | 2 cycles × DP4，8/8 exact；accept length 8.0，rate 100% |

这里的 100% DSPARK smoke 使用固定重复 token，只能证明 speculative 全链路健康，
不能代替正式 random workload 接受率。因此最终结论仍以三轮 8K/1500 A/B 为准。

## 7. 正式 A/B Test

### 7.1 测试口径

三组结果使用同一 Kimi-K3 模型、DSPARK 配置、R14-P10 FlashMLA wheel 和固定
8K/1500 workload。R22 环境还保留了 R15--R17 的独立诊断修复，但 caller trace
和生产配置已经证明它们未进入当前 topk1/Graph 热路径；唯一被 actual-tensor
证据确认进入当前生产路径并解释精度变化的变量，是 MLA prefill KV location
修复。每组运行 3 轮，每轮 16 prompts、RR=1、MC=8，并在开始前清空 cache。

### 7.2 汇总结果

| 方案 | TTFT mean/median | TPOT mean/median | E2E mean/median | 输出吞吐 mean/median | 接受率 mean/median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 DCP8 R14-P10 | 9.92/9.91 s | 36.94/35.54 ms | 65.29/63.00 s | 168.82/177.38 tok/s | 73.86%/78.10% |
| DCP1 control | 9.89/9.60 s | 38.82/38.74 ms | 68.08/67.68 s | 166.83/167.89 tok/s | 80.09%/80.17% |
| **R22 DCP8** | **10.01/9.86 s** | **32.60/32.46 ms** | **58.89/58.52 s** | **193.63/194.79 tok/s** | **80.54%/80.69%** |

R22 三轮明细：

| Rep | TTFT | TPOT | 输出吞吐 | accept length | 接受率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.73 s | 33.49 ms | 189.84 tok/s | 6.656 | 80.71% |
| 2 | 9.86 s | 32.46 ms | 194.79 tok/s | 6.650 | 80.69% |
| 3 | 10.44 s | 31.86 ms | 196.25 tok/s | 6.611 | 80.23% |

### 7.3 相对收益

| 对照 | TTFT | TPOT | E2E latency | 输出吞吐 | accept length | 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R22 vs 旧 DCP8 | +0.95% | **改善 11.74%** | **改善 9.81%** | **提高 14.69%** | **提高 7.65%** | **+6.68 pp** |
| R22 vs DCP1 | +1.25% | **改善 16.01%** | **改善 13.50%** | **提高 16.06%** | 提高 0.50% | **+0.45 pp** |

TTFT 没有改善，且处于约 10 秒的同一噪声范围；主要收益集中在 decode TPOT、
E2E latency 和输出吞吐。这一结果与“修复 KV 后提高 DSPARK 有效 token 产出”
的机制一致，而不是一次 prefill 性能优化。

## 8. 性能收益来自哪里

### 8.1 R22 相对旧 DCP8：来自接受率恢复，而不是新 kernel

R22 与旧 DCP8 使用同一个 R14-P10 FlashMLA wheel，SM90 attention 和 peer
merge kernel 没有改变。因此 11.74% TPOT 与 14.69% 输出吞吐收益不能记成
kernel 加速。

修复后，target logits 不再被缺失 KV 污染，DSPARK 的 accept length 从 6.167
提高到 6.639，即每次 target verify 产出的有效 token 增加 7.65%。这直接减少了
生成固定 token 数量所需的 verify 轮数。更少的拒绝还意味着更少的 rollback、
ReplaySSM/KV 状态处理、调度与 batch 扰动，因此最终 TPOT 改善大于 accept
length 的单项变化。

这部分收益可以概括为：

```text
KV 正确性恢复
  -> target 与 draft 重新对齐
  -> accept length/rate 上升
  -> 单位输出 token 的 target verify 与回滚开销下降
  -> TPOT、E2E latency 和输出吞吐改善
```

需要注意，当前数据能证明强相关和执行机制，但没有对 rollback、verify、调度等
子项分别计时。因此不能把 11.74% 严格拆成每个子模块的百分比。

### 8.2 R22 相对 DCP1：来自 DCP 序列分片和 R14-P10 peer 路径

R22 的接受率已经与 DCP1 基本相同，mean 只高 0.45 个百分点。在 speculative
质量基本受控后，R22 仍比 DCP1 的 TPOT 快 16.01%，输出吞吐高 16.06%。这部分
才体现 DCP 路径本身的性能价值：

1. DCP8 让每个 attention rank 只保存并扫描约八分之一的 sequence KV；
2. rank-local FlashMLA attention 的 KV 工作量随之下降；
3. canonical fixed peer tree 合并八份局部 O/LSE，保持确定的全局归约顺序；
4. R14-P10 peer reduce-scatter 直接生成 compact H/8 output，删除 full-H 到 H/8
   的额外复制。

也就是说，旧 DCP8 的 attention 加速曾被低接受率带来的 speculative 浪费部分
抵消；R22 修复 KV 后，DCP 的局部 attention 加速才完整反映到 E2E TPOT 上。

### 8.3 R14-P10 operator 收益仍然存在，但不是 R22 的增量

R14-P10 相对 P7 consumer-ready 路径的 operator 收益约为 2%：

- C1024/KV8192：2.53%；
- C1216/KV8192：2.04%；
- C1216/KV9500：1.87%。

这来自删除 peer-RS 后的 full-H 到 H/8 copy。它是 R22 部署所继承的底层收益，
但由于旧 DCP8 和 R22 使用同一个 wheel，它不能解释两者之间的 11.74% TPOT
差异。

### 8.4 为什么 TTFT 没有收益

R22 修复的是 prefill KV 的写入正确性，而不是降低 prefill FLOPs、通信量或 kernel
latency。正式结果中 R22 TTFT mean 反而比旧 DCP8 高 0.95%，属于同一约 10 秒
区间的运行波动。因此不能宣称 R22 加速了 prefill。

当前绝对 SLO 仍未通过：

| SLO | R22 mean | 状态 |
| --- | ---: | --- |
| TTFT ≤ 5 s | 10.01 s | 未通过 |
| TPOT ≤ 20 ms | 32.60 ms | 未通过 |

后续优化应把 TTFT 和 decode TPOT 分开 profile，继续定位完整 K3 serving 中的
prefill、MoE、DSPARK verify、调度与通信占比，不能把剩余差距继续归因于已经通过
bitwise 验证的 SM90 DCP merge。

## 9. 最终结论

1. **这不是一种“有偏差的 DCP attention 方法”。** 在同一实际输入上，
   canonical DCP8 FlashMLA 与 CP1 output bitwise identical。
2. **接受率下降的根因是 SGLang 集成层的 KV location 双重映射。** Triton
   backend 和 MLA KV pool 重复执行 owner filter 与 `loc//8`，导致大部分
   prefill KV 未写入。
3. **R22 恢复了正确性。** KV coverage 从 553/8001 恢复到 8001/8001，
   target-only Graph 12/12 exact，正式接受率回到 80.54%，追平 DCP1。
4. **R22 相对旧 DCP8 的收益主要来自 speculative 有效工作恢复。** kernel
   wheel 不变；更高接受率减少了 verify、拒绝和回滚开销。
5. **R22 相对 DCP1 的约 16% TPOT/吞吐优势来自 DCP sequence sharding、局部
   FlashMLA 和 R14-P10 compact peer merge。** 这是排除接受率差异后更可信的
   DCP 性能收益。
6. **修复已接受，部署 SLO 仍待优化。** 精度和相对性能 gate 已通过，但
   5 s/20 ms 绝对 SLO 尚未满足。

## 10. 复现与数据

- [R14-P10--R22 完整 ECS 报告](../profile/dcp-canonical-r14p10-ecs-20260827/REPORT.md)
- [R22 三轮性能与接受率汇总](../profile/dcp-canonical-r14p10-ecs-20260827/e2e-run37-run38/r22_run39_summary.csv)
- [R22 on-node 接受率聚合](../profile/dcp-canonical-r14p10-ecs-20260827/e2e-run37-run38/r22_acceptance_onnode_aggregates.csv)
- [R22 target-only Graph 复测](../profile/dcp-canonical-r14p10-ecs-20260827/e2e-run37-run38/dcp8-r22-graph-dp-consistency-smoke27.json)
- [R22 DSPARK smoke](../profile/dcp-canonical-r14p10-ecs-20260827/e2e-run37-run38/dcp8-r22-dspark-dp-consistency-smoke28.json)
- [R22 SGLang source + regression patch](../profile/dcp-canonical-r14p10-ecs-20260827/source-archives/sglang-r22-mla-prefill-single-dcp-translation-20260827.patch)
- [SM90 DCP 总优化记录](dcp_sm90_optimization_log.md)

所有正式数据均来自指定的四台 ECS，没有使用 K8s 生成接受结论。原始 tensor
trace 只保留在 ECS；本地文档保存匿名聚合、结果 JSON 和可恢复源码 patch。
