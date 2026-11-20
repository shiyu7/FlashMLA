# R14-P7 token-cyclic canonical DCP：ECS 复验与 P8/P9 筛选

## 结论

R14-P7 在不改变 SGLang 现有 token-cyclic KV allocator 的前提下，把每 rank
本地 paged FP8 KV 重新组装为固定的 1024-token global chunk，再复用 R14-P6
的 direct-leaf exact peer reduce-scatter。两台独立 ECS H20 节点、6 次正式测量
的 critical latency 均值为 **0.123786 ms**，同轮 CP1 fused 均值为
**0.112967 ms**，代价为 **+9.58%**。6/6 output 和 LSE 都与 CP1
**bitwise equal**，最大绝对误差为 **0**；两节点的 dense FP8 回归均为
**16 passed**。

R14-P7 因而通过“无 DCP-size 数值偏差”和运行时布局兼容两项 operator
correctness gate，但没有通过 operator performance gate：同一 `.92` 节点的
当前生产 token-cyclic DCP8/NCCL operator p50 是 **0.099192 ms**，P7 正式均值
约慢 **24.92%**。这不能直接等价为 Kimi-K3 TPOT 回退 24.92%，因为完整模型
还有投影、MoE、通信和调度；也不能当作 DSPARK acceptance rate。P7 只保留为
E2E 候选，必须在四节点 8K A/B 中同时检查 TPOT 和 acceptance。

P8 的 cooperative persistent pack 和 P9 的 per-consumer read epoch 都保持
bitwise correctness，但分别比同轮 P7 smoke 慢 **1.12%** 和 **1.17%**，因此
均已拒绝，当前源码恢复到 P7。

## ECS 环境硬约束

本阶段只使用以下指定 ECS。SSH 必须走公网 IP 并显式指定
`/Users/bytedance/perfTest.pem`；私网 IP 仅用于节点间分布式通信。K8s 数据不计入
本报告的正式接受结论。

| rank | 公网 IP（SSH） | 私网 IP（分布式） | operator 复验 |
| ---: | --- | --- | --- |
| 0 | `115.191.2.23` | `192.168.44.90` | 3 次正式测量 |
| 1 | `115.191.21.96` | `192.168.44.93` | 未占用，留给四节点 E2E |
| 2 | `115.191.51.207` | `192.168.44.91` | 未占用，留给四节点 E2E |
| 3 | `101.126.12.182` | `192.168.44.92` | 3 次正式测量、P8/P9 筛选、nsys |

两台 operator 节点均为 8 x NVIDIA H20、SM90、NV18 全互联；容器为
`kimi-dcp-20260826`，PyTorch `2.11.0+cu130`，CUDA/NVCC `13.0`，driver
`535.161.08`。隔离源码目录为
`/data00/kimi-dcp/FlashMLA-DCP-r14p6-ecs-20260827`，原始源码快照未修改。

## Revision 迭代表

正式测量固定 B1/Q8/H128/Dv512/global-KV8192、8 x 1024-token chunk、FP8
cache、CUDA Graph、100 warm-up + 1000 iterations。P8/P9 是同一 `.92` 节点的
10 warm-up + 50 iterations 单变量筛选，只用于 accept/reject，不与六次正式
均值混算。

| revision_id | commit_info | 修改点 | performance | correctness | 结论 |
| --- | --- | --- | --- | --- | --- |
| current-DCP8 | base `3a0b48b8` 上的现有 token-cyclic + NCCL 路径 | local FlashMLA 后用 SGLang/NCCL merge | `.92` p50：local **0.025744 ms**，merge **0.072832 ms**，operator **0.099192 ms** | 既有生产路径 | performance baseline |
| R14-P7 | uncommitted；archive SHA256 `b24a2d12...90668` | local paged KV -> symmetric staging；8-peer canonical gather；write epoch + aggregated read-count；P6 direct leaf + exact peer-RS | 两 ECS 6-run：**0.123786 ms**；CP1 **0.112967 ms**；**+9.58%**；`.92` 对 current-DCP8 约 **+24.92%** | 6/6 output/LSE bitwise；max abs 0；两节点各 16 passed | **correctness accepted；E2E candidate** |
| R14-P8 | uncommitted rejected archive SHA256 `285c7014...928` | cooperative-groups persistent kernel 融合 prepare/stage/publish/wait/assemble/ack | smoke **0.126314 ms**；CP1 0.113306 ms；+11.48%；vs P7 smoke **+1.12%** | output/LSE bitwise；max abs 0 | **rejected** |
| R14-P9 | uncommitted rejected archive SHA256 `11b9b7c9...9cb` | 每 source 等待 8 个 consumer read epoch，替代 remote read-count atomic | smoke **0.126375 ms**；CP1 0.113492 ms；+11.35%；vs P7 smoke **+1.17%** | output/LSE bitwise；max abs 0 | **rejected** |

P7 的逐节点正式结果：

| ECS 私网标识 | critical mean | CP1 mean | overhead mean | bitwise |
| --- | ---: | ---: | ---: | --- |
| `.90` | **0.123660 ms** | 0.112688 ms | **+9.74%** | 3/3 output + LSE |
| `.92` | **0.123911 ms** | 0.113247 ms | **+9.42%** | 3/3 output + LSE |
| 合并 | **0.123786 ms** | 0.112967 ms | **+9.58%** | 6/6 output + LSE |

## 修改点与同步协议

生产 allocator 的 global token `i` 由 rank `i % 8` 持有，本地 offset 为
`i // 8`。R14-P6 benchmark 原先直接假设 rank 持有连续 1024-token chunk，若
直接接入 SGLang 会读取错误 KV。P7 保留 allocator，只增加以下 adapter：

1. `prepare` 等待上一 epoch 的 staging 被 8 个 consumer 全部确认读取；
2. `stage` 按 block table 把本地 token-cyclic paged FP8 KV 写入 symmetric
   staging；
3. `publish` 以 system-scope release atomic 发布 write epoch；
4. `wait` 以 system-scope acquire atomic 等待全部 8 个 peer；
5. `assemble` 按固定 global chunk 顺序从 8 个 peer staging 直接读取；
6. `ack` 对每个 source 的 aggregated read-count 做 system-scope release atomic。

`stage` 和 `assemble` 均使用 16-byte `uint4` copy，ptxas 均为 26 registers、
0 spill。epoch kernels 同样 0 spill。该协议把缓存布局兼容性和 exact fixed-tree
语义作为正确性约束，并允许 CUDA Graph 重放时安全复用 symmetric buffer。

## Profile 结论

`.92` 的 Nsight Systems 2026.4.1 报告位于 ECS
`/workspace/kimi-dcp/results/r14p7-nsys-92.nsys-rep`。跨 rank/replay 的 kernel
median：

| kernel | median |
| --- | ---: |
| canonical local attention leaf | 51.921 us |
| exact peer reduce-scatter | 38.576 us |
| wait for peers | 9.024 us |
| acknowledge | 8.960 us |
| publish | 3.072 us |
| assemble | 3.792 us |
| stage | 1.696 us |
| prepare | 1.664 us |

真实 KV copy（stage + assemble）只约 **5.49 us**；adapter 的主要额外成本来自
跨 rank epoch wait/ack 和多个短 kernel launch，而不是显存带宽。P8 尝试合并
launch 后反而更慢，P9 减少 remote atomic 写也没有收益，因此下一步更值得尝试
在 SGLang 侧把 P7 pack 与 Q quantization/projection 做跨 stream overlap，而不是
继续堆叠同步协议变体。

## 复现命令

构建：

```bash
FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=8 \
  python setup.py build_ext --inplace
```

正式 operator benchmark：

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
  benchmark/bench_dcp_canonical_tree.py \
  --transport peer-rs --leaf-output direct --kv-layout token-cyclic \
  --cuda-graph --warmup 100 --iterations 1000 \
  --output /workspace/kimi-dcp/results/r14p7-ecs-formal-<node>-rep<rep>.json
```

回归：

```bash
PYTHONPATH=. pytest -q tests/test_dense_fp8_dcp.py
```

## 数据与本地保存状态

- 六份 `r14p7-ecs-formal-*.json`：两节点正式数据；
- `r14p7-current-dcp8-baseline-92.json`：当前生产 DCP8 operator baseline；
- `r14p7-ecs-smoke-92.json`、`r14p8-ecs-smoke-92.json`、
  `r14p9-ecs-smoke-92.json`：单变量筛选；
- `source-archives/`：P7/P8/P9 三份源码归档，SHA256 已在表中记录。

因此即使释放 ECS，本阶段 accepted/rejected 源码和全部性能数据仍可从本地报告
目录恢复。当前工作树保留 P7，不包含 P8/P9 的 rejected 实现。
