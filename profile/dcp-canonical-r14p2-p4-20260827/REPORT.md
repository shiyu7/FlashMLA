# R14-P2--P4：SM90 无漂移 DCP8 fixed-tree

> 环境状态：本报告数据来自早期 K8s 原型筛选，按最新硬约束不再作为正式接受
> 依据。后续必须只在四台指定 ECS 上复测；SSH 使用公网 IP 和
> `/Users/bytedance/perfTest.pem`。公网/私网映射与完整约束见
> [`docs/dcp_sm90_optimization_log.md`](../../docs/dcp_sm90_optimization_log.md)。

## 结论

R14-P2 导出了固定 global chunk 的 FP32 attention leaf；模拟
CP=1/2/4/8 的 chunk-cyclic ownership 并恢复 global chunk id 后，每个 leaf、
最终 output 和 LSE 均逐 bit 相同。R14-P3 用 NCCL P2P 执行三层 adjacent
tree，精度仍为 bitwise equal，但 8K critical path 平均 **0.19946 ms**，比
CP1 fused **0.11187 ms** 慢 **78.31%**，因此拒绝。

R14-P4 改用 PyTorch symmetric memory 和 SM90 system-scope epoch：每个
receiver kernel 直接从 NV18 peer mapping 读取 FP32 state，三层树不再发起
NCCL P2P。三轮均值为 **0.10550 ms**，相对同轮 CP1 fused
**0.11166 ms 加速 5.52%**；三轮 output/LSE 都与 CP1 **bitwise equal**，
max abs 均为 0。P4 通过本轮 operator correctness/performance 门。

P4 仍是 root-owned tree 原型：最终完整 state 只在 rank 0，有效性能包含 local
leaf、写入 symmetric workspace、peer read 和 fixed merge，但不包含生产运行时
需要的 output ownership/后续分发，也未进入 Kimi-K3/DSPARK E2E。因此这里不
声明 acceptance rate 改善；E2E acceptance 必须在完成 head-sharded
reduce-scatter ABI 和 SGLang 接入后另测。

## 环境与命令

- Pod：`default/dev-sq-roleset-mknrd-decode-5678dcfc86-0`
- GPU：8 x NVIDIA H20，SM90，GPU 间均为 NV18
- PyTorch：2.11.0+cu130
- CUDA/NVCC：13.0
- build：`FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=8 python setup.py build_ext --inplace`
- shape：B1/Q8/H128/Dv512/global KV8192，8 x 1024-token chunks
- 每轮：100 warm-up + 1000 iterations；P2 为 200 + 2000

P3/P4：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$PWD \
torchrun --standalone --nproc-per-node=8 \
  benchmark/bench_dcp_canonical_tree.py \
  --transport {nccl,peer} --warmup 100 --iterations 1000
```

## Revision 数据

| revision_id | commit_info | 修改点 | performance（三轮均值） | 精度 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| R14-P1 | uncommitted；kernel `40453d11...d47c` | 8 leaf + local fixed tree fused | 0.11135 ms | repeat bitwise | accepted local baseline |
| R14-P2 | leaf ABI `f027e16b...bf2f`；bench `63bc1444...22c3` | export `[B,C,Q,H,D]` FP32 leaf；CP ownership 后恢复 chunk id | leaf-only **0.08287 ms**；输出 16,809,984 bytes | CP1/2/4/8 leaf/output/LSE bitwise | **accepted** |
| R14-P3 | pair/tree bench `284507d3...1835` | NCCL P2P 三层树；ordered FP32 pair merge | **0.19946 ms**；vs CP1 **+78.31%** | 3/3 bitwise；max abs 0 | **rejected** |
| R14-P4 | peer kernel `dbe90312...0bde`；binding `a56555a4...bb42` | symmetric memory；row epoch；receiver direct NVLink read；in-place tree | **0.10550 ms**；CP1 0.11166 ms；**-5.52%** | 3/3 bitwise；max abs 0 | **operator accepted** |

P4 三轮 critical latency：0.105071、0.105730、0.105708 ms；对应 CP1：
0.111660、0.111667、0.111668 ms。P3 三轮 critical latency：0.201644、
0.195546、0.201205 ms。

## 修改点与解释

### R14-P2

- canonical metadata 扩展到单 local chunk，使 CP8 每 rank 能生成同一个 global
  chunk 的真实 SM90 leaf。
- mainloop 后直接返回 FP32 accumulator view，不运行 local combine。
- 固定 chunk 的算术边界独立于 CP ownership；测试按 global chunk id 复原后
  使用同一 adjacent balanced tree。

### R14-P3

- 新增 128-thread FP32 `float4` ordered pair merge，ptxas 为 26 registers、
  1 barrier、0 spill。
- tree edges 固定为 1→0/3→2/5→4/7→6，2→0/6→4，4→0。
- correctness 达标，但 critical path 需要三层、每层 output+LSE 两个 NCCL P2P，
  6.30 MiB critical payload 加上 collective launch/host 调度使 8K 回退。

### R14-P4

- output/LSE/epoch 使用 persistent symmetric memory rendezvous；CUDA kernel
  通过 peer virtual address 直接 load，不 materialize all-gather。
- 每 row 保留 leaf/level1/level2/read 四个 system-scope epoch，防止 CUDA Graph
  replay 或连续迭代提前覆盖尚未消费的 state。
- pair merge继续严格使用 left global subtree、right global subtree顺序；所有
  中间 output/LSE 均为 FP32，因而精确复现 CP1 的三层算术图。

## 验证

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD pytest -q tests/test_dense_fp8_dcp.py
16 passed in 1.74s
```

新测试覆盖 SM90 leaf export 的 CP1/2/4/8 invariant，并让 standalone pair
kernel 与 P1 fused tree 做 rtol=0/atol=0 对比。P3/P4 的 8-rank benchmark
另与 CP1 fused output/LSE 做 `torch.equal`。

原始 JSON 位于本目录的 `r14p2/`、`r14p3/` 和 `r14p4-run-*.json`。
