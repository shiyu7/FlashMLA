# R14-P6：ECS SM90 exact DCP8 peer reduce-scatter

## 结论

R14-P6 已在两台独立 ECS H20 节点完成正式 operator 复测。固定
B1/Q8/H128/Dv512/global KV8192（每 rank 一个 1024-token global chunk）、
CUDA Graph、100 warm-up + 1000 iterations 的 6 次运行中：

- DCP8 critical latency 均值为 **0.095784 ms**；
- 同轮 CP1 fused 均值为 **0.112692 ms**；
- 每轮相对提升均值为 **15.00%**，范围 **14.70%--15.22%**；
- 6/6 output 和 LSE 与 CP1 **bitwise equal**，max abs 均为 0；
- 两台节点独立编译后，`tests/test_dense_fp8_dcp.py` 均为 **16 passed**。

因此 R14-P6 通过 ECS operator correctness/performance 门。这里的“精度接受”是
算子 output/LSE 逐 bit 相等，不是 DSPARK speculative acceptance rate。P6 尚未
接入四节点 Kimi-K3/SGLang，不能声明 acceptance length/rate 已改善。

## 调试环境硬约束

本轮只使用指定 ECS，SSH 用公网 IP 和密钥
`/Users/bytedance/perfTest.pem`；私网 IP 只用于节点间通信。K8s 未参与本轮正式
编译、测试或性能数据生成。

| 角色 | 公网 IP（SSH） | 私网 IP | hostname | 本轮用途 |
| --- | --- | --- | --- | --- |
| ECS-90 | `115.191.2.23` | `192.168.44.90` | `ECS-serving-3` | 独立 build/test + 3 reps |
| ECS-92 | `101.126.12.182` | `192.168.44.92` | `ECS-serving-0` | 独立 build/test + 3 reps |

其余指定 ECS `.93`/`.91` 保留给后续四节点 E2E。本轮两台节点均为 8 x NVIDIA
H20、SM90、GPU 间 NV18 全互联，driver 535.161.08。容器和工具链为：

- image：`iaas-gpu-cn-beijing.cr.volces.com/sunqi/sglang:kimi-0810`；
- image digest：`sha256:7c4f7f8fe146...16eb6`；
- PyTorch：2.11.0+cu130；
- CUDA/NVCC：13.0 / 13.0.88；
- build：`FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=8 python setup.py build_ext --inplace`。

源码从 host `/data00/kimi-dcp/FlashMLA-DCP-3a0b48b-20260826` 复制到隔离目录
`/data00/kimi-dcp/FlashMLA-DCP-r14p6-ecs-20260827`，未覆盖原始快照。容器内对应
路径为 `/workspace/kimi-dcp/FlashMLA-DCP-r14p6-ecs-20260827`。

## Revision 数据

| revision_id | commit_info | 修改点 | performance | 精度/接受 | 状态 |
| --- | --- | --- | --- | --- | --- |
| R14-P4 | uncommitted；peer root tree `dbe90312...0bde` | rank0-owned symmetric peer tree | K8s 预筛选 0.10550 ms；CP1 0.11166 ms | K8s 3/3 bitwise | 历史预筛选，非正式接受 |
| R14-P5 | uncommitted；peer-RS `aa3a4e3a...60a` | contiguous H/8 head-sharded exact reduce-scatter | 由 P6 覆盖 | K8s 预筛选 bitwise | subsumed |
| R14-P6 | base `3a0b48b8`；archive `1739d613...b9a`；mainloop `40453d11...47c`；bench `143b203f...dd7` | direct symmetric FP32 leaf output + P5 peer-RS + CUDA Graph | **0.095784 ms**；CP1 0.112692 ms；**15.00% faster** | ECS 6/6 bitwise；两节点各 16 passed | **ECS operator accepted**；E2E pending |

P5 的最终 kernel ptxas 为 30 registers、1 barrier、20 bytes shared memory、0
spill；prepare kernel 为 10 registers、0 spill。主 FP8 attention translation unit
仍存在部分 576-dim variant 的 255-register 和 16-byte store/40-byte load spill，
是下一轮 local-kernel 优化候选，但未在本轮混入额外变量。

## 正式性能数据

| ECS | rep | DCP8 critical (ms) | CP1 fused (ms) | DCP8 提升 | output/LSE |
| --- | ---: | ---: | ---: | ---: | --- |
| `.90` | 1 | 0.095706 | 0.112557 | 14.971% | bitwise / bitwise |
| `.90` | 2 | 0.095740 | 0.112617 | 14.986% | bitwise / bitwise |
| `.90` | 3 | 0.095608 | 0.112530 | 15.038% | bitwise / bitwise |
| `.92` | 1 | 0.095846 | 0.113059 | 15.225% | bitwise / bitwise |
| `.92` | 2 | 0.095944 | 0.113017 | 15.107% | bitwise / bitwise |
| `.92` | 3 | 0.095858 | 0.112372 | 14.696% | bitwise / bitwise |
| **6-run mean** |  | **0.095784** | **0.112692** | **15.004%** | **6/6 exact** |

每次测量的 critical path 是 8 ranks 中最大 latency；每个 rank 最终只拥有连续
H/8 output shard。critical path payload 为 1,838,592 bytes。

## 命令与验证

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  pytest -q tests/test_dense_fp8_dcp.py -s

PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
  benchmark/bench_dcp_canonical_tree.py \
  --transport peer-rs --leaf-output direct --cuda-graph \
  --warmup 100 --iterations 1000 --output <result.json>
```

`.90` 测试为 `16 passed in 1.80s`；`.92` 为 `16 passed in 1.68s`。最窄的
canonical leaf invariant 在 `.92` 先单独运行，为 `1 passed, 15 deselected`。

## 修改点

- local SM90 FP8 attention 对单 global chunk 直接写 caller-owned symmetric FP32
  leaf，避免中间 output allocation/copy；
- fixed adjacent tree 保持 `(0,1) (2,3) (4,5) (6,7)`、`(0,2) (4,6)`、
  `(0,4)` 的全局 operand 顺序；
- 每 rank 只归约并持有连续 H/8 head shard，避免 root-owned full-state 后续分发；
- persistent symmetric workspace 使用 leaf/level1/level2/read epoch，保持 CUDA
  Graph replay 的覆盖安全和 system-scope acquire/release 顺序；
- LSE 当前仍有 4 KiB allocation/copy，可继续做小幅 P7 优化，但必须用相同 6-run
  ECS protocol 接受或拒绝。

## 原始证据

本目录的六个 `r14p6-ecs-formal-*-rep*.json` 是机器可读原始数据。后续 SGLang
接入必须另存四节点 8K E2E JSON，并同时突出 TPOT/output throughput 和 DSPARK
acceptance rate/length。
