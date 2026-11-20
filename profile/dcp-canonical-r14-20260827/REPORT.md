# R14-P0 canonical DCP 原型

## 结论

固定 global chunk 边界和固定 adjacent balanced merge tree 可以让 attention
结果不再依赖 DCP size。B1/Q8/H128/Dv512/KV8192、chunk=1024 的 H20 实验中，
CP=1/2/4/8 的 output 和 LSE 均 **bitwise identical**。这证明“无 DCP-size
偏差”的方案在算法上成立。

当前实现只是 PyTorch oracle：四种 CP 模拟延迟均约 7.46 ms，而同 shape 的
R13 真实 DCP8 FP32 operator 为 0.0966 ms，约慢 **77x**。因此 R14-P0 的状态是
`rejected-performance-proof-correctness`；它不能接入 SGLang，只作为下一版
fused SM90 leaf/tree kernel 的数值规范。

## 数据

| CP | bitwise invariant | prototype latency | output p99 | NRMSE |
| ---: | --- | ---: | ---: | ---: |
| 1 | true | 7.4717 ms | 2.32831e-9 | 5.06338e-7 |
| 2 | true | 7.4571 ms | 2.32831e-9 | 5.06338e-7 |
| 4 | true | 7.4719 ms | 2.32831e-9 | 5.06338e-7 |
| 8 | true | 7.4623 ms | 2.32831e-9 | 5.06338e-7 |

当前 R13 对照：8 卡、B1/Q8/H128/Dv512/KV8192、CUDA Graph、100 warm-up、
100 samples、每 sample 10 calls，FP32 `nccl_operator` p50 **0.0966 ms**。

## 验证

- `pytest -q tests/test_dcp_canonical.py`：**5 passed**。
- 输入 leaf 按模拟 rank 顺序计算，但按 global chunk id 放回固定槽位；tree
  运算图对所有 CP size 完全一致。
- `(0,-inf)` 中性态、ragged query suffix、causal chunk mask 均覆盖。

## 产物

- `r14-canonical-prototype.json`
- `r14-current-dcp8-fp32-b1q8.json`
- 机器可读迭代表：`iterations.csv`
- 完整设计：[`docs/dcp_canonical_merge.md`](../../docs/dcp_canonical_merge.md)
