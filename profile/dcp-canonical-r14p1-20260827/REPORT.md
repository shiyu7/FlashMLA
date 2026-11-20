# SM90 DCP R14-P1：fused canonical leaf + fixed local tree

## 结论

**R14-P1 通过 operator 门禁，继续进入 distributed fixed-tree 阶段。**

在单张 NVIDIA H20、B1/Q8/H128/Dv512/KV8192、8 个固定 1024-token
chunk 上，fused canonical operator 三轮均值为 **0.11135 ms**：相对生产
BF16 regular 的 0.12753 ms 加速 **12.69%**，相对 R13 FP32 regular 的
0.12878 ms 加速 **13.54%**，并低于设计停止线 0.12 ms。

这个结果只接受 P1 单卡算子，不代表 R14 已可进入生产。chunk-cyclic KV
layout、跨 rank 流式 fixed tree、CUDA Graph 和 8K DSPARK 接受率仍须在
P2--P4 完成。

## 修改点

- 新增 CUDA-Graph-safe canonical metadata kernel：每个 scheduler partition
  对应一个固定 global chunk；ragged tail 输出 `(0, -inf)` 中性态。
- 复用现有 SM90 FP8 FlashMLA TMA/WGMMA mainloop，一次 launch 批量生成
  8 个 FP32 leaf，避免 R14-P0 的逐 chunk Python 调度。
- 新增固定 adjacent balanced-tree combine kernel；lane 0 生成统一 LSE 树，
  全 warp 按同一树归并 FP32 output。
- 保持现有 BF16/FP32 regular 路径不变，canonical API 仍为 opt-in。
- 新增 ragged metadata、bitwise repeatability 和相对 regular FP32 的 GPU test。

## 环境与验证

| 项目 | 值 |
| --- | --- |
| 环境 | K8s `dev-sq-roleset-mknrd-decode-5678dcfc86-0` |
| GPU | NVIDIA H20，SM90，测试 GPU 0；节点共 8 卡且测试前均为 0 MiB |
| PyTorch / CUDA | 2.11.0+cu130 / CUDA 13.0 |
| 构建 | `FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=8 python setup.py build_ext --inplace` |
| correctness | `pytest -q tests/test_dense_fp8_dcp.py`：**15 passed** |
| 计时 | warm-up 200，iterations 2000，三轮配对；CUDA Event |

ECS 四节点本轮未用于 P1：本机 Kerberos cache 为空，公网入口在跳板机
GSSAPI 阶段被拒绝。K8s 数据只作为 H20 operator 门禁；P3/P4 仍需回到指定
ECS 环境验证。

## 性能数据

| revision_id | commit_info | rep | BF16 regular | FP32 regular | canonical fused | vs BF16 | vs FP32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R14-P1 | uncommitted；kernel `bfa7ac98...6953`；metadata `7f13737c...9b63` | 1 | 0.127514 ms | 0.128799 ms | **0.111285 ms** | **-12.73%** | **-13.60%** |
| R14-P1 | same binary | 2 | 0.127535 ms | 0.128773 ms | **0.111529 ms** | **-12.55%** | **-13.39%** |
| R14-P1 | same binary | 3 | 0.127555 ms | 0.128773 ms | **0.111239 ms** | **-12.79%** | **-13.62%** |
| R14-P1 mean | same binary | — | 0.127535 ms | 0.128782 ms | **0.111351 ms** | **-12.69%** | **-13.54%** |

三轮均满足：重复执行 output/LSE **bitwise identical**；相对 regular FP32，
output max abs 为 `1.464982e-4`，LSE max abs 为 `9.536743e-7`。两条路径的
归约图不同，因此不要求与 regular bitwise 相同。

## Profile 解释

Nsight Systems 的 NVTX 区间为 BF16 0.12907 ms/次、FP32 0.12975 ms/次、
canonical 0.11221 ms/次，与 CUDA Event 结果一致。

Nsight Compute 的关键对比：

| kernel | grid | waves/SM | duration | compute | registers |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 regular leaf | 80 CTAs | 1.03 | 139.23 us | 48.41% | 251 |
| canonical FP32 leaf | 128 CTAs | 1.64 | 116.29 us | 59.65% | 255 |
| canonical fixed-tree combine | 128 CTAs | 0.41 | 10.98 us active | 15.52% | 64 |

收益来自固定 1024-token leaf 缩短单 CTA KV 循环、提高计算利用率和负载均衡，
不是 FP32 写回本身。combine 的 88-byte stack frame没有 spill；其大部分 timeline
跨度是 PDL 等待 leaf，当前不在关键路径上，不为它做低收益的寄存器重写。

## Artifacts

- 三轮 JSON：[`rep1`](r14p1-h20-bf16-rep1.json)、
  [`rep2`](r14p1-h20-bf16-rep2.json)、
  [`rep3`](r14p1-h20-bf16-rep3.json)
- Nsight Systems：[`r14p1-h20-timeline.nsys-rep`](r14p1-h20-timeline.nsys-rep)
- Nsight Compute：[`canonical leaf`](r14p1-h20-canonical-leaf.ncu-rep)、
  [`BF16 leaf`](r14p1-h20-bf16-leaf.ncu-rep)、
  [`canonical combine`](r14p1-h20-canonical-combine.ncu-rep)

## 下一步

R14-P2 增加 leaf-state 导出和 standalone fixed-tree merge，使 CP1/2/4/8 在
真实 SM90 leaf 上恢复同一 global chunk 顺序；通过后再实现每层 point-to-point
流式树，避免 all-gather 全部 leaves。
