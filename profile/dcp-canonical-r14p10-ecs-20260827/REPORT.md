# R14-P10--R22 compact canonical DCP：ECS 优化与精度闭环

## 结论

R14-P10 保留 R14-P7 的 token-cyclic KV adapter 和 fixed peer tree，只把最终
peer reduce-scatter 的 owned H/8 state 直接写入 compact BQHD/BHQ buffer。与
“P7 full-H 输出后再显式 compact copy”的 consumer-ready 对照相比，`.92` 上
C1024/KV8192 加速 **2.53%**，C1216/KV8192 加速 **2.04%**，
C1216/KV9500 加速 **1.87%**。三组 output/LSE 均与 CP1 bitwise equal，最大
绝对误差为 0，因此 P10 通过 operator 单变量接受门禁。

四节点 Kimi-K3 的固定 8K/1500、3-rep A/B 现已完成。P10 DCP8 的 mean/median
TPOT 为 **36.94/35.54 ms**，输出吞吐为 **168.82/177.38 tok/s**；同代码 DCP1
对照为 **38.82/38.74 ms** 和 **166.83/167.89 tok/s**。性能方向为正，但 DCP8
active-request 加权 acceptance rate 的 mean/median 为 **73.86%/78.10%**，
对照为 **80.09%/80.17%**。P10 相对 R13 DCP8 的 72.10% 均值已有改善，但仍
未追平 DCP1，且第一轮只有 65.09%。因此 P10 保持 **operator accepted** 和
**E2E candidate**，还不能把约 2% operator 收益直接等同于可发布的模型收益。

后续 R15 证明 accepted speculative KV relocation 的 cross-owner 实现对
`topk>1` tree compaction 确有缺口，底层 raw-FP8 搬运测试也通过；但生产 DSPARK
配置的 `speculative_eagle_topk=1`，该调用点根本不会执行。locdebug 四节点日志中
没有任何 relocation trace，说明 smoke9/10 的改善只是运行方差，不能归因给
R15。进一步关闭 DSPARK 后，target-only DCP8 的四个 DP 仍产生四种输出 hash，
第一次偏离 reference 分别在 token 4、13、19、105；剩余根因因此位于 K3 target
DCP8 KV/attention 集成，而非 speculative accepted-KV commit。

R16 曾定位到一个真实但未命中当前 K3 路径的 allocator 缺口：普通 MLA DCP allocator 使用
`64*8=512` 的原子逻辑页，Kimi hybrid SWA allocator 却仍按 64-token 小页分配；
KV 写入再执行 `loc//8`，FlashMLA block table 同时按 512 采样。页复用/碎片后，
一个 DCP logical page 不再保证映射到连续的 local 64-token page。R16 将 hybrid
allocator 的 size/page 同步扩展到 DCP logical geometry；allocator 回归 2 passed，
但生产日志明确为 `hybrid_ssm=True, hybrid_swa=False`，smoke17 仍 0/4 exact。
因此 R16 作为独立修复保留，对当前 K3 blocker **rejected**。

R17--R22 最终把剩余问题定位到 SGLang prefill KV 写入，而不是 SM90 DCP
FlashMLA 的 merge 数学。旧 `TritonAttnBackend._set_kv_buffer()` 先把 MLA 的
logical location 除以 DCP size；`MLATokenToKVPool.set_kv_buffer()` 随后又按
已经除过的 location 做 owner filter 和第二次除法。8K prefill 因此只有
**553/8001** 个 global RoPE KV row 非零。R22 让 MLA pool 独占这次
logical-to-local translation，MHA 的原 masked local write 保持不变。修复后
KV 非零覆盖恢复为 **8001/8001**，target-only Graph 与 DCP1 参考连续
**12/12 逐 token exact**，DSPARK smoke **8/8 exact**。

正式 R22-run39 三轮 8K/1500 的 TPOT mean/median 为
**32.60/32.46 ms**，输出吞吐为 **193.63/194.79 tok/s**，active-request 加权
acceptance rate 为 **80.54%/80.69%**，accept length 为 **6.639/6.650**。
相对 R14-P10-run37，mean TPOT 改善 **11.74%**、输出吞吐提高 **14.69%**、
acceptance 提高 **6.68 pp**；相对 DCP1 control，mean TPOT 改善 **16.01%**、
输出吞吐提高 **16.06%**，acceptance 还高 **0.45 pp**。因此 R22 的
**精度修复与相对性能 accepted**；但 TTFT mean 为 **10.01 s**、TPOT mean 为
**32.60 ms**，仍未达到 5 s / 20 ms 绝对 SLO，不能标记为整套部署发布接受。

## ECS 环境硬约束

本阶段只能使用以下四台 ECS。SSH 必须使用公网 IP 和
`/Users/bytedance/perfTest.pem`；私网 IP 只用于节点间 NCCL/GLOO 和
`--dist-init-addr`。禁止使用 K8s 生成正式接受数据。

| `node_rank` | 公网 IP（SSH） | 私网 IP（分布式） | 本报告用途 |
| ---: | --- | --- | --- |
| 0 | `101.126.12.182` | `192.168.44.92` | rendezvous、HTTP、benchmark、8-GPU 回归 |
| 1 | `115.191.51.207` | `192.168.44.91` | DP1 / TP8--15 |
| 2 | `115.191.21.96` | `192.168.44.93` | DP2 / TP16--23 |
| 3 | `115.191.2.23` | `192.168.44.90` | DP3 / TP24--31 |

`.92` 为 8 x NVIDIA H20（SM90/NV18），容器为 `kimi-dcp-20260826`，隔离
源码树为 `/data00/kimi-dcp/FlashMLA-DCP-r14p6-ecs-20260827`。

## Revision 数据

固定 B1/Q8/H128/D576/Dv512、FP8 cache、CUDA Graph、20 warm-up + 200
iterations。`critical` 是 8 个 rank 中最慢者。P7 raw 行只用于说明 kernel 本体
代价；正式 accept/reject 使用 P7+copy consumer-ready 对照。

| revision_id | commit_info | shape | 修改点 | critical performance | CP1 | correctness | 结论 |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| R14-P7-raw | uncommitted P7 | C1216/KV8192 | full-H peer-RS，不计 compact copy | 0.133964 ms | 0.129385 ms | bitwise | 非消费者口径 |
| R14-P7-copy | uncommitted P7 | C1024/KV8192 | full-H peer-RS + 显式 H/8 copy | 0.128682 ms | 0.113358 ms | bitwise | consumer-ready baseline |
| R14-P10 | archive `55ae0bb8...20fc` | C1024/KV8192 | peer-RS 直接写 compact H/8 | **0.125432 ms（-2.53%）** | 0.113398 ms | bitwise；max abs 0 | **accepted** |
| R14-P7-copy | uncommitted P7 | C1216/KV8192 | full-H peer-RS + 显式 H/8 copy | 0.137522 ms | 0.129430 ms | bitwise | consumer-ready baseline |
| R14-P10 | archive `55ae0bb8...20fc` | C1216/KV8192 | peer-RS 直接写 compact H/8 | **0.134721 ms（-2.04%）** | 0.129405 ms | bitwise；max abs 0 | **accepted** |
| R14-P7-copy | uncommitted P7 | C1216/KV9500 | full-H peer-RS + 显式 H/8 copy | 0.137453 ms | 0.129736 ms | bitwise | consumer-ready baseline |
| R14-P10 | archive `55ae0bb8...20fc` | C1216/KV9500 | peer-RS 直接写 compact H/8 | **0.134879 ms（-1.87%）** | 0.129663 ms | bitwise；max abs 0 | **accepted** |

P10 raw kernel 会比 P7 raw 慢约 0.6%--1.2%，因为 compact index/store 进入了
reduce-scatter kernel；但它删除了下游必需的 full-H -> H/8 copy。若忽略这次
copy，会得到错误的性能结论。

## 容量边界

每个 rank 固定一个 `chunk_size` token 的 canonical chunk，总容量为
`8 * chunk_size`：

- C1024 容量正好 8192，只能覆盖 8K prompt，decode 第一个新 token 就越界；
- C1216 容量 9728，可覆盖 8192 prompt + 1500 output，并余 36 token；
- C2048 容量 16384，但 KV8192/KV9500 的 critical 分别为 0.173222/
  0.186901 ms，明显不适合作为当前 8K 测试候选。

因此四节点 E2E 候选固定为 C1216，而不是 operator 表面更快但容量不足的
C1024。

## 8-GPU workspace 回归

工作区 API 负责 token-cyclic pack、固定 chunk leaf、exact peer tree 和 compact
输出。首次回归发现 symmetric staging 的底层存储为 `uint8`，Python 调用 pack
前漏了 FP8 dtype view；修复为 `.view(torch.float8_e4m3fn)` 后，在 `.92` 运行：

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=8 -m pytest -q \
  tests/test_dcp8_canonical_workspace.py
```

结果为 8 个 rank 全部 `1 passed`，每个 rank 约 6.67 秒。测试同时覆盖 global
KV 8192 和 9692，逐 rank compact output/LSE 与 CP1 canonical reference 均
`rtol=0, atol=0`。

E2E 方差出现后，又在 `.91`、`.93`、`.90` 使用同一个已安装 wheel 和 SHA256
为 `86e8b9eb...e49726` 的测试文件逐节点复跑；三台机器的 8 个 rank 也全部
`1 passed`，耗时范围分别为 6.44--6.58、6.82--6.94、6.49--6.62 秒。至此四台
ECS 的单节点 DCP8 workspace 都覆盖 KV8192/9692 且 bitwise 通过，排除了单台
节点 peer operator 错误；剩余 E2E 方差定位到 SGLang 请求/Graph/状态集成层。

## 四节点 8K Kimi-K3/DSPARK E2E

生产接入使用 FlashMLA archive `55ae0bb8...20fc`、wheel
`ea3f5027...9c57`、SGLang effective overlay `df34f3a5...212f` 和启动脚本
`88c06949...b91`。接入 smoke 依次修复了两个 Graph 边界：workspace 的 Q head
数必须使用 DCP 聚合后的 `4 * 8 = 32`，以及 canonical block table 必须使用
预分配的连续 Graph buffer，不能返回带大 stride 的窄 slice。v3 随后完成全部
target-verify/draft CUDA Graph 捕获、HTTP health 和真实短生成。

正式 workload 固定为 8000 input、1500 output、16 prompts、RR=1、MC=8、
temperature=0；每个 rep 前 `/flush_cache` 的后端日志均确认
`Cache flushed successfully!`。DCP1 只切换 `--dcp-size 1` 和
`SGLANG_DCP_CANONICAL_PEER=0`，其余 wheel、overlay、模型和 DSPARK 配置相同。
SGLang `bench-serving` 当前打印的 `Accept length` 只来自
`server_info.internal_states` 中第一个返回的 DP，且该列表顺序跨 rep 不稳定，
不能作为 DP4 全局 acceptance。本报告沿用 R13 口径：在每轮 UTC 窗口内收集
四份 `Decode batch` 日志，并按 `#running-req` 对 accept length/rate 加权。
分析器和逐 rank 明细分别保存为 `analyze_acceptance.py`、
`acceptance_summary.csv` 和 `acceptance_by_rank.csv`。

| revision_id | commit_info | rep | TPOT | output throughput | accept length / rate | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| R14-P10-run37 | archive `55ae0bb8...20fc`; overlay `df34f3a5...212f` | 1 | 41.85 ms | 140.48 tok/s | 5.548 / 65.09% | completed |
| R14-P10-run37 | 同上 | 2 | 35.54 ms | 177.38 tok/s | 6.486 / 78.37% | completed |
| R14-P10-run37 | 同上 | 3 | 33.43 ms | 188.60 tok/s | 6.466 / 78.10% | completed |
| R14-P10-run37 | 三轮 mean / median | - | **36.94 / 35.54 ms** | **168.82 / 177.38 tok/s** | **6.167 / 6.466；73.86% / 78.10%** | **E2E candidate** |
| R14-P10-control-run38 | 同 wheel/overlay；DCP1 | 1 | 39.12 ms | 163.57 tok/s | 6.613 / 80.17% | completed |
| R14-P10-control-run38 | 同上 | 2 | 38.74 ms | 167.89 tok/s | 6.591 / 79.87% | completed |
| R14-P10-control-run38 | 同上 | 3 | 38.60 ms | 169.02 tok/s | 6.615 / 80.23% | completed |
| R14-P10-control-run38 | 三轮 mean / median | - | **38.82 / 38.74 ms** | **166.83 / 167.89 tok/s** | **6.606 / 6.613；80.09% / 80.17%** | baseline |

P10 DCP8 相对 DCP1 的 mean/median TPOT 分别改善 **4.84%/8.26%**，输出吞吐
提高 **1.19%/5.65%**；但 acceptance rate 分别下降 **6.23/2.07 pp**。
DCP8 第一轮的 DP2 加权 acceptance 只有 41.61%，把全局拉到 65.09%；后两轮
全局为 78.37%/78.10%。四台 ECS 的 8-GPU operator 回归都 bitwise 通过，因此
下一轮重点是逐 DP 的同请求复现和 first-use/warm-up 状态，而不是继续修改已通过
的 SM90 fixed-tree 数学路径。

与 R13 DCP8 三轮基线相比，P10 的 mean/median TPOT 改善 0.83%/6.33%，输出
吞吐提高 1.12%/8.75%；active-request 加权 acceptance mean/median 提高
**1.76/6.87 pp**，weighted accept length 提高 **1.98%/8.07%**。P10 已同时
改善当前 DCP8 性能与精度，但尚未消除相对 DCP1 的剩余 gap。

## R15：跨 owner accepted-KV relocation 诊断（当前路径不接受）

### 根因与修改

R12 的 `MLATokenToKVPool.move_kv_cache` 使用：

```python
owned = (tgt_loc % dcp_size == dcp_rank) & (src_loc % dcp_size == dcp_rank)
```

这要求 accepted source slot 与 committed target slot 恰好归属于同一个 DCP
rank。对 `topk>1` 的 speculative tree compaction，source/target owner 可以
不同，旧逻辑会直接丢弃这部分 KV。

R15 将 source owner 与 target owner 分开处理：先把所有 MLA layer 的 source
rows stack 为一个 raw `uint8` buffer，用一次 DCP `all_gather_into_tensor` 收齐；
随后按 source owner 选择原始 FP8 bytes，仅由 destination owner 写入 compact
physical rows。逻辑地址上界也改为
`(self.size + self.page_size) * dcp_size`。这保证 FP8 storage bitwise 搬运，且
每次 relocation 只做一次 collective，而不是每 layer 一次。低层修复本身有效，
但 `eagle_worker_common.py` 只在 `speculative_eagle_topk > 1` 时调用
`move_accept_tokens_to_target_kvcache`；当前 DSPARK 的 topk 为 1，所以它不是本轮
生产请求的执行路径。

| revision_id | commit_info | 修改点 | acceptance / correctness | performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R14-P10-DP1 | smoke6；同 wheel/overlay，DCP1 | 确定性 8K、greedy、2 cycles x 4 requests | **8/8 exact；mean/min 100%；1 hash** | 非正式 smoke | 对照通过 |
| R14-P10-canonical | smoke5；`memory_pool.py` `809c881c...47bc` | 旧 owner-local relocation + canonical peer | **5/8 exact；mean 72.76%；min 16.33%；4 hashes** | 非正式 smoke | rejected |
| R14-P10-cyclic | smoke8；同旧 relocation | 关闭 canonical，恢复 cyclic DCP8 | **4/8 exact；mean 69.38%；min 9.68%；5 hashes** | 非正式 smoke | 排除 canonical-only 根因 |
| R15-unit | patched file `8b4157bb...d86`；patch `35029b9a...ad1` | raw-byte all-gather + destination-owner writeback | 8 ranks x 3 layers；8 cross-owner + 8 same-owner pairs；failure **0** | 未计时 | low-level accepted |
| R15-smoke9 | 同 patched file；canonical P10 | 首次 2 cycles x 4 requests | **7/8 exact；mean/median 88.94%/100%；min 11.55%；2 hashes** | 非正式 smoke | variance；不能归因给 R15 |
| R15-smoke10 | 同服务、再次 flush；3 cycles x 4 requests | post-flush 稳定性复测 | **10/12 exact；mean/median 87.07%/100%；min 15.57%；3 hashes** | 非正式 smoke | rejected |
| R15-ring32-bs12 | diagnostic script `7e8132cb...25b` | ring 32；Graph max-bs 12 | **8/12 exact**；失败 rate 12.70%/13.95%/16.45%/20.68% | 非正式 smoke | ring-length 假设 rejected |
| R15-sync | patched file `78a0e3a1...bbe3` | relocation 后显式 `torch.cuda.synchronize()` | **7/12 exact**；失败 rate 14.63%--43.54% | 非正式 smoke | stream-visibility 假设 rejected |
| R15-locdebug | patched file `d0cede03...7ffe` | relocation caller trace | **7/8 exact**；四节点日志 **0 trace** | 非正式 smoke | 证明 topk1 不调用 R15 |
| target-only-DCP8 | 关闭 DSPARK；canonical P10；Graph | 1 cycle x DP4，8K/128 | **0/4 exact；4 hashes**；first bad 19/105/4/13 | 29.27--32 s/request | 根因在 target DCP8 路径 |
| target-only-eager | 同上；关闭 CUDA Graph | 首次 decode 报 workspace 未初始化 | 无生成数据 | 无 | 暴露 workspace 生命周期缺陷 |

smoke9 + smoke10 合计 **17/20 exact（85%）**，请求级 acceptance rate
mean/median 为 **87.82%/100%**；但 locdebug 证明 relocation 在这些请求中没有
执行，因此这组数字只能描述服务方差，不能描述 R15 收益。关闭 DSPARK 后
target-only DCP8 仍 0/4 exact，进一步排除了 accepted-KV relocation、ReplaySSM
commit 和 speculative acceptance 本身。由于 correctness gate 未通过，本轮没有
运行 R15 8K/1500 正式性能；可引用性能仍只有 R14-P10 operator 与 run37。

ring=32 不是 OOM，也不是 ring=16 被静态硬编码。原 graph max-bs 16 启动失败是
因为 ring32 降低了 draft attention backend 的有效 request capacity 到 14，却仍
捕获 bs16，最终在 Triton `qo_indptr` 更新时出现 target 长 15、source 长 17 的
shape mismatch。将 graph max-bs 降到 12 后服务正常就绪，但只有 8/12 exact，
所以 ring length 不是剩余根因。

### R16：Kimi hybrid DCP allocator 原子页

普通非 hybrid 分支已经使用 `size*dcp_size` 和 `page_size*dcp_size` 创建 paged
allocator。Kimi-K3 的 hybrid SWA 分支却把物理 size/page 原样传给 allocator；
与此同时 MLA KV write 按 owner 过滤 logical loc 并执行 `loc//dcp_size`，FlashMLA
block table 用 `PAGED_SIZE=64*dcp_size`。这三个约定不一致会把一个应当原子分配的
512-token logical page 拆成八个独立 64-token allocator pages。

| revision_id | commit_info | 修改点 | correctness | performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R16-unit | patch `f94bf55d...3ac0`；SGLang working tree uncommitted | hybrid full/SWA size/page 按 DCP8 扩展；resize 保持 logical geometry；新增 allocator regression | rank0 容器 **2 passed**；四 ECS source SHA 一致 | 未计时 | 独立修复 accepted |
| R16-smoke17 | 同 patch；canonical Graph；target-only 8K/128 | 四 DP 单轮 | **0/4 exact；4 hashes**；first bad 43/24/18/82 | 28.96--31.23 s/request | 当前 K3 路径 rejected |

生产日志明确显示 `hybrid_swa=False`，说明 K3 当前是 hybrid SSM、full-attention
KV 走 generic paged allocator，R16 修改没有进入生产实例。另一个独立缺陷是
canonical workspace 只在 `init_cuda_graph_state()` 分配，导致关闭 Graph 后首个
decode 直接报 workspace 未初始化；该生命周期修复顺延为 R17。

## R17--R22：prefill KV 双重 DCP 映射根因与修复

R17 先把 canonical workspace 从 CUDA Graph 初始化解耦，使 eager decode 也能
安全分配和复用 workspace。eager 服务随后能完整生成，但跨 DP 输出仍不一致，
说明生命周期缺陷是真实的独立问题，却不是精度 blocker。R18 关闭 Q replication
后仍得到四种 hash；R19 又在同一个 DCP group 的八个 rank 上证明聚合后的 FP8 Q
bitwise identical，因此“八个 rank 在合并不同 query”这一假设被拒绝。

R19 保存首个 decode step 的实际 Q、selected KV、block table 和 output，并在
rank0 GPU 上用同一份输入重建 CP1 FlashMLA。canonical DCP8 与 CP1 output
**bitwise equal，max abs=0、relative L2=0**。这一步把 SM90 canonical
pack/merge/attention 数学路径从嫌疑中排除。R20 的 DCP1 trace 则显示八个 TP
rank 的 KV 完全一致，而修复前 DCP8 与 DCP1 的 KV cosine 只有约 0.266、
relative L2 约 0.964；问题发生在进入 FlashMLA 之前。

R21 恢复生产默认的 Q replication 后，DCP8 与 DCP1 的 real Q cosine 已为
0.999952、relative L2 为 0.01019，但 KV 仍只有 **553/8001** 个 global RoPE
位置非零。每个 DCP local rank 只写入 69 个旧 token；rank0 额外写入 decode 新
token，所以 global nonzero runs 为 `0--551` 和 `8000`。代码审计定位到双重映射：

1. `TritonAttnBackend._set_kv_buffer()` 先把
   `forward_batch.out_cache_loc // dcp_size` 传入 hybrid KV pool；
2. hybrid MLA 分支丢弃 `dcp_kv_mask`，把已除过的 loc 交给
   `MLATokenToKVPool`；
3. `MLATokenToKVPool.set_kv_buffer()` 再按该 loc 做 owner filter，并再次
   `loc // dcp_size`。

R22 在 backend 边界按 pool 类型分流：MLA 直接传 logical `loc_info`，由 MLA
pool 唯一执行 owner filter 和 logical-to-compact-local translation；MHA 保持
既有的 pre-translated loc + `dcp_kv_mask` 行为。四 ECS active source SHA256
均为 `0af3260592c00de2a86bba71a1871a7534b9446757eabd288cfdcb62b1da1bca`；
本地可审阅实现与测试分别为
`/Users/bytedance/Repos/sglang/python/sglang/srt/layers/attention/triton_backend.py`
和
`/Users/bytedance/Repos/sglang/test/registered/unit/layers/attention/test_triton_backend_dcp_kv_write.py`。

| revision_id | commit_info | 修改点 | correctness / acceptance | performance | 结论 |
| --- | --- | --- | --- | --- | --- |
| R17 | uncommitted SGLang lifecycle fix | canonical workspace eager/Graph 共用生命周期 | eager 可运行；跨 DP 仍不一致 | 非正式 smoke | 独立修复 accepted；当前 blocker rejected |
| R18 | smoke20；关闭 Q replication | 不复制 query projection，直接使用 DCP gather Q | **0/4 exact；4 hashes**；组内 gathered Q 后续证实 bitwise | 29.60--31.43 s/request | rejected |
| R19 | no-Q-rep eager tensor trace | 实际首步 Q/KV/table/output + 同输入 CP1 GPU reconstruction | canonical DCP8 vs CP1 **bitwise；max abs 0；relL2 0** | diagnostic only | SM90 operator accepted；排除 merge 偏差 |
| R20 | DCP1 eager tensor trace | 与 R19 逐 tensor A/B | DCP1 4/4 exact；DCP8 KV cosine 0.266、relL2 0.964 | diagnostic only | blocker 位于 pre-attention KV |
| R21 | default Q replication trace | 检查 global/local KV 非零覆盖 | Q cosine 0.999952；KV 仅 **553/8001** 非零 | diagnostic only | 双重 DCP loc 映射根因 |
| R22-unit | local source `789d7912...e7dd`；test `d58aa686...8660` | MLA 只在 pool 中映射一次；MHA 行为不变 | rank0 容器 **2 passed** | CPU unit | accepted |
| R22-eager | active source `0af32605...1bca` | 修复后的首步 tensor trace | DP4 **4/4 exact**；KV **8001/8001** 非零；output cosine 0.999946、relL2 0.018674 | 非正式 smoke | accepted |
| R22-Graph-smoke27 | 同 source；clean FlashMLA backend `7157b824...c6b` | target-only 8K/128，3 cycles x DP4 | 与 DCP1 reference **12/12 逐 token exact；1 hash** | 28.75--29.20 s/request | correctness accepted |
| R22-DSPARK-smoke28 | 同 source；8K/256，2 cycles x DP4 | 恢复 speculative 全路径 | **8/8 exact；1 hash；accept 8.0/100%** | 7.63--10.39 s/request | chain smoke accepted |

合成 smoke 的 100% acceptance 只用于确认 DSPARK 链路健康；正式接受结论使用与
run37/run38 完全相同的 random workload：8000 input、1500 output、16 prompts、
RR=1、MC=8、temperature=0，每轮前服务日志确认 `Cache flushed successfully!`。
接受率继续按四个 DP 的 `Decode batch`、以 `#running-req` 加权。

| revision_id | commit_info | rep | TPOT | output throughput | accept length / rate | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| R22-run39 | SGLang `0af32605...1bca`；FlashMLA R14-P10 wheel `ea3f5027...9c57` | 1 | 33.49 ms | 189.84 tok/s | 6.656 / 80.71% | completed |
| R22-run39 | 同上 | 2 | 32.46 ms | 194.79 tok/s | 6.650 / 80.69% | completed |
| R22-run39 | 同上 | 3 | 31.86 ms | 196.25 tok/s | 6.611 / 80.23% | completed |
| R22-run39 | 三轮 mean / median | - | **32.60 / 32.46 ms** | **193.63 / 194.79 tok/s** | **6.639 / 6.650；80.54% / 80.69%** | **fix accepted；SLO pending** |

R22 相对 R14-P10-run37 的 mean/median TPOT 改善 **11.74%/8.67%**，输出吞吐
提高 **14.69%/9.82%**，acceptance 提高 **6.68/2.59 pp**。相对 DCP1 control
run38，mean/median TPOT 改善 **16.01%/16.21%**，输出吞吐提高
**16.06%/16.02%**，acceptance 还提高 **0.45/0.52 pp**。三轮 acceptance
范围仅 80.23%--80.71%，旧 run37 首轮 65.09% 的异常不再出现。
不过三轮 TTFT 为 9.73/9.86/10.44 s，TPOT 为 33.49/32.46/31.86 ms，三轮均
未同时达到 5 s TTFT 与 20 ms TPOT。R22 解决的是 DCP 精度和相对性能 blocker；
剩余绝对 SLO 需要继续优化完整 K3 serving 路径。

机器可读性能和 on-node 日志聚合分别保存为
`e2e-run37-run38/r22_run39_summary.csv` 与
`e2e-run37-run38/r22_acceptance_onnode_aggregates.csv`。原始服务日志因可能包含
内部信息没有复制到本地；每台 ECS 只在本机解析对应 UTC 窗口并返回上述匿名聚合
四元组。四份 result JSON、Graph smoke27 和 DSPARK smoke28 已保存在同一目录。

### 环境释放状态

R15 结束时曾释放四台 ECS；用户随后明确要求继续，当前 R22 DCP8/DSPARK
run39 服务仍在四台 ECS 上运行。所有权重、源码、patch、结果与日志保留在
`/data00/kimi-dcp`；没有删除容器或持久文件。R17--R22 的诊断、正确性和正式
性能数据只使用本报告指定的四台 ECS，没有使用 K8s。

## 源码与数据保存

最终源码归档：
`source-archives/flashmla-r14p10-delta-20260827-final.tar.gz`，SHA256：
`55ae0bb882293ec93c9f125a726ec3200780b91696afcc28cca515bf8f3020fc`。
相同归档也保存在 `.92` 的
`/data00/kimi-dcp/results/flashmla-r14p10-delta-20260827-final.tar.gz`，远端哈希
已核对一致。

本目录保存 P10、P7+copy、C1216 和 C2048 的全部 operator JSON；
`e2e-run37-run38/results/` 保存六组 8K benchmark 原始 JSON/JSONL/stdout，
`e2e-run37-run38/service-logs/` 保存四节点 DCP8/DCP1 的八份完整服务日志，
`e2e-run37-run38/workspace-tests/` 保存 `.90/.91/.93` 的新增 8-GPU 回归日志，
`e2e-run37-run38/summary.csv`、`acceptance_summary.csv` 和
`acceptance_by_rank.csv` 是机器可读汇总。所有正式 E2E 数据均来自上述四台
ECS，没有使用 K8s。

R15 额外保存：`source-archives/memory_pool-r15-cross-owner-20260827.py`
（SHA256 `8b4157bb2721bd4d53b954bb1cd226418077d4abdce5f25de4336416ed597d86`）、
`source-archives/sglang-r15-cross-owner-kv-relocation.patch`（SHA256
`35029b9aa6595044a8f632e1618f46dbc6534fb0ea32d14372aafd3555d2dad1`）、
底层测试及两个确定性 JSON，以及 smoke9/ring32 的四节点完整日志。可整体恢复的
R15 诊断归档为
`source-archives/sglang-r15-cross-owner-diagnostics-20260827.tar.gz`，SHA256
`a735331cd6dd5e77968ac30abae6b9b196b372f810d675e0cfc6061997d349c7`。

R16 allocator 补丁保存为
`source-archives/sglang-r16-hybrid-dcp-atomic-page-20260827.patch`，SHA256
`f94bf55d363243b9d12b2781291635130ffe34e4a3924e99846f1a31994a3ac0`。

R22 的可继续开发版本保存在 `/Users/bytedance/Repos/sglang`：backend SHA256
`789d791201a983b7a5233d9648e8daf4210b32270b6c8c16f5ac1d046bede7dd`，CPU
regression SHA256
`d58aa68659ee5f94a30662f8199388cd28d3bd0aae7a6c3945210da5e8918660`。
可独立恢复的 source + regression patch 为
`source-archives/sglang-r22-mla-prefill-single-dcp-translation-20260827.patch`，
SHA256 `141cce72e49759da0a65263e4d1b87d98e1ce51e9be8e22bc75eb92984caa231`，
并已通过 `git apply --check --reverse` 验证与当前本地修改一致。
四台 ECS 的 active backend 和
`/data00/kimi-dcp/results/triton_backend-r22.py` 均为
`0af3260592c00de2a86bba71a1871a7534b9446757eabd288cfdcb62b1da1bca`；更新前
文件保留为 `.pre-r22-20260827`。R19--R22 的 raw tensor trace 只保留在 rank0
`/data00/kimi-dcp/results/dcp-r19-noqrep-eager-smoke22`、
`dcp-r20-dcp1-eager-smoke23`、`dcp-r21-qrep-eager-smoke24` 和
`dcp-r22-qrep-eager-smoke25`，没有复制到本地。
