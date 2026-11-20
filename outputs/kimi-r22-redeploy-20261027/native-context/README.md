# Kimi-K3 原生上下文恢复（2026-10-27）

**当前状态（2026-10-28）：trial19部署验收及三档缓存吞吐评测已完成，四台ECS服务保持运行。**
已恢复模型原生上下文配置1048576，TP32 / DP4 / DCP8 / EP32，P/D共置，
DSPARK block7。当前使用`ag_rs` DCP通信后端，保留CUDA Graph24；
全服务请求上限96（配置24/DP），实测日志中的解码峰值为21/DP。
这不是96条1M请求同时驻留的承诺，也没有完成整模型1M长请求测试。

| 验收项 | trial19结果 |
| --- | --- |
| 96并发客户端，1024输入/512输出 | 96/96完整返回，四DP各完成24条，墙钟89.08s |
| 四DP，8000输入/256输出 | 4/4完整token与历史R22基线完全一致 |
| 131072输入/32输出 | 完整返回且匹配此前128K基线，194.60s，无retraction |
| 连续健康检查 | 8/8 HTTP 200 |
| 配置与KV容量 | context1048576，local149312 × DCP8 =1194496 token/DP |
| 四节点源码/环境 | 9项active文件哈希一致，32进程EAGER与启动预热核验通过 |

8K和128K为重复token合成smoke，接受长度8、接受率100%，不能外推真实业务。
128K请求含8192 token前缀缓存命中；以上耗时不是正式性能A/B或SLO结果。
使用CUDA skill进行现场取证后，定位到a2a图回放附近的推进分歧，改用已有ag_rs
路径完成验收。这是已验证的部署替代方案，不等同已经证明a2a挂起的底层根因。

服务入口：`http://192.168.44.92:8080/v1`，模型名`Kimi-K3`。
本机请走原跳板机/公网IP建立SSH隧道；未验证公网8080直连。
SSH退避为2/4/8分钟、最多3次；不改变认证方式，不自动重放未知结果的变更命令。

实际命令：`bash /workspace/restore/native-context/validate_trial.sh trial19`，退出0；
本地审计：`python3 verify_trial.py --trial trial19 --expected-dcp-backend ag_rs`，PASS。
结果为本目录`trial19-*.json/.log/.sha256`，四节点服务日志在`trial19-ready/`。
环境：4×8 NVIDIA H20（SM90），PyTorch2.11.0+cu130、CUDA runtime13.0；
复用历史FlashMLA wheel，本次没有重新编译或调整编译架构覆盖。

## 缓存命中吞吐评测（已完成）

完整过程和原始文件索引见 [CACHED_CONTEXT_BENCHMARK.md](CACHED_CONTEXT_BENCHMARK.md)。
使用相同 ShareGPT 构造前缀逐 DP 预热，逐请求核验命中；服务配置保持 trial19。
按附件的平均 TPOT < 50ms 口径，四节点/32 卡整个服务的达标复测结果如下：

| 缓存前缀 | 客户端并发 | 达标输出 tok/s | TPOT mean / P99 ms |
|---|---:|---:|---|
| 16K | 8 | 173.30 | 42.31 / 68.24 |
| 32K | 4 | 97.55 | 38.05 / 50.62 |
| 128K | 已测 1/2/4 均未达标 | — | — |

16K/32K 仅平均 SLO 通过，P99 均超出 50ms。128K/MC4 复测为 45.56 output tok/s、
平均 TPOT 83.94ms；MC2 为 21.61 / 91.58ms，MC1 为 11.14 / 89.51ms，均不符合 SLO。
实际输入为缓存前缀加 16-token 后缀，输出固定 1024；三档最低命中率为
99.9024% / 99.9512% / 99.9878%，全部正式请求命中完整长前缀。
14 轮正式测量、794 条请求、813056 个输出 tokens 已完整备份并逐请求复算；
16K 原汇总序列化故障与离线恢复在详细记录中如实保留，未伪造完成标记。
所有客户端已退出；最终健康检查 8/8 HTTP 200，原生 1M 上下文、全局 96 请求上限不变。
服务与 kernel 未改变，仍为 MoE A2A=none、overlap 关闭的稳定路径，不代表 DeepEP 优化性能。

## 目标与配置

### ECS连接退避（2026-10-28）

仅使用以下四台ECS的公网IP，经原跳板机和密钥登录，禁止使用K8s：

| rank | 公网IP（SSH） | 私网IP（集群通信） |
| --- | --- | --- |
| 0 | 101.126.12.182 | 192.168.44.92 |
| 1 | 115.191.51.207 | 192.168.44.91 |
| 2 | 115.191.21.96 | 192.168.44.93 |
| 3 | 115.191.2.23 | 192.168.44.90 |

连接脚本为父目录`ecs_ssh.py`，固定`-J jumpecs-lf.byted.org`和
`-i /Users/bytedance/perfTest.pem`。对临时网络错误或该跳板机间歇性GSSAPI拒绝，
先等待2分钟，再4分钟、8分钟，最多重试3次。只读查询/连接探测使用
`--retry-safe`；变更命令不自动重放，连接中断后先查询远端状态。
主机身份变化、host-key验证失败不重试。重试耗尽后停止并报告，不切换认证路径。
网络原因尚属推测，不能将有效票据下的每次认证拒绝都断言为网络故障。
`python3 -B ../test_ecs_ssh.py`离线5项测试通过，覆盖成功重试、重试耗尽、
变更命令不重放、主机身份/非指定认证错误不重试；没有实际sleep或SSH。
若前一个SSH/SCP已出现已确认可重试的错误，可加`--after-failure`：先等2分钟
再探测，而不是立即重试；此模式总共最多3次探测，含随后4/8分钟退避。
2026-10-28 02:31日志回传再次出现GSSAPI拒绝，已实际启动该退避探测。

主模型 `text_config.max_position_embeddings` 和 DSpark draft 的
`max_position_embeddings` 均为 **1,048,576**。恢复该原生上下文，不再使用
此前 9728-token 的实验上限。仍只在原四台 ECS 上运行，通过公网 IP、跳板机
`jumpecs-lf.byted.org` 和密钥 `/Users/bytedance/perfTest.pem` 登录；不使用 K8s。

- TP32 / DP4 / DCP8 / EP32，P/D 共置，DSPARK block7。
- 使用常规 token-cyclic DCP8 FlashMLA，关闭固定长度 canonical peer workspace。
- FP32 local attention output，合并后再转换；Mamba 状态保持 FP32。
- 保留 R22 prefill KV 映射修复与 R12 draft DCP1 隔离。
- `context-length=1048576`。
- `max-running-requests=96`（全服务）；目标每 DP 24。
- `max-mamba-cache-size=384`（全服务），分配到每 DP 96 状态槽，按 4 槽/请求。
- `cuda-graph-max-bs=24`；`mem-fraction-static=0.86`。
- `chunked-prefill-size=2048`（命令行），运行时每 DP 为 512。
- `CUDA_MODULE_LOADING=EAGER`，启动期DSPARK辅助算子与hidden提交预热。
- 当前DCP通信`ag_rs`，`dcp_replicate_q_proj=false`；FP32校正与归并后统一转换保留。
- 当前关闭overlap，Mamba使用`extra_buffer`；Graph保持开启，未使用target-eager gate。
- 保留`NCCL_GRAPH_MIXING_SUPPORT=1`与`NCCL_LAUNCH_ORDER_IMPLICIT=1`；
  二者单独未解决a2a停滞，不能归因为已独立证明有效的优化。
- MoE runner仍为Marlin，`moe_a2a_backend=none`，不是peer MoE/DeepEP性能方案。

原生单请求长度与并发上限是不同限制。上述并发数不表示 96 条 1M 请求可同时
驻留；真实并发还取决于 KV 总量、请求长度和可用状态槽。

## 最小调度修复

原始版本的 DCP allocator 已提供 `local_tokens × DCP` 的全局地址空间，
PrefillContext 也按全局容量调度，但两个入口仍使用本地 token 数限制请求：

1. `tp_worker.py` 的初始化校验和 `get_worker_info()`：单请求上限改为
   `min(context_len - 1, local_token_capacity × DCP - 1)`。
2. `scheduler.py` 的输出长度预算：与 PrefillContext 一致，使用全局 DCP 容量。
3. `engine.py` 保留显式 `CUDA_MODULE_LOADING`：将无条件赋值AUTO改为
   `setdefault`，未指定时仍保留原默认值。`test_env_loading.py` CPU检查通过。

原源码保留只读，部署副本位于各节点
`/data01/kimi-r22-20261027/native-context/python`。未修改主仓库的其他工作树文件，
未重编译 FlashMLA。旧启动脚本、旧服务日志仍保留在父目录。

## 已完成验证

以下按轮次保留历史证据；旧轮次的失败和当时计划不代表上方当前运行状态。

首轮尝试 `mem_fraction=0.89`、prefill chunk2048/DP：服务启动和 Graph32
捕获完成，但第一个 8K 请求在 prefill DCP correction 的临时 384 MiB 张量
分配时 OOM。该配置拒绝，日志保留在远端 `trial01-memory-089/`。
第二轮保持 BS32 和 FP32 状态，降低静态预算至 0.88、prefill chunk 至1024/DP。
第二轮日志确认每 DP 的实际并发为32，target/draft Graph 覆盖32。
单卡 target KV153664 token，DCP8 全局容量1229312 token/DP；主模型和draft
共用全局地址空间，接口 `max_model_len=1048576`。每 DP128个Mamba状态槽，
保持FP32。真实8K请求已通过，未再触发首轮OOM。

后续步骤：完成128并发测试，并检查健康状态及显存余量。

第二轮128K完整请求通过：131072输入/32输出，服务端报告未截断，接受率100%，
耗时181.24秒，包含8192 token前缀缓存命中；此数值不是正式无缓存性能基准。

第二轮128并发（1024输入/512输出、尾token不同）被拒绝：各DP在prefill填充
后停止推进，没有OOM报错。`py-spy --native`确认TP0/TP1停在
`FinalizeAcceptLens.execute -> triton._init_handles -> cuModuleLoadData`。
保存 `stall-bs32-*.txt` 及 `trial02-bs32-coldload/service-rank*.log`；
停止服务后128客户端均返回连接关闭，不把它们记作成功请求。

第三轮仅增加启动阶段预热：在 `DSparkWorkerV2.init_attention_backends()` 调用
`deploy_dspark_warmup.py`，遍历BS1..32、int32/int64组合，预加载接受长度、
截断长度、bonus gather和输出token辅助算子，并逐元素对照原Torch参考。
不替换生产接受规则，不修改attention kernel，不在热路径添加同步。
独立H20验证通过：BS1..32，共480组整数辅助算子参考检查。
部署文件SHA256：`deploy_dspark_warmup.py` 为
`33e9c304d7961a2f3d4d9253cb5de91c15f50e52b128651f5ba401a149c42490`；
`dspark_worker_v2.py` 为
`9ec7d56970c891571469d020952195178fa25005eafdae553bb755848085645a`。

第三轮32个GPU进程均完成480组检查，但128并发仍停滞。TP0和TP1原生栈
均为 `commit_hidden -> write_target_hidden_kv -> project_target_hidden ->
nn.Linear -> cublasGemmEx -> cuKernelSetAttribute`，没有OOM报错。
这说明已越过此前的整数辅助算子首次加载点，但尚未消除所有首次初始化阻塞。
日志保存于远端 `trial03-helper-warmup/`，原生栈 `stall-bs32-warm-native*.txt`。

第四轮进一步在启动期间预热真实 dense draft 的 hidden→KV 提交，覆盖每DP
BS1..32、每请求8个verify token。使用真实生产方法，KV数据暂写入启动时尚无
请求拥有的1..256槽位，不修改allocator元数据，后续请求正常覆盖；保留0号槽。
检查投影shape和有限值，所有同步均位于启动阶段，热路径与生产计算公式不变。
新文件SHA256：helper
`ac646c6bbcff30b1c104612b8e3bafe89ae98d760858f267f20464742d019067`，worker
`fadd54cd6c41a50490386f2642cc222665ec1f7b7cf7b6791b7d6f3273cf2c7c`。

第四轮32个GPU进程均完成32种hidden提交形状，但并发测试仍未完成。
DP0全部8个rank停在CPU Gloo广播等待；DP1/2/3代表rank停在
`compute_dflash_correct_drafts_and_bonus -> cumprod -> cuLaunchKernel`。
保存 `trial04-hidden-warmup/` 和 `stall-fullwarm-*.txt`。
第五轮增加BS1..32、int32/int64 token输入的greedy正确前缀和bonus精确校验，
预热与生产相同的Torch scan/reduction路径；整数检查总数从480增至544。
生产接受算法仍未修改。

第五轮544组检查通过，128并发随后停在
`_commit_target_mamba_states_after_verify -> bitwise_and -> cuLaunchKernel`。
第六轮统一设置 `CUDA_MODULE_LOADING=EAGER`，并保留已验证的Triton辅助算子
和hidden提交预热，避免只逐一补齐ATen首次加载。其余服务参数不变，重新
核对显存与全局KV容量后再进行并发验收。
此方向与NVIDIA说明的lazy loading/concurrent kernel死锁风险一致，属于
结合本次栈的诊断推断，最终仍以负载复测为准；参见
[CUDA Lazy Loading：Potential Hazards](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/lazy-loading.html#potential-hazards)。
第五轮helper SHA256为
`3887b0894534d9c24c6d8b74fd615a79efe6717be2853484be7e62063fc0791f`。

第六轮启动脚本指定EAGER，仍为153664本地KV token / 1229312全局token，
BS32/DP。但首个并发prefill在Marlin的`intermediate_cache13`申请768MiB
时OOM（GPU0仅剩501MiB）。临时缓冲与该轮token数成正比；第七轮仅将
prefill chunk从1024/DP改为512/DP，保持1M上下文、KV容量与decode BS32。
此调整可能降低长输入prefill效率，不声称预填充性能不变。

**第七轮诊断更正**：虽然脚本指定EAGER，但调度进程`/proc/<pid>/environ`
实际为 `CUDA_MODULE_LOADING=AUTO`。部署版`entrypoints/engine.py:1515`
无条件覆盖了此设置。因此第六、七轮不能作为实际EAGER模式的验证。
第七轮仍停在Mamba提交的ATen bitwise_and加载。
第八轮将该行改为`setdefault`，并在每个GPU进程启动时调用
`cuModuleGetLoadingMode`，要求实际模式为`CU_MODULE_EAGER_LOADING`，否则
立即报错；不再只凭启动脚本判断模式。启动期间的544组整数校验和32种
hidden提交预热保留。

第八轮已部署文件SHA256：

| 文件 | SHA256 |
| --- | --- |
| `engine.py` | `081d1dda19887bbd37419babf6650929c4692f55758a0b51179d922e34bf802c` |
| `deploy_dspark_warmup.py` | `dcf5a51896fd8699f49cb7438a37e196f6b2072af0729c642d885fafb9f629d2` |
| `dspark_worker_v2.py` | `09914dce710db28af203855b05c79a9d88801def2fbe2c38dc138d46118ba8ff` |
| `serve.sh` | `6d375f7c4c0ebcdc013b2a6732592e23a3db35dd79cd439a9fb4fca038318d2a` |

第八轮只读复查结果：四节点各8条driver EAGER确认，KV为136768本地token，
全局1094144，仍高于1M。但启动VLM warmup发生
`dp_gather_replicate -> all_reduce -> ncclUnhandledCudaError: Cuda failure 2 'out of memory'`，
主进程退出，其余节点等待断开的TCPStore。日志归档`trial08-true-eager-bs32/`。
因此第九轮将总并发128→96、每DP Mamba状态槽128→96、Graph32→24，并把
静态内存预算0.88→0.86；优先保留原生上下文，为通信和临时分配留出余量。
这些是部署参数调整，未降低状态精度或更改生产接受算法。

第九轮启动结果：四节点各8条EAGER确认与408组整数精确检查，hidden提交覆盖
BS1..24。实际本地KV149312 token，DCP8全局1194496；Graph完成后可用显存
5.58GiB，启动VLM warmup正常返回。96并发首次decode时主节点空闲显存约
2.5–3.9GiB，未发生OOM，但停止推进。DP1代表rank原生栈为
`commit_kda_replayssm_after_verify -> commit_kda_replayssm_spec_all_layers ->
triton._init_handles -> cuModuleLoadData`，保存在该节点
`native-context/trial09-stall-native.txt`。这属于尚未预热的运行时Triton模块；
实际EAGER已解决此前ATen模块加载，但不代表覆盖动态生成的模块。

待补齐的启动预热方案（尚未实现/部署）：使用真实ReplaySSM tensor布局，
调用既有提交入口，所有state index/track index设为-1、conv step设为-1，
通过已有null-slot/negative-step分支跳过状态读写，预加载fold与conv rollback
模块。已检查本地对应源码的early-return逻辑；恢复登录后须先与部署源码
校验一致，并测试不修改状态，再部署。生产提交规则不变。

第十轮仅增加ReplaySSM提交预热：使用真实pool形状/stride/dtype，覆盖BS1..24、
int32/int64索引与接受长度、track/no-track分支，全部状态索引和conv步骤为-1。
fold在访问状态前返回，conv在读取源缓存前返回；不触碰生产接受公式，不在热路径
添加同步。先以小型非连续conv窗口验证所有9个底层buffer逐元素不变：
H20、PyTorch2.11.0+cu130、CUDA13.0，BF16/FP16各288组，共576组通过。
首轮完整memcheck退出1，报告21个导入期`cuGetProcAddress_v2` INVALID_VALUE；
两种dtype测试本身通过。保留原日志，并以`--report-api-errors no`单独复查访存，
不能把首轮完整sanitizer记为通过。没有重编译FlashMLA，无新增编译flags。
访存专项复查退出0，`ERROR SUMMARY: 0 errors`，576组仍通过。
完整命令为`CUDA_MODULE_LOADING=EAGER compute-sanitizer --tool memcheck
--report-api-errors no --error-exitcode 1 python test_replayssm_warmup.py`。
API错误报告被关闭，但访存错误检测保留；不宣称全部CUDA API检查通过。
日志`trial10-warmup-test.log`、`trial10-warmup-memcheck*.log`。
四节点已校验新helper SHA256
`988ee76a235abdfb7f766f0b8f30ee44cfb4d53d903770196d399997f4615f87`，
worker SHA256
`a277fd1e24e39faa9129bfcfc04a6653e4974ee86d46687de4c0c9acacf96444`。

第十轮32个进程均完成新增288组启动预热，启动VLM请求200；但96并发仍未完成。
四个DP leader以及DP0 TP1的原生栈均变为`process_batch_result_decode ->
copy_done.synchronize -> cuEventSynchronize`，不再停在ReplaySSM模块加载。
不能据此认定整个死锁已修复，也不能仅凭CPU栈断定具体NCCL原因。
该轮没有任何成功并发请求，停止服务时尚未完成的请求按失败/中止记录。
原生栈`trial10-stall-native-*.txt`，日志归档`trial10-replayssm-event-wait/`。
一次有30秒限制的cuda-gdb附加未取得GPU kernel列表；其输出仅保存为
`trial10-gpu-kernels.txt`，不能用作具体GPU阻塞算子的证据。
第十一轮仅新增`--disable-overlap-schedule`做对照：仍为1M上下文、24/DP、
原有Graph24、FP32状态和DSPARK block7。可能牺牲CPU/GPU重叠收益，最终性能
必须另行测试，不能将此配置当作无损性能优化。
四节点第十一轮启动脚本SHA256：
`0da2442fa378b9d60be8664adb9151fe28c4234c9ddb00766a2a653bac7cd935`。
第十一轮初始化失败：`Lazy extra buffer requires overlap schedule`。
这是Mamba allocator的配置约束，不是GPU测试结果；没有绕过断言。
第十二轮保留关闭overlap，改用已有的`extra_buffer`策略（非lazy）。
该策略在非overlap下只需要一个track缓冲，代码明确支持；FP32状态、
24/DP上限、Graph24、1M上下文均保持。验收仍须重新完成。
第十二轮启动脚本SHA256：
`c311c9d68ddf55d1301825051a4e7a36bbac81005ba8027d2508ffe7dfa5e86e`。
第十二轮所有32个scheduler均停在`run_batch:3495`（new_seq_lens的D2H），
没有任何并发请求成功。`ncclras -v -t 8`显示4节点/32进程齐全，所有通信组
RUNNING/OK；这是host/RAS健康状态，不意味着GPU collective已经完成。
完整栈保存在`trial12-all-stacks-rank*.txt`，RAS为`trial12-ncclras.txt`。

进一步核对实际环境发现`NCCL_GRAPH_MIXING_SUPPORT=0`：部署版
`engine.py:1509`在DCP>1且未显式指定时设置0，但保留用户显式值。
[NCCL 2.28.9文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2289/user-guide/docs/env.html#nccl-graph-mixing-support)
说明关闭mixing时，同一通信组上尚未完成的graph与普通collective混用不受支持，
即使有stream依赖也不保证。当前regular DCP包含captured/eager通信，故这是
待验证的原因，不能在实验前认定根因。
第十三轮只显式设置`NCCL_GRAPH_MIXING_SUPPORT=1`，保留第十二轮其他参数；
不同时修改`NCCL_LAUNCH_ORDER_IMPLICIT`，也不移除Graph，便于归因。
第十三轮启动脚本SHA256：
`15246ded5231ffb735d3a944414fa48c321759a3740c387cc1388a54d3d39223`。
第十三轮32个进程均核验EAGER/mixing=1，启动成功，但相同96并发仍无成功返回。
主rank停在`run_batch -> new_seq_lens.to("cpu") -> cuMemcpyDtoHAsync_v2`。
因此mixing=0并非已证明的唯一原因，显式设1单独不足以解决本次停滞。
第十四轮仅新增`--disable-cuda-graph`，保持第十三轮其他参数，隔离图回放因素。
这是诊断配置，可能降低性能，不算性能优化，也不能替代并发与精度验收。
第十四轮启动脚本SHA256：
`bb714e0a64407bc623871ef2041beeda60464163867dc636e6f34901bf5cd81d`。
第十四轮尚未完成启动VLM warmup：DP0 TP0/TP1停在
`DSV3FusedAGemmKernel<7168,2112,true> -> cudaFuncSetAttribute -> cuLibraryLoadData`。
原生栈`trial14-startup-stacks.txt`。关闭Graph也跳过了原本的模型算子预热，
因此该轮不是有效的“充分预热后eager与graph”对照，不用于断言Graph根因。
先停止完整模型，运行`test_nccl_paths.py`：32卡world32与节点内8卡组的BF16/FP32
多消息大小collective正确性，再测graph/eager混用；每节点launcher总超时240秒。

SSH退避实测：02:31开始120秒等待，第一次重试成功；未重新执行部署命令。
02:42首次上传通信诊断脚本又遭GSSAPI拒绝，尚未启动任何通信测试进程。
对相同两份文件的SCP使用同一错误分类和2/4/8分钟等待，最多三次，避免只探测
成功却对文件传输无限重试。所有脚本及已下载失败日志保留在本地；四个模型服务
当前停止，容器、权重、部署目录和缓存均未删除。
该次SCP在120秒等待后的第一次重试成功，随后其余三节点文件分发成功。

独立NCCL检查第一版：14组eager正确性、8轮graph/eager混用的数值检查通过，
但在退出清理时未结束，最终240秒timeout退出124。因此只记录计算检查通过，
不把首版进程总体记为PASS。第二版先`graph.reset()`、释放graph，再销毁
process group，并把最终PASS输出移到正常销毁后；日志另存`nccl-probe-v2-*`。
第二版四节点退出0。第三版扩大到与DCP输出相当的24–48MiB FP32消息，覆盖
BS16/20/21/22/24/32、head width512及打包513；38组普通collective检查和8轮
约36MiB graph/eager混用通过，四节点均已确认退出0。
这验证的是独立Torch NCCL路径，不等价于SGLang的全部通信/图回放逻辑正确。

第十五轮准备：恢复图捕获作为模型预热，仅在`DecodeCudaGraphRunner.can_run_graph`
增加部署诊断gate：`SGLANG_DEPLOY_TARGET_EAGER_REPLAY=1`时target返回False，
draft保持原逻辑；保留捕获过程，避免第十四轮跳过预热。CPU gate四组合测试通过。
原始runner SHA256 `5c17a5d3d442f3413bd5647d79a1d072d581581bf3478ecd2f7d2b596e2ddb29`；
诊断runner SHA256 `cb13208db5cb3375cdc11f14cb940c8c3057178cea1e4ad49e6d113d48a07a73`。
启动脚本SHA256 `427c4ea82c610398a5c60073feef1be8fe7e1e983baa9b8526111b281dbc1256`。
该gate不改变数学计算，但目标模型eager可能降低性能，不能当作最终性能优化。

第十五轮启动失败，未进入并发验收：DP1/2/3处于idle target-verify时，KDA的
`mixed_qkv.view(0, 8, -1)`报错，随后其它rank通信异常退出；末尾重复metrics注册
错误是后续清理路径报错，不是首次异常。完整日志归档`trial15-target-eager-empty/`。
第十六轮仅增加KDA空token分支，返回同dtype/device的`[1, 0, H_v, D_v]`空张量；
不访问或修改Mamba状态，不跳过模型层外的MoE collective。非空计算原路径保留。
`test_kda_idle.py`提取实际函数检查空张量的shape/dtype/device与无状态访问，并验证
非空输入继续原路径；这是边界单测，不是非空KDA数值验证。必须再跑完整模型回归。
KDA原始SHA256：`ac443579529e9dd06bd939a34acc5c27384d57ec8dade2ebe7c50f990d6e2d68`。
修订SHA256：`b9e9082ff0c8bb30ea498922f34c51b4f934e79ffc46cc2f2471d43ff6e88858`。
四节点均执行`docker exec kimi-r22-20261027 python -B
/workspace/restore/native-context/test_kda_idle.py`：每台CPU/CUDA共12个边界组合PASS。
仅为空张量契约检查，没有执行非空KDA kernel。trial13–15与NCCL v1/v2/v3日志
已从全部四节点保存到本地`ecs-rank<N>-diagnostics/`，未删除远端原件。

第十六轮启动仍失败：idle DP绕过KDA空reshape后，在`deepseek_v2.py:1932`
触发`short-circuiting allreduce will lead to hangs`。这是eager idle与当前DCP
路径的契约问题，未删除或绕过断言。该轮仍不是有效的eager/graph性能对照。
第十七轮恢复原始KDA和DecodeCudaGraphRunner，撤下target-eager诊断gate；
相对第十三轮仅显式新增`NCCL_LAUNCH_ORDER_IMPLICIT=1`，其余配置保持。
[NCCL 2.28.9文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2289/user-guide/docs/env.html#nccl-launch-order-implicit)
说明该选项按host顺序约束同一设备上不同通信组的启动，默认关闭；本轮检验
多通信组顺序假设，不能在结果出来前声称根因或修复。
四节点启动脚本SHA256均为`caca0c11767fb8b34859444a15d7c98390fb50f845ac6485b9bdcb8476a2c48e`；
原始KDA和runner哈希已逐台核验，trial16日志及诊断源码均另存本地。
第十七轮启动成功，VLM warmup/health均200，32个进程均核验implicit order=1。
96并发仍停在约20个running requests、new_seq_lens D2H处，停止前0成功返回；
因此该选项单独不足以修复。主节点空闲显存约1.9–3.3GiB，无OOM错误。
cuda-gdb关闭auto-solib后仍未在45秒内取得GPU kernel列表，timeout强制结束；
目标scheduler核对未停在T状态。该输出不能用于认定具体GPU算子。
第十八轮沿用第十七轮参数，只在rank0通过`launch_profile.sh`启动Nsight：
CUDA/NVTX、graph node粒度、不采样CPU，延迟120秒，采集180秒，`--kill=none`。
只作为诊断，profiling期间的延迟不能当作性能收益。
第十八轮启动和health均通过，19:22:17 UTC开始96并发；19:22:39 UTC后同样
停在约20个running requests，尚无成功返回。正在等待限时采集产出，不作为验收通过。
采集结束后处于Generation状态，保存日志并停止peer服务；03:27本地时间，
`trial18-nsys.nsys-rep`写出完成约939MiB，`trial18-profiler.exit=0`。随后停止
主节点挂起服务，再离线导出SQLite分析。模型验收仍失败，profiler退出0不代表服务通过。
完整Nsight报告保留主节点宿主机`/data01/kimi-r22-20261027/native-context/`。
本机仅余约6GiB，暂不复制完整报告或大型SQLite；文本摘要与服务日志仍回传本地。
SQLite导出完成约2.3GiB、47694857事件，摘要`trial18-nsys-tail.jsonl`已在本地。
采样窗口约89.012s处，8卡最后推进位置分歧：FlashMLA local/warp-grouped combine
已完成，GPU1/5的SendRecv与后续输出投影有完成记录，其他卡停在合并附近拷贝。
未完成操作可能不在CUPTI活动表中，因此不能仅凭末尾记录宣布哪个kernel内部死锁。
源码确认实际为SGLang DCP `a2a`打包输出/LSE的uint8 SendRecv，不是独立通信
测试中主要覆盖的all-reduce。此前独立测试不能排除该路径问题。

第十九轮：使用已有`ag_rs`（LSE AllGather、FP32校正、ReduceScatter）对照，
保留local FP32、1M上下文、96上限和Graph24，关闭其不支持的Q投影复制。
Kimi配置resolver原本无条件覆盖用户后端为a2a；仅在部署副本`overrides.py`
保留显式`ag_rs + --no-dcp-replicate-q-proj`组合，其他默认行为不变。18组CPU
选择测试PASS；不是GPU数值验证。Q投影从重复计算改为分片后all-gather，仍须做
完整8K/128K回归，不能把该切换提前视为无损性能优化。
resolver原SHA256：`cfd5de8039e77f8171952cfc5ab9fc9fa448aef1d121f57fb2bff236a4706b88`。
四节点原文件哈希匹配，18组CPU选择测试均通过，当前resolver SHA256
`6264794b127c27e9e4eab2526ba05c7b59ad2cd50849e2af672a0582c3f42048`，启动脚本
`cdf7d80168d4239c821369432dea7cbdf65ccf6ea9f4a9530a679f210910e69f`。
第十九轮96并发客户端通过：96/96均为1024输入/512输出，四DP各完成24条；
总墙钟89.08秒（含prefill与排队），仅容量/稳定性smoke，不是吞吐A/B。
DP0日志观察到的同时解码峰值为21；配置上限24/DP不能等同实际一直满24。
第十九轮8K回归：8000输入/256输出，四DP完整token逐一匹配历史R22基线，
SHA256 `7155cd77ab5b0fd25c118f03c95239c30e521439fa396aec651fd9dc26ef4732`。
各请求accept_len=8、accept_rate=1；仍为重复token合成smoke，不能外推真实业务。
各DP耗时约9.09–9.24s，不是受控性能A/B。

历史验收清单（trial19已完成下列请求/健康检查；不得对运行中的rank重复启动）：

1. 每节点8条`CUDA driver module loading verified: EAGER`，无异常退出。
2. 核实实际KV容量×DCP至少1048576、运行上限24/DP、Graph24完成。
3. 运行96并发（1024输入/512输出）并要求96/96完整返回；此前
   `probe-concurrency128*.json`均为失败轮次，不得覆盖或当成通过。
4. 运行四DP确定性8K回归并比较完整token列表，再做128K请求与8次健康检查。
5. 将新结果保存为新文件名，复制回本地后更新本记录。

trial19最终128K请求131072/32完整返回，194.6018s，accept_len=8、rate=1，
cached_tokens8192、num_retractions0，完整token与第二轮128K基线一致。
输出SHA256 `0f169635a72a8e4ef439b715dd0b5feadf7fc731169e8de9c5f80222c74791b2`。
8次健康检查均200，四DP有效请求配置24、各96个Mamba状态槽、Graph24。
`trial19-active-files-rank<N>.sha256`是当前运行源码的权威清单。
本目录`decode_cuda_graph_runner.py`、`kda_backend.py`保留的是已撤下的诊断候选，
**不要将本目录全部Python文件无差别覆盖到部署源码**；运行中的两份文件均已恢复原版。

本轮遵循CUDA skill的现场取证与分阶段验证：先采集原生栈、保存失败轮次，
再调整启动配置与显存安排；未重编译或改写attention CUDA kernel。

### 完整模型8K回归

`eval_dp_consistency.py --input-len 8000 --output-len 256 --cycles 1`：
四个DP的完整输出token列表逐一与历史R22 smoke28比较，4/4完全一致；
`accept_len=8.0`、接受率100%。耗时分别为DP0 8.29s、DP1 10.72s、
DP2 11.16s、DP3 11.14s（包含首次调用，不作为正式性能对照）。
输出SHA256为`7155cd77ab5b0fd25c118f03c95239c30e521439fa396aec651fd9dc26ef4732`。
这是重复token合成smoke，不能把100%接受率外推到真实业务。

### 算子与边界验证

- `test_limits.py`：两个 worker 边界和 scheduler 输出预算，覆盖 DCP1/DCP8、
  8K、128K、近 1M；本机与四节点 CPU 检查通过。
- `test_long_dcp.py`：8 张 H20（SM90），PyTorch 2.11.0+cu130，CUDA runtime 13.0，
  使用历史 wheel；无本次编译 flags。DCP8 + NCCL merge 对照单卡 FP32 FlashMLA，
  Q=8、H=96，包含 batch 内非整齐长度：

| Batch | 最大 KV 长度 | 全 rank 最大绝对误差 | 结果 |
| --- | --- | --- | --- |
| 1 | 8192 | 0.00015922845 | PASS |
| 32 | 1024 | 0.00052325986 | PASS |
| 1 | 131072 | 0.0001067113 | PASS |
| 1 | 1048576 | 0.00013168532 | PASS |

容差 output `rtol=0.02, atol=0.002`，LSE `rtol=0.001, atol=0.002`。
这是算子数值验证，尚不能替代完整模型 1M 请求验证，也不声明 bitwise 一致。

## 启动与日志

容器名仍为 `kimi-r22-20261027`。确认旧模型进程已停止后，按原 rank 映射运行：

```bash
bash /data01/kimi-r22-20261027/native-context/launch.sh <node-rank>
```

日志：`/data01/kimi-r22-20261027/native-context/service-rank<N>.log`。
不要重复运行尚在服务的 rank；服务进程不会随容器重启自动恢复。
