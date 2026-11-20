# Kimi-K3 R22 四节点恢复记录（2026-10-27）

**当前状态（2026-10-28 03:47 CST）：原生1M上下文配置恢复，trial19部署验收通过，
四台ECS服务保持运行。** 详见[原生上下文部署记录](native-context/README.md)。
当前TP32 / DP4 / DCP8 / EP32，P/D共置，DSPARK，DCP通信使用ag_rs，Graph24。
请求上限96（24/DP；日志实测峰值21/DP）；96/96并发客户端请求完整返回，
四DP的8K完整token与历史R22基线一致，128K完整请求及8/8健康检查通过。
尚未做整模型1M请求或正式性能A/B；关闭overlap、Marlin/无peer MoE的配置均明确
记录，不宣称恢复了此前最优吞吐。旧a2a图回放停滞的底层根因仍未完全确定。

按用户授权，`ecs_ssh.py --retry-safe <公网IP> '<只读命令>'`在指定临时连接错误
后等待2/4/8分钟，最多重试3次；固定原跳板机、公网IP和密钥。变更命令不自动
重放，未知结果先核对远端状态。已发生一次临时失败时加`--after-failure`，
先等2分钟再探测。5项离线测试通过，详见上述原生上下文部署记录。

以下为此前已完成的 **9728-token 实验服务历史记录**，不是当前运行状态。
该轮权重加载、Graph 捕获、8K 跨 DP 一致性回归及 HTTP 健康检查均已通过。

## 节点与访问

仅使用以下四台 ECS，SSH 必须通过 `jumpecs-lf.byted.org` 跳板机、
公网 IP 和 `/Users/bytedance/perfTest.pem` 密钥。此部署不使用 K8s。

| 本次 node rank / DP rank | 公网 IP | 私网 IP | 主模型宿主机路径 |
| --- | --- | --- | --- |
| 0 | 101.126.12.182 | 192.168.44.92 | `/data01/models/Kimi-K3` |
| 1 | 115.191.51.207 | 192.168.44.91 | `/data00/Kimi-K3` |
| 2 | 115.191.21.96 | 192.168.44.93 | `/data00/Kimi-K3` |
| 3 | 115.191.2.23 | 192.168.44.90 | `/data01/models/Kimi-K3` |

四台 DSpark 权重均在 `/data01/models/Kimi-K3-DSpark`。
该目录是投机解码的 draft 模型，不替代 Kimi-K3 主模型；本次同时加载两者。
容器中统一挂载主模型为 `/data01/models/Kimi-K3`，draft 路径保持不变。
检查通过：每台主模型 96 个分片，共 1,560,936,091,448 字节；draft 1 个
分片，共 4,498,585,858 字节。本次检查的是分片存在性和大小，未重新计算全部权重哈希。

### 服务入口

- 集群内 API：`http://192.168.44.92:8080/v1`
- OpenAI-compatible 模型名：`Kimi-K3`
- 健康检查：`http://192.168.44.92:8080/health`

本机访问建议通过 SSH 隧道（公网 IP 只用于登录，未验证公网直连 8080）：

```bash
ssh -J jumpecs-lf.byted.org -i /Users/bytedance/perfTest.pem \
  -N -L 18080:127.0.0.1:8080 root@101.126.12.182
```

然后使用 `http://127.0.0.1:18080/v1`；可先请求
`http://127.0.0.1:18080/v1/models` 检查模型列表。

## 部署配置

- 共置 Prefill + Decode；4 节点 × 8 张 H20，总计 32 GPU。
- TP32 / DP4 / DCP8 / EP32，启用 DP attention 和 DP LM head。
- Prefill Triton；Decode FlashMLA R14-P10 canonical peer；chunk size 1216。
- DSPARK block size 7，保留 R12 draft DCP1 隔离和 R22 KV 写入修复。
- FP8 E4M3 KV，Marlin，`moe-a2a-backend=none`。
- `chunked-prefill-size=8192` 是命令行值；运行时按 DP4 调整为每 DP 2048。
- CUDA Graph max batch 16；max running requests 128；mem fraction 0.85。
- 本次显式设置 `context-length=9728`，与固定 canonical 容量 `1216 × 8`
  一致；覆盖既有 8000/8192 输入 + 1500 输出测试。
- API 服务名 `Kimi-K3`；rank0 监听 `0.0.0.0:8080`。

镜像固定为：

```text
iaas-gpu-cn-beijing.cr.volces.com/sunqi/sglang@sha256:7c4f7f8fe146b87bedc389a49df9e9269b2c66b869e39d3ae7eb8151ac716eb6
```

容器名均为 `kimi-r22-20261027`，entrypoint 为 `/bin/bash`。
四台均使用 PyTorch `2.11.0+cu130` / CUDA runtime `13.0`。
FlashMLA 使用历史验证通过的预编译 wheel，未重新编译：

```text
flash_mla-1.0.0+2026.8.27.9.31.12-cp312-cp312-linux_x86_64.whl
SHA256 ea3f5027f2a33d22d8141d250f97260df39f76f1252dd1818ef81490c5959c57
```

四台关键源码 SHA256 一致：

| 文件 | SHA256 |
| --- | --- |
| `triton_backend.py` | `0af3260592c00de2a86bba71a1871a7534b9446757eabd288cfdcb62b1da1bca` |
| `flashmla_backend.py` | `7157b824f9749ac4f39576cddef95896fd70aa4c12b716a9d3ffbec6a1385c6b` |
| `spec_utils.py` | `210d7159af81628f8a128bae596640e7c4d59c8a38e598cddb43b1d931b9b9ce` |

历史源码 `/data00/kimi-dcp` 只读挂载为 `/workspace/kimi-dcp`。
运行目录 `/data01/kimi-r22-20261027` 挂载为 `/workspace/restore`。
缓存复用 `/data01/kimi-dcp-cache-20260826`。`.90` 的 `/data00` 已满，
本次不向该盘写日志或生成缓存。

## 已释放的服务

已停止 `.92` 和 `.91` 上的 `deepseek-v4-pro-best-decode-pp` 容器，释放 16 张 GPU。
另外两台 GPU 原本空闲。未删除任何旧容器、模型或持久文件。
停止前在各节点运行目录保存：

- `previous-service-inspect.json`
- `previous-service-processes.txt`
- `previous-service-docker.log`

这些记录及原容器可用于恢复旧服务；其模型进程原由 `docker exec` 启动，
仅 `docker start` 旧容器并不会自动恢复模型服务。

## 验证记录

1. 四台 preflight：已通过关键源码哈希、checkpoint 分片和依赖导入检查。
2. `.92` 的 8-H20 DCP workspace 回归：8 个 rank 各 `1 passed`，约 6.55–6.71 秒。
   覆盖 KV8192/9692，使用已安装 wheel，没有重新编译。
3. 8K DSPARK 跨 DP 回归：输入为 8000 个固定 token，输出 256 token，
   共 2 轮 × 4 请求。每轮覆盖 DP0–DP3，轮间清空 radix cache。
   8/8 请求的完整输出 token 列表与历史 R22 smoke28 基准逐 token 一致。
   8/8 请求 `accept_len=8.0`、`accept_rate=100%`。
4. 短请求路径预热后，`/health` 连续 8 次返回 HTTP 200；最后 4 次约 1 秒。
   `/v1/models`、`/model_info` 和 `/server_info` 均可用。
5. `/server_info` 实测确认 TP32 / DP4 / DCP8 / EP32、DSPARK block7、
   context9728；`disaggregation_mode="null"`，即 P/D 共置而非分离部署。

本次 8K/256 请求耗时（秒；包含 prefill 和 decode）：

| DP rank | 第一轮（冷路径） | 第二轮（已预热） | 接受率（两轮） |
| --- | --- | --- | --- |
| 0 | 25.78 | 7.59 | 100% |
| 1 | 25.87 | 7.71 | 100% |
| 2 | 37.99 | 7.67 | 100% |
| 3 | 37.74 | 7.58 | 100% |

输出 SHA256 均为
`7155cd77ab5b0fd25c118f03c95239c30e521439fa396aec651fd9dc26ef4732`。
这是重复 token 的正确性 smoke，不是正式随机负载性能 A/B；100% 接受率
不能代表真实业务负载。此次没有重跑正式吞吐或 DCP8/DCP1 对照实验。

启动后首次 8K 请求以及单 token 健康请求均存在冷路径开销；最初健康检查
曾超过其 20 秒超时而返回 503，短请求路径预热后恢复。没有修改健康检查
实现，也没有关闭其实际生成探测；重启后需留足预热时间。

本地验证产物：

- `workspace-test.log`：8-GPU workspace 回归输出。
- `smoke-8k-256.json` / `smoke-8k-256.log`：逐请求结果及 token。
- `health-check.log`：最后 4 次健康检查结果。
- `runtime-summary.json`：服务返回的实际并行与后端配置。

回归命令：

```bash
docker exec kimi-r22-20261027 torchrun --standalone --nproc-per-node=8 \
  -m pytest -q -p no:cacheprovider /workspace/restore/test_dcp8_canonical_workspace.py
```

8K smoke 命令（rank0 宿主机执行）：

```bash
python3 /data01/kimi-r22-20261027/eval_dp_consistency.py \
  --base-url http://127.0.0.1:8080 --input-len 8000 --output-len 256 \
  --cycles 2 --timeout 300 \
  --output /data01/kimi-r22-20261027/smoke-8k-256.json
```

## 日志与启动

各节点日志：`/data01/kimi-r22-20261027/service-rank<N>.log`。
若服务退出，同目录生成 `service-rank<N>.exit`。
启动脚本、环境建立脚本及 preflight 脚本保存于本目录，并已同步四台 ECS。

首次启动或确认原进程已停止后，在每台执行对应 rank：

```bash
bash /data01/kimi-r22-20261027/launch.sh <node-rank>
```

`launch.sh` 通过 `docker exec -d` 启动；容器重启后需重新启动服务进程。
不要在同一节点重复启动尚在运行的服务。
