#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: kimi-dcp-serve-20260826.sh dcp8|dcp8-e2e|tp32dp4|dcp8-dspark|tp32dp4-dspark NODE_RANK}
node_rank=${2:?usage: kimi-dcp-serve-20260826.sh dcp8|dcp8-e2e|tp32dp4|dcp8-dspark|tp32dp4-dspark NODE_RANK}

spec_args=()
chunked_prefill_size=512
mamba_full_memory_ratio=0.8
cuda_graph_max_bs=128
moe_a2a_backend=none
decode_attention_backend=${DECODE_ATTENTION_BACKEND:-flashmla}
disable_cuda_graph_args=()
if [[ ${DISABLE_CUDA_GRAPH:-0} == 1 ]]; then
  disable_cuda_graph_args=(--disable-cuda-graph)
fi
skip_server_warmup_args=()
if [[ ${SKIP_SERVER_WARMUP:-0} == 1 ]]; then
  skip_server_warmup_args=(--skip-server-warmup)
fi
dcp_qrep_args=()
if [[ ${DISABLE_DCP_REPLICATE_Q_PROJ:-0} == 1 ]]; then
  dcp_qrep_args=(--no-dcp-replicate-q-proj)
fi

case "${mode}" in
  dcp8)
    dcp_args=(--dcp-size 8)
    disagg_args=(
      --disaggregation-mode decode
      --disaggregation-ib-device mlx5_1,mlx5_2,mlx5_3,mlx5_4
    )
    ;;
  dcp8-e2e)
    dcp_args=(--dcp-size 8)
    disagg_args=()
    ;;
  tp32dp4)
    dcp_args=(--dcp-size 1)
    disagg_args=()
    ;;
  dcp8-dspark)
    dcp_args=(--dcp-size 8)
    disagg_args=()
    spec_args=(
      --speculative-algorithm DSPARK
      --speculative-draft-model-path /data/models/models/Kimi-K3-DSpark
      --speculative-dspark-block-size 7
      --enable-dp-lm-head
      # The deployed SGLang revision renamed the older
      # --enable-gdn-replayssm-spec flag to this GDN/KDA-neutral spelling.
      --enable-linear-replayssm-spec
    )
    chunked_prefill_size=8192
    mamba_full_memory_ratio=0.85
    # The K3 ReplaySSM draft pool has 18 request slots in this memory layout.
    # DP-attention graph alignment rounds that capacity to an invalid bs=24
    # bucket in this SGLang revision, so keep the captured buckets <= 16.
    cuda_graph_max_bs=16
    ;;
  tp32dp4-dspark)
    dcp_args=(--dcp-size 1)
    disagg_args=()
    spec_args=(
      --speculative-algorithm DSPARK
      --speculative-draft-model-path /data/models/models/Kimi-K3-DSpark
      --speculative-dspark-block-size 7
      --enable-dp-lm-head
      --enable-linear-replayssm-spec
    )
    chunked_prefill_size=8192
    mamba_full_memory_ratio=0.85
    cuda_graph_max_bs=16
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
export NCCL_IB_GID_INDEX=3
export NCCL_MIN_NCHANNELS=24
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_CUMEM_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_K3_ATTN_RES_MODE=jit
export SGLANG_MOE_FUSED_GATE_RADIX=1
export SGLANG_RAGGED_VERIFY_MODE=static
export PYTHONPATH=/workspace/kimi-dcp/sglang-kimi-k3-f748-dcp-20260826/python:/workspace/kimi-dcp/pycompat:/workspace/kimi-dcp/FlashMLA-DCP-3a0b48b-20260826:${PYTHONPATH:-}

exec sglang serve \
  --trust-remote-code \
  --model-path /data/models/models/Kimi-K3 \
  --served-model-name /data/models/DeepSeek-V4-Flash-FP8/ \
  --tp-size 32 \
  "${dcp_args[@]}" \
  --dp-size 4 \
  --enable-dp-attention \
  --nnodes 4 \
  --node-rank "${node_rank}" \
  --dist-init-addr 192.168.44.92:20000 \
  --ep-size 32 \
  --moe-a2a-backend "${moe_a2a_backend}" \
  --moe-runner-backend marlin \
  --decode-attention-backend "${decode_attention_backend}" \
  --prefill-attention-backend triton \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size "${chunked_prefill_size}" \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --dist-timeout 3600 \
  --watchdog-timeout 14400 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mamba-full-memory-ratio "${mamba_full_memory_ratio}" \
  --host 0.0.0.0 \
  --port 8080 \
  --cuda-graph-max-bs "${cuda_graph_max_bs}" \
  --max-running-requests 128 \
  --kv-cache-dtype fp8_e4m3 \
  --disable-custom-all-reduce \
  "${disable_cuda_graph_args[@]}" \
  "${skip_server_warmup_args[@]}" \
  "${dcp_qrep_args[@]}" \
  "${disagg_args[@]}" \
  "${spec_args[@]}" \
  --enable-metrics
