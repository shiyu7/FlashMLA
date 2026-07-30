__version__ = "1.0.0"

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    get_mla_metadata_dense_fp8,
    flash_mla_dense_fp8,
    flash_mla_dense_fp8_out,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd
)
from flash_mla.dcp import (
    cyclic_local_length,
    make_cyclic_local_causal_seqlens,
    merge_local_attention,
)
from flash_mla.dcp2 import (
    DCP2AttentionWorkspace,
    create_dcp2_attention_workspace,
    dcp2_attention_payload_bytes,
    is_dcp2_peer_attention_eligible,
    merge_dcp_attention,
    nccl_merge_attention,
)

__all__ = [
    "get_mla_metadata",
    "get_mla_metadata_dense_fp8",
    "flash_mla_dense_fp8",
    "flash_mla_dense_fp8_out",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd",
    "cyclic_local_length",
    "make_cyclic_local_causal_seqlens",
    "merge_local_attention",
    "DCP2AttentionWorkspace",
    "create_dcp2_attention_workspace",
    "dcp2_attention_payload_bytes",
    "is_dcp2_peer_attention_eligible",
    "merge_dcp_attention",
    "nccl_merge_attention",
]
