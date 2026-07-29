__version__ = "1.0.0"

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    get_mla_metadata_dense_fp8,
    flash_mla_dense_fp8,
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

__all__ = [
    "get_mla_metadata",
    "get_mla_metadata_dense_fp8",
    "flash_mla_dense_fp8",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd",
    "cyclic_local_length",
    "make_cyclic_local_causal_seqlens",
    "merge_local_attention",
]
