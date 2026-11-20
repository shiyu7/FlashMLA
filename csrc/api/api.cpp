#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"
#include "extension/sm90/dense_fp8/dense_fp8_api.h"
#include "extension/sm90/dense_fp8/canonical_merge.h"
#include "extension/sm90/dense_fp8/canonical_peer_kv_pack.h"
#include "extension/sm90/dense_fp8/canonical_peer_reduce_scatter.h"
#include "extension/sm90/dcp2/dcp2_merge.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
    m.def("dense_decode_fp8_fwd", &fwd_kvcache_mla_fp8);
    m.def("dense_decode_fp8_fwd_fp32", &fwd_kvcache_mla_fp8_fp32);
    m.def("dense_decode_fp8_fwd_canonical", &fwd_kvcache_mla_fp8_canonical);
    m.def(
        "dense_decode_fp8_fwd_canonical_leaves",
        &fwd_kvcache_mla_fp8_canonical_leaves);
    m.def(
        "dense_decode_fp8_fwd_canonical_leaves_out",
        &fwd_kvcache_mla_fp8_canonical_leaves_out);
    m.def("dense_decode_fp8_fwd_out", &fwd_kvcache_mla_fp8_out);
    m.def("canonical_merge_pair_out", &canonical_merge_pair_out);
    m.def("canonical_peer_tree_prepare", &canonical_peer_tree_prepare);
    m.def("canonical_peer_tree_merge", &canonical_peer_tree_merge);
    m.def(
        "canonical_peer_reduce_scatter_prepare",
        &canonical_peer_reduce_scatter_prepare);
    m.def(
        "canonical_peer_reduce_scatter",
        &canonical_peer_reduce_scatter);
    m.def(
        "canonical_peer_reduce_scatter_compact",
        &canonical_peer_reduce_scatter_compact);
    m.def(
        "canonical_peer_pack_token_cyclic_kv",
        &canonical_peer_pack_token_cyclic_kv);
    m.def("dcp2_prepare_attention", &dcp2_prepare_attention);
    m.def("dcp2_merge_attention", &dcp2_merge_attention);
    m.def(
        "get_mla_metadata_dense_fp8",
        &get_mla_decoding_metadata_dense_fp8);
    m.def(
        "get_mla_metadata_canonical_dense_fp8",
        &get_mla_decoding_metadata_canonical_dense_fp8);
}
