#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"
#include "extension/sm90/dense_fp8/dense_fp8_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
    m.def("dense_decode_fp8_fwd", &fwd_kvcache_mla_fp8);
    m.def(
        "get_mla_metadata_dense_fp8",
        &get_mla_decoding_metadata_dense_fp8);
}
