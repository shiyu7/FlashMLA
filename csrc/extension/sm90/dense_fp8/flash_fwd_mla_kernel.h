/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

using namespace cute;

#include "named_barrier.h"
#include "utils.h"
#include "softmax.h"
#include "static_switch.h"
#include "flash_mla.h"
#include "fp8_transpose_v.h"


template<typename PrecType, int DIM, int DIM2 = DIM, cute::GMMA::Major major = GMMA::Major::K>
constexpr auto getSmemLayoutK() {
    constexpr int headSizeBytes = sizeof(PrecType) * DIM;
    constexpr int headSizeBytes2 = sizeof(PrecType) * DIM2;

    if constexpr (major == GMMA::Major::K) {
        if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
            return GMMA::Layout_K_SW128_Atom<PrecType>{};
        } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
            return GMMA::Layout_K_SW64_Atom<PrecType>{};
        } else {
            return GMMA::Layout_K_SW32_Atom<PrecType>{};
        }
    } else {
        if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
            return GMMA::Layout_MN_SW128_Atom<PrecType>{};
        } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
            return GMMA::Layout_MN_SW64_Atom<PrecType>{};
        } else {
            return GMMA::Layout_MN_SW32_Atom<PrecType>{};
        }
    }

}

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t, typename elem_type_o = cutlass::bfloat16_t, int kHeadDimV_ = 0>
struct Flash_fwd_kernel_traits_mla {
    using Element = elem_type;
    using ElementO = elem_type_o;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t>;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kNWarpsS = 4;
    static constexpr int kNThreadsS = kNWarpsS * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kHeadDimV <= kHeadDim);

    static constexpr int kBlockKSmem = Is_FP8 ? (kHeadDim % 128 == 0 ? 128 : 64) : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kBlockKSmemO = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzleO = kBlockKSmemO == 32 ? 2 : 3;

    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 ? GMMA::Major::MN : GMMA::Major::K;

    using TiledMma = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<kNWarpsS / 4>, _1, _1>>{}));

    static constexpr int AtomLayoutNO = kNThreads / kNThreadsS;
    using TiledMmaO = decltype(make_tiled_mma(
            cute::GMMA::rs_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kHeadDimV / AtomLayoutNO>, Int<kBlockN>>,
                    GMMA::Major::K, MmaMajorV>(),
            Layout<Shape<Int<kNWarpsS / 4>, Int<AtomLayoutNO>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutP = std::conditional_t<
        Is_FP8,
        Layout<Shape<Shape<_4, _2>, Int<kNThreadsS>, _1, _2, Int<kBlockN / 32>>>,
        Layout<Shape<Shape<_2, _2>, Int<kNThreadsS>, _1, _2, Int<kBlockN / 16>>>
    >;
    using SmemLayoutRow = Layout<Shape<_2, Int<kNThreadsS>>, Stride<_1, _2>>;

    using SmemLayoutAtomO = decltype(composition(
            Swizzle<kSwizzleO, 3, 3>{},
            Layout<Shape<Int<8>, Int<kBlockKSmemO>>, Stride<Int<kBlockKSmemO>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, ElementO>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    static constexpr int kNThreadsLoad = kNThreads - kNThreadsS;
    static_assert(kNThreadsLoad % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    static constexpr int kGmemElemsPerLoadO = sizeof(cute::uint128_t) / sizeof(ElementO);
    static_assert(kHeadDim % kGmemElemsPerLoadO == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadO");
    static constexpr int kGmemThreadsPerRowO = kBlockKSmemO / kGmemElemsPerLoadO;
    static_assert(kNThreadsLoad % kGmemThreadsPerRowO == 0, "kNThreads must be a multiple of kGmemThreadsPerRowO");

    using GmemLayoutAtom = Layout<
            Shape<Int<kNThreadsLoad / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;


    using GmemTiledCopy = decltype(make_tiled_copy(
            Copy_Atom<Gmem_copy_struct, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 vals per read

    using GmemLayoutAtomO = Layout<
            Shape<Int<kNThreadsS / kGmemThreadsPerRowO>, Int<kGmemThreadsPerRowO>>,
            Stride<Int<kGmemThreadsPerRowO>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementO>{},
            GmemLayoutAtomO{},
            Layout<Shape<_1, Int<kGmemElemsPerLoadO>>>{}));  // Val layout, 8 vals per store

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowAccum = kBlockKSmemO / kGmemElemsPerLoadAccum;
    using GmemLayoutAtomOaccum = Layout<
            Shape<Int<kNThreadsS / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
            Stride<Int<kGmemThreadsPerRowAccum>, _1>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            GmemLayoutAtomOaccum{},
            Layout<Shape<_1, Int<kGmemElemsPerLoadAccum>>>{}));  // Val layout, 4 vals per store


    // ------ for f8 ------
    using SmemFp8Tranpose = SmemTransposeFp8_64x64<kBlockN, kHeadDimV, SmemLayoutK>;
    using SmemLayoutVtMMa = typename SmemFp8Tranpose::SmemLayoutVt;
};

namespace flash {

using namespace cute;

template<typename Kernel_traits>
struct SharedStorageMLA {
    using SmemV_t = std::conditional_t<Kernel_traits::Is_FP8,
                                    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutVtMMa>>,
                                    cute::array_aligned<typename Kernel_traits::Element, 0>>;
    union {
        struct {
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK> * 2> smem_k;  // Double buffer
            SmemV_t smem_vt;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutP>> smem_p;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_scale;
        };
        struct {
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_max;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_sum;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;
        };
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Split, typename SharedStorage, typename AccO, typename Softmax>
__forceinline__ __device__ void store(const DecodingParams_fp8 &params, const int bidb, const int bidh, const int m_block, const int n_split_idx,
                                      SharedStorage &shared_storage, AccO tOrO, Softmax softmax, float descale_k, float scale_softmax) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    using Element = typename Kernel_traits::ElementO;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    // Epilogue

    const int split_offset = __ldg(params.num_splits_ptr + bidb);

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
        tOrO, scale_softmax, descale_k);

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shared_storage.smem_o.data())), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    // SM90_U32x4_STSM_N is the fast 16-bit output path.  A float output has
    // the same representation as the split accumulator path and must use the
    // generic 128-bit copy atom; STSM_N cannot vectorize the 32-bit register
    // fragment into this shared-memory layout.
    using SmemTiledCopyO = std::conditional_t<
            !Split && !cute::is_same_v<Element, ElementAccum>,
            typename Kernel_traits::SmemCopyAtomO,
            typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(tOrO);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    __syncthreads();

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h_k + bidh) * params.q_seq_per_hk + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lse = (bidb * params.h_k + bidh) * params.q_seq_per_hk + m_block * kBlockM;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h_k + bidh) * params.q_seq_per_hk + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (Split ? row_offset_lseaccum : row_offset_lse)),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    if (tidx >= kNThreadsS) { return; }

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // ((MMA=4, X), MMA_M, MMA_K=1)
    Tensor taccOcO_row = taccOcO(make_coord(0, _, 0), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < params.q_seq_per_hk - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.q_seq_per_hk - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla(const DecodingParams_fp8 &params,
                                                                   const int bidb, const int bidh, const int m_block,
                                                                   const int n_split_idx, const int seqlen_k,
                                                                   const int n_block_min, const int n_block_max, const bool NoSplit,
                                                                   SharedStorage &shared_storage, const float descale_k, const float scale_softmax, const float scale_softmax_log2) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    static_assert(kNThreads == 256 and kNThreadsS == 128);
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;
    int n_block = n_block_max - 1;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{});

    auto sV = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutV{});
    auto sVt = [&](){
        if constexpr(Kernel_traits::Is_FP8){
            return make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), typename Kernel_traits::SmemLayoutVtMMa{});
        } else {
            return make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutVtransposed{});
        }
    }();

    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Kernel_traits::SmemLayoutP{});
    Tensor tPsP = sP(_, tidx % kNThreadsS, _, _, _);
    Tensor sScale_o = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tScale_osScale_o = sScale_o(_, tidx % kNThreadsS);
    Tensor sRow_max = make_tensor(make_smem_ptr(shared_storage.smem_max.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_maxsRow_max = sRow_max(_, tidx % kNThreadsS);
    Tensor sRow_sum = make_tensor(make_smem_ptr(shared_storage.smem_sum.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_sumsRow_sum = sRow_sum(_, tidx % kNThreadsS);

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrO = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
    clear(tOrO);

    flash::Softmax<2 * size<1>(tOrO)> softmax;

    int warp_group_idx = cutlass::canonical_warp_group_idx();
    if (warp_group_idx == 0) {
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
        Tensor tSrK = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

        if (n_block % 2 == 1) {
            // Double buffer for sK
            constexpr int sK_offset = size(sK);

            if constexpr (Kernel_traits::Is_FP8) {
                tSrK.data() = tSrK.data() + sK_offset / 16;
            } else {
                tSrK.data() = tSrK.data() + sK_offset / 8;
                tOrVt.data() = tOrVt.data() + sK_offset / 8;
            }
        }

        // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
        // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
        // We will have at least 1 "masking" iteration.
        // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
        // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
        const bool has_explicit_causal_seqlens =
            params.causal_seqlens_k_ptr != nullptr;
        int min_visible_k = seqlen_k;
        if (has_explicit_causal_seqlens) {
            // Rows are laid out as [query, q_head_per_hk]. The first valid
            // row in this M tile therefore owns the smallest causal boundary
            // because query positions increase monotonically.
            const int first_row = min(m_block * kBlockM, params.q_seq_per_hk - 1);
            const int first_query_idx = first_row / params.q_head_per_hk;
            min_visible_k = __ldg(
                params.causal_seqlens_k_ptr
                + bidb * params.causal_seqlens_k_batch_stride
                + first_query_idx);
        }
#pragma unroll 1
        for (int masking_step = n_masking_steps; n_block >= n_block_min; --masking_step, --n_block) {
            __syncthreads();

            Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
            flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma, tSrQ, tSrK, tSrS);

            const bool is_masking_step =
                has_explicit_causal_seqlens
                    ? (n_block + 1) * kBlockN > min_visible_k
                    : masking_step > 0;
            const bool is_first_iteration = n_block == n_block_max - 1;

            if (is_masking_step) {
                Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
                Tensor tScS = thr_mma.partition_C(cS);
#pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    if constexpr (!Is_causal) {  // Just masking based on col
                        if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) tSrS(i) = -INFINITY;
                    } else {
                        int row = int(get<0>(tScS(i)));
                        const int logical_row = m_block * kBlockM + row;
                        const int query_idx = min(
                            logical_row / params.q_head_per_hk,
                            params.s_q - 1);
                        const int visible_k =
                            has_explicit_causal_seqlens
                                ? __ldg(
                                      params.causal_seqlens_k_ptr
                                      + bidb * params.causal_seqlens_k_batch_stride
                                      + query_idx)
                                : seqlen_k
                                      - (params.s_q - query_idx - 1);
                        const int col_limit_right =
                            visible_k - 1 - n_block * kBlockN;
                        if (int(get<1>(tScS(i))) > col_limit_right) tSrS(i) = -INFINITY;
                    }
                }
            }

            // We have key_padding_mask so we'll need to Check_inf
            Tensor scale_o = is_first_iteration
                             ? softmax.template softmax</*Is_first=*/true,  /*Check_inf=*/Is_causal>(tSrS, scale_softmax_log2)
                             : is_masking_step ?
                               softmax.template softmax</*Is_first=*/false, /*Check_inf=*/Is_causal>(tSrS, scale_softmax_log2)
                                               : softmax.template softmax</*Is_first=*/false, /*Check_inf=*//*Is_local=*/false>(tSrS, scale_softmax_log2);

            if constexpr (Kernel_traits::Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMmaO>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);

            cute::copy(tOrP, tPsP); // send Aregs of MMA1 instead of Cregs of MMA0
            cute::copy(scale_o, tScale_osScale_o);

            cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::SReady));

            flash::rescale_o(tOrO, scale_o);

            if constexpr (Kernel_traits::Is_FP8) {
                cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::TransVReady));
            }
            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);

            // Double buffer for sK
            const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
            if constexpr (Kernel_traits::Is_FP8) {
                tSrK.data() = tSrK.data() + sK_offset / 16;
            } else {
                tSrK.data() = tSrK.data() + sK_offset / 8;
                tOrVt.data() = tOrVt.data() + sK_offset / 8;
            }
        }

        cute::copy(softmax.row_max, tRow_maxsRow_max);
        cute::copy(softmax.row_sum, tRow_sumsRow_sum);
        cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::SoftmaxReady));
    } else {
        const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        int cur_block_table = __ldg(&block_table[n_block]);

        const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
        Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_Q;
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx - kNThreadsS);
        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                                                              params.q_seq_per_hk - m_block * kBlockM);

        const index_t row_offset_k = (bidh / params.h_h_k_ratio) * params.k_head_stride;
        Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.k_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_K;
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx - kNThreadsS);
        Tensor tKgK = gmem_thr_copy_K.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
        Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_K.partition_S(cK);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));

        if (n_block % 2 == 1) {
            // Double buffer for sK
            constexpr int sK_offset = size(sK);
            tKsK.data() = tKsK.data() + sK_offset;
            if constexpr (!Kernel_traits::Is_FP8) {
                tOrVt.data() = tOrVt.data() + sK_offset / 8;
            }
        }

        // We need to clear the sK smem tiles because K is V.
        const index_t offset_k = cur_block_table * params.k_batch_stride;
        tKgK.data() = tKgK.data() + offset_k;
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK,
                                                                                        seqlen_k - n_block * kBlockN);
        tKgK.data() = tKgK.data() + -offset_k;
        cute::cp_async_fence();

        if (n_block - 1 >= n_block_min) {
            cur_block_table = __ldg(&block_table[n_block - 1]);
        }

#pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
            flash::cp_async_wait<0>();
            __syncthreads();

            if (n_block - 1 >= n_block_min) {
                // Double buffer for sK
                const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
                tKsK.data() = tKsK.data() + sK_offset;

                const index_t offset_k = cur_block_table * params.k_batch_stride;
                tKgK.data() = tKgK.data() + offset_k;
                flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK);
                tKgK.data() = tKgK.data() + -offset_k;
                cute::cp_async_fence();
            }

            if constexpr (Kernel_traits::Is_FP8) {
                auto TransV = [&]() {
                    using SmemFp8Tranpose = typename Kernel_traits::SmemFp8Tranpose;
                    SmemFp8Tranpose smem_transpose_V;
                    Tensor sV_divide = as_position_independent_swizzle_tensor(
                        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename SmemFp8Tranpose::SmemLayoutTransposeV{}));
                    Tensor sVt_divide = as_position_independent_swizzle_tensor(
                        make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), typename SmemFp8Tranpose::SmemLayoutTransposeVt{}));

                    if (n_block % 2 == 1) {
                        sV_divide.data() = sV_divide.data() + size(sK);
                    }

                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < shape<2>(typename SmemFp8Tranpose::SmemLayoutTransposeV{}); ++j) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < shape<1>(typename SmemFp8Tranpose::SmemLayoutTransposeV{}); ++i) {
                            smem_transpose_V.transpose(flatten(sV_divide(_, i, j)), flatten(sVt_divide(_, i, j)));
                        }
                    }
                };

                TransV();
                cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::TransVReady));
            }

            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::SReady));

            if (n_block - 2 >= n_block_min) {
                cur_block_table = __ldg(&block_table[n_block - 2]);
            }

            typename Kernel_traits::TiledMma tiled_mma;
            auto tSrS_layout = flash::convert_layout_acc_Aregs<Kernel_traits::TiledMmaO>(partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}).layout());
            Tensor tOrP = make_tensor<Element>(tSrS_layout);
            Tensor scale_o = make_tensor<float>(Shape<_2>{});
            cute::copy(tScale_osScale_o, scale_o);
            cute::copy(tPsP, tOrP);

            flash::rescale_o(tOrO, scale_o);

            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);

            if constexpr (!Kernel_traits::Is_FP8) {
                // Double buffer for sK
                const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
                tOrVt.data() = tOrVt.data() + sK_offset / 8;
            }
        }

        cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::SoftmaxReady));
        cute::copy(tRow_maxsRow_max, softmax.row_max);
        cute::copy(tRow_sumsRow_sum, softmax.row_sum);
    }

    if (NoSplit)
        store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, shared_storage, tOrO, softmax, descale_k, scale_softmax);
    else
        store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, shared_storage, tOrO, softmax, descale_k, scale_softmax);
}

template<typename Kernel_traits>
__forceinline__ __device__ void store_neutral_attention_state(
    const DecodingParams_fp8 &params,
    const int batch_id,
    const int bidh,
    const int m_block,
    const int n_split_idx,
    const bool split_output) {
    using ElementO = typename Kernel_traits::ElementO;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    const int row_begin = m_block * kBlockM;
    const int num_rows = min(kBlockM, params.q_seq_per_hk - row_begin);

    if (split_output) {
        const int split_offset = __ldg(params.num_splits_ptr + batch_id);
        const int split_idx = split_offset + n_split_idx;
        float *out_ptr =
            reinterpret_cast<float *>(params.oaccum_ptr)
            + (((split_idx * params.h_k + bidh) * params.q_seq_per_hk
                + row_begin) * kHeadDimV);
        for (int idx = threadIdx.x; idx < num_rows * kHeadDimV;
             idx += blockDim.x) {
            out_ptr[idx] = 0.f;
        }
        float *lse_ptr =
            reinterpret_cast<float *>(params.softmax_lseaccum_ptr)
            + (split_idx * params.h_k + bidh) * params.q_seq_per_hk
            + row_begin;
        for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
            lse_ptr[row] = -INFINITY;
        }
        return;
    }

    ElementO *out_ptr =
        reinterpret_cast<ElementO *>(params.o_ptr)
        + batch_id * params.o_batch_stride
        + bidh * params.o_head_stride
        + row_begin * params.o_row_stride;
    for (int idx = threadIdx.x; idx < num_rows * kHeadDimV;
         idx += blockDim.x) {
        const int row = idx / kHeadDimV;
        const int col = idx % kHeadDimV;
        out_ptr[row * params.o_row_stride + col] = ElementO(0);
    }
    float *lse_ptr =
        reinterpret_cast<float *>(params.softmax_lse_ptr)
        + (batch_id * params.h_k + bidh) * params.q_seq_per_hk
        + row_begin;
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        lse_ptr[row] = -INFINITY;
    }
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage, bool UsePdl>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const DecodingParams_fp8 params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;
    if (begin_idx >= params.b || begin_idx < 0) {
        if constexpr (UsePdl) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
        return;
    }
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

    float descale_k = 1.f;
    float scale_softmax = params.scale_softmax;
    float scale_softmax_log2 = params.scale_softmax_log2;
    if constexpr (Kernel_traits::Is_FP8) {
        float descale_q = __ldg(params.descale_q_ptr);
        descale_k = __ldg(params.descale_k_ptr);
        scale_softmax = scale_softmax * descale_q * descale_k;
        scale_softmax_log2 = scale_softmax_log2 * descale_q * descale_k;
    }

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = __ldg(params.seqlens_k_ptr + batch_id);
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        const int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit =
            params.canonical_num_chunks == 0 && n_block_min == 0 &&
            n_block_max == cute::ceil_div(seqlen_k, kBlockN);
        if (batch_id > begin_idx) {
            __syncthreads();  // Barrier between two tiles.
        }
        if constexpr (UsePdl) {
            if (batch_id == end_idx) {
            // The dependent combine grid may now run its address-generation
            // prologue. Its grid-dependency wait remains ahead of every load
            // from the accumulators written below.
                cudaTriggerProgrammaticLaunchCompletion();
            }
        }
        if (seqlen_k == 0 || n_block_min >= n_block_max) {
            // Empty cyclic shards occur for short requests when cp_size is
            // larger than the global prefix, and inactive canonical tail
            // chunks are represented by an empty block interval. Do not
            // touch block_table[-1]; emit the exact neutral state instead.
            store_neutral_attention_state<Kernel_traits>(
                params,
                batch_id,
                bidh,
                m_block,
                n_split_idx,
                params.canonical_num_chunks > 0);
            continue;
        }
        flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, NoSplit, shared_storage, descale_k, scale_softmax, scale_softmax_log2);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Keep the original 128-thread row mapping for small launches.  It exposes
// four output elements per thread and is faster when there are too few rows
// to amortize the longer per-thread accumulation of the warp-grouped kernel.
template<typename Element, typename ElementAccum, typename index_t, int kHeadDimV, int kMaxSplits>
__global__ void __launch_bounds__(256, 1, 1)
flash_fwd_splitkv_mla_combine_128_kernel(__grid_constant__ const DecodingParams_fp8 params) {
    constexpr int kNThreads = 128;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int hs = params.h_k * params.q_seq_per_hk;
    const int batch_idx = bidx / hs;
    const int hs_idx = bidx % hs;
    const int head_idx = hs_idx / params.q_seq_per_hk;
    const int row = hs_idx - head_idx * params.q_seq_per_hk;

    static_assert(kHeadDimV % kNThreads == 0);
    constexpr int Elements = kHeadDimV / kNThreads;
    if (params.q_lens_ptr != nullptr) {
        const int query_idx = row / params.q_head_per_hk;
        const int q_len = __ldg(params.q_lens_ptr + batch_idx);
        FLASH_DEVICE_ASSERT(q_len >= 0 && q_len <= params.s_q);
        if (query_idx >= q_len) {
            if (tidx == 0) {
                reinterpret_cast<ElementAccum*>(
                    params.softmax_lse_ptr)[bidx] = -INFINITY;
            }
            auto o_ptr =
                reinterpret_cast<Element*>(params.o_ptr)
                + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride
                + row * params.o_row_stride;
#pragma unroll
            for (int i = 0; i < Elements; ++i) {
                o_ptr[tidx * Elements + i] = Element(0);
            }
            return;
        }
    }

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int actual_num_splits = __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    FLASH_DEVICE_ASSERT(actual_num_splits <= kMaxSplits);
    if (actual_num_splits <= 1) return;

    __shared__ ElementAccum sLseScale[kMaxSplits];

    const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
    const index_t row_offset_lse = bidx;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                   Shape<Int<kMaxSplits>>{}, make_stride(hs));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
        constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

        float local_lse[kNLsePerThread];
        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            local_lse[i] = split < actual_num_splits ? gLSEaccum(split) : -INFINITY;
        }

        float max_lse = -INFINITY;
        for (int i = 0; i < kNLsePerThread; ++i) max_lse = max(max_lse, local_lse[i]);
        for (int offset = 16; offset >= 1; offset /= 2) max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;

        float sum_lse = 0;
        for (int i = 0; i < kNLsePerThread; ++i) sum_lse = sum_lse + expf(local_lse[i] - max_lse);
        for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        const bool has_attendable_key = sum_lse > 0.f && sum_lse == sum_lse;
        float global_lse = has_attendable_key ? logf(sum_lse) + max_lse : -INFINITY;
        if (tidx == 0) gLSE(0) = global_lse;

        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            if (split < actual_num_splits) {
                sLseScale[split] =
                    has_attendable_key
                        ? expf(local_lse[i] - global_lse)
                        : 0.f;
            }
        }
    }
    __syncthreads();

    const index_t row_offset_oaccum = (split_offset * hs + hs_idx) * kHeadDimV;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kHeadDimV>>{}, Stride<_1>{});
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            Layout<Shape<Int<kNThreads>>>{},
            Layout<Shape<Int<Elements>>>{}));
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    for (int split = 0; split < actual_num_splits; ++split) {
        cute::copy(tOgOaccum, tOrOaccum);
        ElementAccum lse_scale = sLseScale[split];
        for (int i = 0; i < size(tOrO); ++i) {
            tOrO(i) += lse_scale * tOrOaccum(i);
        }
        tOgOaccum.data() = tOgOaccum.data() + hs * kHeadDimV;
    }

    Tensor rO = flash::convert_type<Element>(tOrO);
    auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride + row * params.o_row_stride;
    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + tidx * Elements), Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
    cute::copy(rO, gO);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Element, typename ElementAccum, typename index_t, int kHeadDimV, int kMaxSplits, int kRowsPerBlock>
__global__ void __launch_bounds__(kRowsPerBlock * 32, 1, 1)
flash_fwd_splitkv_mla_combine_warp_grouped_kernel(__grid_constant__ const DecodingParams_fp8 params) {
    constexpr int kWarpSize = 32;
    constexpr int kVectorSize = 4;
    static_assert(kRowsPerBlock > 0);
    static_assert(kRowsPerBlock <= 8);
    static_assert(kHeadDimV % (kWarpSize * kVectorSize) == 0);

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int hs = params.h_k * params.q_seq_per_hk;
    const int blocks_per_batch = cute::ceil_div(hs, kRowsPerBlock);
    const int batch_idx = bidx / blocks_per_batch;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = tidx % kWarpSize;
    const int hs_idx = (bidx - batch_idx * blocks_per_batch) * kRowsPerBlock + warp_idx;
    if (hs_idx >= hs) return;
    const int head_idx = hs_idx / params.q_seq_per_hk;
    const int row = hs_idx - head_idx * params.q_seq_per_hk;

    // See the 128-thread fallback above: ragged rows also write final output,
    // so they must not bypass the producer dependency.
    cudaGridDependencySynchronize();

    if (params.q_lens_ptr != nullptr) {
        const int query_idx = row / params.q_head_per_hk;
        const int q_len = __ldg(params.q_lens_ptr + batch_idx);
        FLASH_DEVICE_ASSERT(q_len >= 0 && q_len <= params.s_q);
        if (query_idx >= q_len) {
            if (lane_idx == 0) {
                reinterpret_cast<ElementAccum*>(
                    params.softmax_lse_ptr)[batch_idx * hs + hs_idx] = -INFINITY;
            }
            auto o_ptr =
                reinterpret_cast<Element*>(params.o_ptr)
                + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride
                + row * params.o_row_stride;
#pragma unroll
            for (int i = 0; i < kHeadDimV / (kWarpSize * kVectorSize); ++i) {
                cutlass::Array<Element, kVectorSize> zeros;
                zeros.clear();
                *reinterpret_cast<cutlass::Array<Element, kVectorSize> *>(
                    o_ptr + lane_idx * kVectorSize + i * kWarpSize * kVectorSize) = zeros;
            }
            return;
        }
    }

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int actual_num_splits = __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    FLASH_DEVICE_ASSERT(actual_num_splits <= kMaxSplits);
    if (actual_num_splits <= 1) return;

    __shared__ ElementAccum sLseScale[kRowsPerBlock][kMaxSplits];

    const index_t row_offset_oaccum = (split_offset * hs + hs_idx) * kHeadDimV;
    const index_t oaccum_split_stride = hs * kHeadDimV;
    const ElementAccum *oaccum_ptr =
        reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum;
    constexpr int kVectorsPerThread = kHeadDimV / (kWarpSize * kVectorSize);
    cutlass::Array<float4, kVectorsPerThread> prefetched_oaccum;
#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
        prefetched_oaccum[i] = *reinterpret_cast<const float4 *>(
            oaccum_ptr + lane_idx * kVectorSize + i * kWarpSize * kVectorSize);
    }

    const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
    const index_t row_offset_lse = batch_idx * hs + hs_idx;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                   Shape<Int<kMaxSplits>>{}, make_stride(hs));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, kWarpSize);

    float local_lse[kNLsePerThread];
#pragma unroll
    for (int i = 0; i < kNLsePerThread; ++i) {
        const int split = i * kWarpSize + lane_idx;
        local_lse[i] = split < actual_num_splits ? gLSEaccum(split) : -INFINITY;
    }

    float max_lse = -INFINITY;
#pragma unroll
    for (int i = 0; i < kNLsePerThread; ++i) max_lse = max(max_lse, local_lse[i]);
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
    max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

    float sum_lse = 0;
#pragma unroll
    for (int i = 0; i < kNLsePerThread; ++i) sum_lse = sum_lse + expf(local_lse[i] - max_lse);
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

    const bool has_attendable_key = sum_lse > 0.f && sum_lse == sum_lse;
    float global_lse = has_attendable_key ? logf(sum_lse) + max_lse : -INFINITY;
    if (lane_idx == 0) gLSE(0) = global_lse;

#pragma unroll
    for (int i = 0; i < kNLsePerThread; ++i) {
        const int split = i * kWarpSize + lane_idx;
        if (split < actual_num_splits) {
            sLseScale[warp_idx][split] =
                has_attendable_key
                    ? expf(local_lse[i] - global_lse)
                    : 0.f;
        }
    }
    __syncwarp();

    cutlass::Array<float4, kVectorsPerThread> result;
    result.clear();

    for (int split = 0; split < actual_num_splits; ++split) {
        const ElementAccum lse_scale = sLseScale[warp_idx][split];
#pragma unroll
        for (int i = 0; i < kVectorsPerThread; ++i) {
            const float4 data = prefetched_oaccum[i];
            result[i].x += lse_scale * data.x;
            result[i].y += lse_scale * data.y;
            result[i].z += lse_scale * data.z;
            result[i].w += lse_scale * data.w;
            if (split + 1 < actual_num_splits) {
                prefetched_oaccum[i] = *reinterpret_cast<const float4 *>(
                    oaccum_ptr + (split + 1) * oaccum_split_stride
                    + lane_idx * kVectorSize + i * kWarpSize * kVectorSize);
            }
        }
    }

    auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride + row * params.o_row_stride;
#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
        cutlass::Array<Element, kVectorSize> converted;
        converted[0] = Element(result[i].x);
        converted[1] = Element(result[i].y);
        converted[2] = Element(result[i].z);
        converted[3] = Element(result[i].w);
        *reinterpret_cast<cutlass::Array<Element, kVectorSize> *>(
            o_ptr + lane_idx * kVectorSize + i * kWarpSize * kVectorSize) = converted;
    }
}

__forceinline__ __device__ float canonical_merge_lse(
    const float left,
    const float right,
    float &left_weight,
    float &right_weight) {
    if (left == -INFINITY && right == -INFINITY) {
        left_weight = 0.f;
        right_weight = 0.f;
        return -INFINITY;
    }
    const float max_lse = max(left, right);
    const float sum = expf(left - max_lse) + expf(right - max_lse);
    const float merged = logf(sum) + max_lse;
    left_weight = left == -INFINITY ? 0.f : expf(left - merged);
    right_weight = right == -INFINITY ? 0.f : expf(right - merged);
    return merged;
}

__forceinline__ __device__ float4 canonical_merge_float4(
    const float4 left,
    const float4 right,
    const float left_weight,
    const float right_weight) {
    return make_float4(
        left_weight * left.x + right_weight * right.x,
        left_weight * left.y + right_weight * right.y,
        left_weight * left.z + right_weight * right.z,
        left_weight * left.w + right_weight * right.w);
}

// R14-P1 fixed-tree epilogue. One warp owns one output row. Lane zero builds
// the exact adjacent balanced LSE tree in shared memory; every lane then uses
// the same per-node weights to merge its float4 output fragments in the same
// tree. Inactive leaves carry (0, -inf), so a fixed 8-leaf launch also handles
// ragged sequences without changing the arithmetic graph.
template<typename Element, typename index_t, int kHeadDimV, int kRowsPerBlock>
__global__ void __launch_bounds__(kRowsPerBlock * 32, 1, 1)
flash_fwd_splitkv_mla_canonical_combine_kernel(
    __grid_constant__ const DecodingParams_fp8 params) {
    constexpr int kWarpSize = 32;
    constexpr int kVectorSize = 4;
    constexpr int kMaxChunks = 8;
    constexpr int kMaxInternalNodes = kMaxChunks - 1;
    static_assert(kHeadDimV % (kWarpSize * kVectorSize) == 0);

    const int tidx = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = tidx % kWarpSize;
    const int hs = params.h_k * params.q_seq_per_hk;
    const int blocks_per_batch = cute::ceil_div(hs, kRowsPerBlock);
    const int batch_idx = blockIdx.x / blocks_per_batch;
    const int hs_idx =
        (blockIdx.x - batch_idx * blocks_per_batch) * kRowsPerBlock
        + warp_idx;
    if (hs_idx >= hs) return;
    const int head_idx = hs_idx / params.q_seq_per_hk;
    const int row = hs_idx - head_idx * params.q_seq_per_hk;

    cudaGridDependencySynchronize();

    auto output_ptr =
        reinterpret_cast<Element *>(params.o_ptr)
        + batch_idx * params.o_batch_stride
        + head_idx * params.o_head_stride
        + row * params.o_row_stride;
    if (params.q_lens_ptr != nullptr) {
        const int query_idx = row / params.q_head_per_hk;
        const int q_len = __ldg(params.q_lens_ptr + batch_idx);
        FLASH_DEVICE_ASSERT(q_len >= 0 && q_len <= params.s_q);
        if (query_idx >= q_len) {
            if (lane_idx == 0) {
                reinterpret_cast<float *>(params.softmax_lse_ptr)[
                    batch_idx * hs + hs_idx] = -INFINITY;
            }
#pragma unroll
            for (int vector = 0;
                 vector < kHeadDimV / (kWarpSize * kVectorSize);
                 ++vector) {
                cutlass::Array<Element, kVectorSize> zeros;
                zeros.clear();
                *reinterpret_cast<cutlass::Array<Element, kVectorSize> *>(
                    output_ptr + lane_idx * kVectorSize
                    + vector * kWarpSize * kVectorSize) = zeros;
            }
            return;
        }
    }

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int actual_num_splits =
        __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    const int num_chunks = params.canonical_num_chunks;
    FLASH_DEVICE_ASSERT(
        actual_num_splits == num_chunks &&
        (num_chunks == 2 || num_chunks == 4 || num_chunks == 8));

    __shared__ float sLeftWeight[kRowsPerBlock][kMaxInternalNodes];
    __shared__ float sRightWeight[kRowsPerBlock][kMaxInternalNodes];
    __shared__ float sFinalLse[kRowsPerBlock];

    const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
    const float *lseaccum_ptr =
        reinterpret_cast<const float *>(params.softmax_lseaccum_ptr)
        + row_offset_lseaccum;
    if (lane_idx == 0) {
        float tree_lse[2 * kMaxChunks - 1];
        const int leaf_base = num_chunks - 1;
#pragma unroll
        for (int chunk = 0; chunk < kMaxChunks; ++chunk) {
            if (chunk < num_chunks) {
                tree_lse[leaf_base + chunk] =
                    lseaccum_ptr[chunk * hs];
            }
        }
        for (int node = leaf_base - 1; node >= 0; --node) {
            tree_lse[node] = canonical_merge_lse(
                tree_lse[2 * node + 1],
                tree_lse[2 * node + 2],
                sLeftWeight[warp_idx][node],
                sRightWeight[warp_idx][node]);
        }
        sFinalLse[warp_idx] = tree_lse[0];
        reinterpret_cast<float *>(params.softmax_lse_ptr)[
            batch_idx * hs + hs_idx] = tree_lse[0];
    }
    __syncwarp();

    const index_t row_offset_oaccum =
        (split_offset * hs + hs_idx) * kHeadDimV;
    const index_t split_stride = hs * kHeadDimV;
    const float *oaccum_ptr =
        reinterpret_cast<const float *>(params.oaccum_ptr)
        + row_offset_oaccum;
    constexpr int kVectorsPerThread =
        kHeadDimV / (kWarpSize * kVectorSize);

#pragma unroll
    for (int vector = 0; vector < kVectorsPerThread; ++vector) {
        const index_t element =
            lane_idx * kVectorSize + vector * kWarpSize * kVectorSize;
        auto load_chunk = [&](const int chunk) {
            return *reinterpret_cast<const float4 *>(
                oaccum_ptr + chunk * split_stride + element);
        };
        float4 result;
        if (num_chunks == 2) {
            result = canonical_merge_float4(
                load_chunk(0), load_chunk(1),
                sLeftWeight[warp_idx][0],
                sRightWeight[warp_idx][0]);
        } else if (num_chunks == 4) {
            const float4 left = canonical_merge_float4(
                load_chunk(0), load_chunk(1),
                sLeftWeight[warp_idx][1],
                sRightWeight[warp_idx][1]);
            const float4 right = canonical_merge_float4(
                load_chunk(2), load_chunk(3),
                sLeftWeight[warp_idx][2],
                sRightWeight[warp_idx][2]);
            result = canonical_merge_float4(
                left, right,
                sLeftWeight[warp_idx][0],
                sRightWeight[warp_idx][0]);
        } else {
            const float4 left01 = canonical_merge_float4(
                load_chunk(0), load_chunk(1),
                sLeftWeight[warp_idx][3],
                sRightWeight[warp_idx][3]);
            const float4 left23 = canonical_merge_float4(
                load_chunk(2), load_chunk(3),
                sLeftWeight[warp_idx][4],
                sRightWeight[warp_idx][4]);
            const float4 left = canonical_merge_float4(
                left01, left23,
                sLeftWeight[warp_idx][1],
                sRightWeight[warp_idx][1]);
            const float4 right45 = canonical_merge_float4(
                load_chunk(4), load_chunk(5),
                sLeftWeight[warp_idx][5],
                sRightWeight[warp_idx][5]);
            const float4 right67 = canonical_merge_float4(
                load_chunk(6), load_chunk(7),
                sLeftWeight[warp_idx][6],
                sRightWeight[warp_idx][6]);
            const float4 right = canonical_merge_float4(
                right45, right67,
                sLeftWeight[warp_idx][2],
                sRightWeight[warp_idx][2]);
            result = canonical_merge_float4(
                left, right,
                sLeftWeight[warp_idx][0],
                sRightWeight[warp_idx][0]);
        }

        cutlass::Array<Element, kVectorSize> converted;
        converted[0] = Element(result.x);
        converted[1] = Element(result.y);
        converted[2] = Element(result.z);
        converted[3] = Element(result.w);
        *reinterpret_cast<cutlass::Array<Element, kVectorSize> *>(
            output_ptr + element) = converted;
    }
}

} // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename SharedStorage>
void run_flash_splitkv_fwd_mla(DecodingParams_fp8 &params, cudaStream_t stream) {
    FLASH_ASSERT(params.page_block_size == Kernel_traits::kBlockN);
    const int num_m_block = cute::ceil_div(params.q_seq_per_hk, Kernel_traits::kBlockM);
    constexpr int kSmallCombineRows = 512;
    constexpr int kEightRowCombineRows = 1024;
    const int rows_per_batch = params.h_k * params.q_seq_per_hk;
    const int total_rows = params.b * rows_per_batch;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        constexpr size_t smem_size = sizeof(SharedStorage);
        if (total_rows <= kSmallCombineRows &&
            params.canonical_num_chunks == 0) {
            auto kernel = &flash::flash_fwd_splitkv_mla_kernel<
                Kernel_traits, Is_causal, SharedStorage, false>;
            CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            kernel<<<dim3(num_m_block, params.h_k, params.num_sm_parts),
                     Kernel_traits::kNThreads, smem_size, stream>>>(params);
        } else {
            auto kernel = &flash::flash_fwd_splitkv_mla_kernel<
                Kernel_traits, Is_causal, SharedStorage, true>;
            CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            cudaLaunchAttribute attributes[1];
            attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[0].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t config = {
                dim3(num_m_block, params.h_k, params.num_sm_parts),
                dim3(Kernel_traits::kNThreads, 1, 1),
                smem_size,
                stream,
                attributes,
                1
            };
            CHECK_CUDA(cudaLaunchKernelEx(&config, kernel, params));
        }
    });
    CHECK_CUDA_KERNEL_LAUNCH();

    if (params.canonical_return_leaves) {
        return;
    }

    if (params.canonical_num_chunks > 0) {
        auto launch_canonical_combine = [&](auto rows_per_block) {
            constexpr int kRowsPerBlock = decltype(rows_per_block)::value;
            const int combine_blocks_per_batch = cute::ceil_div(
                rows_per_batch, kRowsPerBlock);
            auto combine_kernel =
                &flash::flash_fwd_splitkv_mla_canonical_combine_kernel<
                    typename Kernel_traits::ElementO,
                    typename Kernel_traits::index_t,
                    Kernel_traits::kHeadDimV,
                    kRowsPerBlock>;
            cudaLaunchAttribute attributes[1];
            attributes[0].id =
                cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[0].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t config = {
                dim3(params.b * combine_blocks_per_batch, 1, 1),
                dim3(kRowsPerBlock * 32, 1, 1),
                0,
                stream,
                attributes,
                1
            };
            CHECK_CUDA(cudaLaunchKernelEx(&config, combine_kernel, params));
        };
        if (total_rows == kEightRowCombineRows) {
            launch_canonical_combine(cute::Int<8>{});
        } else {
            launch_canonical_combine(cute::Int<4>{});
        }
        CHECK_CUDA_KERNEL_LAUNCH();
        return;
    }

    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        if (total_rows <= kSmallCombineRows) {
            auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_128_kernel<
                    typename Kernel_traits::ElementO, typename Kernel_traits::ElementAccum,
                    typename Kernel_traits::index_t, Kernel_traits::kHeadDimV, kMaxSplits>;
            combine_kernel<<<total_rows, 128, 0, stream>>>(params);
        } else {
            auto launch_grouped_combine = [&](auto rows_per_block) {
                constexpr int kRowsPerBlock = decltype(rows_per_block)::value;
                const int combine_blocks_per_batch = cute::ceil_div(
                    rows_per_batch, kRowsPerBlock);
                auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_warp_grouped_kernel<
                        typename Kernel_traits::ElementO, typename Kernel_traits::ElementAccum,
                        typename Kernel_traits::index_t, Kernel_traits::kHeadDimV, kMaxSplits,
                        kRowsPerBlock>;
                cudaLaunchAttribute attributes[1];
                attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                attributes[0].val.programmaticStreamSerializationAllowed = 1;
                cudaLaunchConfig_t config = {
                    dim3(params.b * combine_blocks_per_batch, 1, 1),
                    dim3(kRowsPerBlock * 32, 1, 1),
                    0,
                    stream,
                    attributes,
                    1
                };
                CHECK_CUDA(cudaLaunchKernelEx(&config, combine_kernel, params));
            };
            if (total_rows == kEightRowCombineRows) {
                launch_grouped_combine(cute::Int<8>{});
            } else {
                launch_grouped_combine(cute::Int<4>{});
            }
        }
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(DecodingParams_fp8 &params, cudaStream_t stream) {
    static_assert(Headdim == 576);
    FLASH_ASSERT(params.d_v == 512);
    using Kernel_traits = Flash_fwd_kernel_traits_mla<576, 64, 64, 8, T, To, 512>;
    run_flash_splitkv_fwd_mla<Kernel_traits, flash::SharedStorageMLA<Kernel_traits>>(params, stream);
}
