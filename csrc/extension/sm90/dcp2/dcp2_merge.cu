#include "dcp2_merge.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kDCPSize = 2;
constexpr int kWriteEpoch = 0;
constexpr int kReadEpoch = 1;
constexpr int kThreads = 256;
constexpr uint32_t kMaxFenceSpins = 100'000'000;

struct PrepareParams {
    const int64_t* flag_ptrs;
    int64_t rows;
    int64_t num_heads;
    int dcp_rank;
};

struct MergeParams {
    const __nv_bfloat16* rank0_out;
    const __nv_bfloat16* rank1_out;
    const float* rank0_lse;
    const float* rank1_lse;
    const int64_t* flag_ptrs;
    __nv_bfloat16* merged_out;
    float* merged_lse;
    int64_t rows;
    int num_heads;
    int head_dim_v;
    int dcp_rank;
};

__device__ __forceinline__ uint64_t atomic_load_acquire_system(
    const int64_t* ptr) {
    uint64_t value;
    const uint64_t zero = 0;
    asm volatile(
        "atom.global.acquire.sys.add.u64 %0, [%1], %2;"
        : "=l"(value)
        : "l"(ptr), "l"(zero)
        : "memory");
    return value;
}

__device__ __forceinline__ uint64_t atomic_inc_release_system(int64_t* ptr) {
    uint64_t old;
    const uint64_t one = 1;
    asm volatile(
        "atom.global.release.sys.add.u64 %0, [%1], %2;"
        : "=l"(old)
        : "l"(ptr), "l"(one)
        : "memory");
    return old + 1;
}

__device__ __forceinline__ int64_t* peer_flag_base(
    const int64_t* flag_ptrs,
    int rank) {
    return reinterpret_cast<int64_t*>(flag_ptrs[rank]);
}

__device__ __forceinline__ void wait_for_epoch(
    const int64_t* ptr,
    uint64_t epoch) {
    for (uint32_t spin = 0; spin < kMaxFenceSpins; ++spin) {
        if (atomic_load_acquire_system(ptr) >= epoch) {
            return;
        }
    }
    asm volatile("trap;");
}

__global__ void dcp2_prepare_attention_kernel(PrepareParams params) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= params.rows) {
        return;
    }

    const int heads_per_rank = params.num_heads / kDCPSize;
    const int head = row % params.num_heads;
    const int owner = head / heads_per_rank;
    const int64_t* local_flags =
        peer_flag_base(params.flag_ptrs, params.dcp_rank) + row * 2;
    const uint64_t previous_epoch =
        atomic_load_acquire_system(local_flags + kWriteEpoch);
    const int64_t* owner_flags =
        peer_flag_base(params.flag_ptrs, owner) + row * 2;
    wait_for_epoch(owner_flags + kReadEpoch, previous_epoch);
}

__global__ void dcp2_merge_attention_kernel(MergeParams params) {
    const int64_t row = blockIdx.x;
    const int head = row % params.num_heads;
    const int heads_per_rank = params.num_heads / kDCPSize;
    const int owner = head / heads_per_rank;
    int64_t* local_flags =
        peer_flag_base(params.flag_ptrs, params.dcp_rank) + row * 2;

    __shared__ float weights[2];
    __shared__ float global_lse;

    if (threadIdx.x == 0) {
        // FlashMLA ran earlier on this stream and wrote the entire local row.
        // The system fence plus release publication makes those writes visible
        // to the owner GPU before it consumes this epoch.
        __threadfence_system();
        atomic_inc_release_system(local_flags + kWriteEpoch);

        if (owner == params.dcp_rank) {
            const uint64_t expected_epoch =
                atomic_load_acquire_system(local_flags + kReadEpoch) + 1;
            for (int peer = 0; peer < kDCPSize; ++peer) {
                const int64_t* peer_flags =
                    peer_flag_base(params.flag_ptrs, peer) + row * 2;
                wait_for_epoch(peer_flags + kWriteEpoch, expected_epoch);
            }

            const float lse0 = params.rank0_lse[row];
            const float lse1 = params.rank1_lse[row];
            const float max_lse = fmaxf(lse0, lse1);
            if (max_lse == -CUDART_INF_F) {
                weights[0] = 0.0f;
                weights[1] = 0.0f;
                global_lse = -CUDART_INF_F;
            } else {
                const float exp0 = expf(lse0 - max_lse);
                const float exp1 = expf(lse1 - max_lse);
                const float denominator = exp0 + exp1;
                weights[0] = exp0 / denominator;
                weights[1] = exp1 / denominator;
                global_lse = logf(denominator) + max_lse;
            }
        }
    }
    __syncthreads();

    // Every rank publishes all rows, but only the owner rank reads peer data
    // and materializes its reduce-scatter-equivalent head shard.
    if (owner != params.dcp_rank) {
        return;
    }

    const int local_head = head - params.dcp_rank * heads_per_rank;
    const int64_t bq = row / params.num_heads;
    const int64_t input_offset = row * params.head_dim_v;
    const int64_t output_offset =
        (bq * heads_per_rank + local_head) * params.head_dim_v;

    const auto* in0 = reinterpret_cast<const __nv_bfloat162*>(
        params.rank0_out + input_offset);
    const auto* in1 = reinterpret_cast<const __nv_bfloat162*>(
        params.rank1_out + input_offset);
    auto* output = reinterpret_cast<__nv_bfloat162*>(
        params.merged_out + output_offset);
    const int pairs = params.head_dim_v / 2;
    for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const float2 value0 = __bfloat1622float2(in0[pair]);
        const float2 value1 = __bfloat1622float2(in1[pair]);
        output[pair] = __floats2bfloat162_rn(
            weights[0] * value0.x + weights[1] * value1.x,
            weights[0] * value0.y + weights[1] * value1.y);
    }
    if (threadIdx.x == 0) {
        params.merged_lse[bq * heads_per_rank + local_head] = global_lse;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        // Only the row owner acknowledges consumption. On the next replay,
        // both producers wait on this counter before overwriting the row.
        atomic_inc_release_system(local_flags + kReadEpoch);
    }
}

void check_sm90() {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major == 9 && properties->minor == 0,
        "DCP2 peer attention merge is only supported on SM90");
}

void check_common_geometry(
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t dcp_rank) {
    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(
        seqlen_q >= 1 && seqlen_q <= 8,
        "DCP2 peer attention merge requires 1 <= seqlen_q <= 8");
    TORCH_CHECK(
        num_heads > 0 && num_heads % kDCPSize == 0,
        "num_heads must be positive and divisible by 2");
    TORCH_CHECK(
        dcp_rank >= 0 && dcp_rank < kDCPSize,
        "dcp_rank must be 0 or 1");
}

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

void dcp2_prepare_attention(
    const at::Tensor& flag_ptrs,
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t dcp_rank) {
    check_cuda_contiguous(flag_ptrs, "flag_ptrs");
    TORCH_CHECK(
        flag_ptrs.scalar_type() == at::kLong && flag_ptrs.numel() == kDCPSize,
        "flag_ptrs must be a contiguous int64 CUDA tensor with 2 elements");

    const c10::cuda::CUDAGuard device_guard(flag_ptrs.device());
    check_sm90();
    check_common_geometry(batch_size, seqlen_q, num_heads, dcp_rank);
    const int64_t rows = batch_size * seqlen_q * num_heads;
    const int blocks = static_cast<int>((rows + kThreads - 1) / kThreads);
    const PrepareParams params{
        flag_ptrs.data_ptr<int64_t>(),
        rows,
        num_heads,
        static_cast<int>(dcp_rank),
    };
    dcp2_prepare_attention_kernel<<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dcp2_merge_attention(
    const at::Tensor& rank0_out,
    const at::Tensor& rank1_out,
    const at::Tensor& rank0_lse,
    const at::Tensor& rank1_lse,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& merged_out,
    const at::Tensor& merged_lse) {
    check_cuda_contiguous(rank0_out, "rank0_out");
    check_cuda_contiguous(rank1_out, "rank1_out");
    check_cuda_contiguous(rank0_lse, "rank0_lse");
    check_cuda_contiguous(rank1_lse, "rank1_lse");
    check_cuda_contiguous(flag_ptrs, "flag_ptrs");
    check_cuda_contiguous(merged_out, "merged_out");
    check_cuda_contiguous(merged_lse, "merged_lse");

    TORCH_CHECK(
        rank0_out.scalar_type() == at::kBFloat16 &&
            rank1_out.scalar_type() == at::kBFloat16 &&
            merged_out.scalar_type() == at::kBFloat16,
        "attention outputs must have dtype bfloat16");
    TORCH_CHECK(
        rank0_lse.scalar_type() == at::kFloat &&
            rank1_lse.scalar_type() == at::kFloat &&
            merged_lse.scalar_type() == at::kFloat,
        "attention LSE tensors must have dtype float32");
    TORCH_CHECK(
        flag_ptrs.scalar_type() == at::kLong && flag_ptrs.numel() == kDCPSize,
        "flag_ptrs must be a contiguous int64 CUDA tensor with 2 elements");
    TORCH_CHECK(
        rank0_out.dim() == 4,
        "peer outputs must have shape [B, Q, H, Dv]");
    TORCH_CHECK(
        rank0_lse.dim() == 3,
        "peer LSE tensors must have shape [B, Q, H]");
    TORCH_CHECK(
        rank0_out.sizes() == rank1_out.sizes(),
        "rank0_out and rank1_out must have the same shape");
    TORCH_CHECK(
        rank0_lse.sizes() == rank1_lse.sizes(),
        "rank0_lse and rank1_lse must have the same shape");

    const int64_t batch_size = rank0_out.size(0);
    const int64_t seqlen_q = rank0_out.size(1);
    const int64_t num_heads = rank0_out.size(2);
    const int64_t head_dim_v = rank0_out.size(3);
    check_common_geometry(batch_size, seqlen_q, num_heads, dcp_rank);
    TORCH_CHECK(
        head_dim_v > 0 && head_dim_v % 2 == 0,
        "head_dim_v must be a positive even number");
    TORCH_CHECK(
        rank0_lse.sizes() ==
            at::IntArrayRef({batch_size, seqlen_q, num_heads}),
        "peer LSE tensors must match output [B, Q, H]");
    TORCH_CHECK(
        merged_out.sizes() ==
            at::IntArrayRef(
                {batch_size, seqlen_q, num_heads / 2, head_dim_v}),
        "merged_out must have shape [B, Q, H / 2, Dv]");
    TORCH_CHECK(
        merged_lse.sizes() ==
            at::IntArrayRef({batch_size, seqlen_q, num_heads / 2}),
        "merged_lse must have shape [B, Q, H / 2]");

    const auto device = rank0_out.device();
    for (const auto* tensor : {
             &rank1_out,
             &rank0_lse,
             &rank1_lse,
             &flag_ptrs,
             &merged_out,
             &merged_lse}) {
        TORCH_CHECK(
            tensor->device() == device,
            "all DCP2 merge tensors must be mapped on the same CUDA device");
    }

    const c10::cuda::CUDAGuard device_guard(device);
    check_sm90();
    const int64_t rows = batch_size * seqlen_q * num_heads;
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max(),
        "DCP2 merge grid is too large");
    const MergeParams params{
        reinterpret_cast<const __nv_bfloat16*>(rank0_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(rank1_out.data_ptr()),
        rank0_lse.data_ptr<float>(),
        rank1_lse.data_ptr<float>(),
        flag_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(merged_out.data_ptr()),
        merged_lse.data_ptr<float>(),
        rows,
        static_cast<int>(num_heads),
        static_cast<int>(head_dim_v),
        static_cast<int>(dcp_rank),
    };
    dcp2_merge_attention_kernel<<<
        static_cast<int>(rows),
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
