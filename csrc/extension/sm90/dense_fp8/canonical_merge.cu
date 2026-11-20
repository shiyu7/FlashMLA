#include "canonical_merge.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kMergeThreads = 128;
constexpr int kPeerWorldSize = 8;
constexpr int kPrepareThreads = 256;
constexpr int kLeafEpoch = 0;
constexpr int kLevel1Epoch = 1;
constexpr int kLevel2Epoch = 2;
constexpr int kReadEpoch = 3;
constexpr int kFlagsPerRow = 4;
constexpr uint32_t kMaxFenceSpins = 100'000'000;

struct CanonicalMergeParams {
    const float* left_out;
    const float* right_out;
    const float* left_lse;
    const float* right_lse;
    float* merged_out;
    float* merged_lse;
    int64_t rows;
    int q_len;
    int num_heads;
    int head_dim_v;
};

struct PeerPrepareParams {
    const int64_t* flag_ptrs;
    int64_t rows;
    int dcp_rank;
};

struct PeerTreeParams {
    const int64_t* output_ptrs;
    const int64_t* lse_ptrs;
    const int64_t* flag_ptrs;
    float* local_out;
    float* local_lse;
    int64_t rows;
    int q_len;
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

__global__ void canonical_peer_tree_prepare_kernel(PeerPrepareParams params) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= params.rows || params.dcp_rank == 0) {
        return;
    }
    const int64_t* flags =
        peer_flag_base(params.flag_ptrs, params.dcp_rank)
        + row * kFlagsPerRow;
    const uint64_t previous_epoch =
        atomic_load_acquire_system(flags + kLeafEpoch);
    wait_for_epoch(flags + kReadEpoch, previous_epoch);
}

__global__ void canonical_peer_tree_merge_kernel(PeerTreeParams params) {
    const int64_t row = blockIdx.x;
    const int64_t batch = row / (params.q_len * params.num_heads);
    const int64_t query_head = row - batch * params.q_len * params.num_heads;
    const int query = query_head / params.num_heads;
    const int head = query_head - query * params.num_heads;
    const int64_t lse_offset =
        (batch * params.num_heads + head) * params.q_len + query;
    int64_t* local_flags =
        peer_flag_base(params.flag_ptrs, params.dcp_rank)
        + row * kFlagsPerRow;

    __shared__ uint64_t epoch;
    __shared__ float left_weight;
    __shared__ float right_weight;
    __shared__ float final_lse;

    // The leaf copy ran earlier on this stream. Flush every participating
    // thread before lane zero publishes the row epoch to peer GPUs.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        epoch = atomic_inc_release_system(local_flags + kLeafEpoch);
    }
    __syncthreads();

#pragma unroll
    for (int level = 0; level < 3; ++level) {
        const int stride = 1 << level;
        const int period = stride << 1;
        const int offset = params.dcp_rank % period;
        if (offset == stride) {
            return;
        }
        if (offset != 0 || params.dcp_rank + stride >= kPeerWorldSize) {
            continue;
        }

        const int peer = params.dcp_rank + stride;
        int64_t* peer_flags =
            peer_flag_base(params.flag_ptrs, peer) + row * kFlagsPerRow;
        if (threadIdx.x == 0) {
            wait_for_epoch(peer_flags + level, epoch);
            const auto* peer_lse =
                reinterpret_cast<const float*>(params.lse_ptrs[peer]);
            const float left = params.local_lse[lse_offset];
            const float right = peer_lse[lse_offset];
            if (left == -CUDART_INF_F && right == -CUDART_INF_F) {
                left_weight = 0.f;
                right_weight = 0.f;
                final_lse = -CUDART_INF_F;
            } else {
                const float max_lse = max(left, right);
                const float sum =
                    expf(left - max_lse) + expf(right - max_lse);
                final_lse = logf(sum) + max_lse;
                left_weight =
                    left == -CUDART_INF_F ? 0.f
                                          : expf(left - final_lse);
                right_weight =
                    right == -CUDART_INF_F ? 0.f
                                           : expf(right - final_lse);
            }
            params.local_lse[lse_offset] = final_lse;
        }
        __syncthreads();

        const int64_t vector_base = row * (params.head_dim_v / 4);
        const auto* peer_output = reinterpret_cast<const float4*>(
            params.output_ptrs[peer]);
        auto* local_output = reinterpret_cast<float4*>(params.local_out);
        for (int vector = threadIdx.x;
             vector < params.head_dim_v / 4;
             vector += blockDim.x) {
            const float4 lhs = local_output[vector_base + vector];
            const float4 rhs = peer_output[vector_base + vector];
            local_output[vector_base + vector] = make_float4(
                left_weight * lhs.x + right_weight * rhs.x,
                left_weight * lhs.y + right_weight * rhs.y,
                left_weight * lhs.z + right_weight * rhs.z,
                left_weight * lhs.w + right_weight * rhs.w);
        }

        // All row fragments must be visible before this stage is published.
        __syncthreads();
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            if (level < 2) {
                atomic_inc_release_system(local_flags + level + 1);
            }
            atomic_inc_release_system(peer_flags + kReadEpoch);
        }
        __syncthreads();
    }
}

__global__ void canonical_merge_pair_kernel(CanonicalMergeParams params) {
    const int64_t row = blockIdx.x;
    if (row >= params.rows) {
        return;
    }
    const int64_t batch = row / (params.q_len * params.num_heads);
    const int64_t query_head = row - batch * params.q_len * params.num_heads;
    const int query = query_head / params.num_heads;
    const int head = query_head - query * params.num_heads;
    const int64_t lse_offset =
        (batch * params.num_heads + head) * params.q_len + query;

    __shared__ float left_weight;
    __shared__ float right_weight;
    __shared__ float final_lse;
    if (threadIdx.x == 0) {
        const float left = params.left_lse[lse_offset];
        const float right = params.right_lse[lse_offset];
        if (left == -CUDART_INF_F && right == -CUDART_INF_F) {
            left_weight = 0.f;
            right_weight = 0.f;
            final_lse = -CUDART_INF_F;
        } else {
            const float max_lse = max(left, right);
            const float sum =
                expf(left - max_lse) + expf(right - max_lse);
            final_lse = logf(sum) + max_lse;
            left_weight =
                left == -CUDART_INF_F ? 0.f : expf(left - final_lse);
            right_weight =
                right == -CUDART_INF_F ? 0.f : expf(right - final_lse);
        }
        params.merged_lse[lse_offset] = final_lse;
    }
    __syncthreads();

    const int64_t vector_base = row * (params.head_dim_v / 4);
    const auto* left = reinterpret_cast<const float4*>(params.left_out);
    const auto* right = reinterpret_cast<const float4*>(params.right_out);
    auto* output = reinterpret_cast<float4*>(params.merged_out);
    for (int vector = threadIdx.x;
         vector < params.head_dim_v / 4;
         vector += blockDim.x) {
        const float4 lhs = left[vector_base + vector];
        const float4 rhs = right[vector_base + vector];
        output[vector_base + vector] = make_float4(
            left_weight * lhs.x + right_weight * rhs.x,
            left_weight * lhs.y + right_weight * rhs.y,
            left_weight * lhs.z + right_weight * rhs.z,
            left_weight * lhs.w + right_weight * rhs.w);
    }
}

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

void canonical_merge_pair_out(
    const at::Tensor& left_out,
    const at::Tensor& left_lse,
    const at::Tensor& right_out,
    const at::Tensor& right_lse,
    const at::Tensor& merged_out,
    const at::Tensor& merged_lse) {
    check_cuda_contiguous(left_out, "left_out");
    check_cuda_contiguous(left_lse, "left_lse");
    check_cuda_contiguous(right_out, "right_out");
    check_cuda_contiguous(right_lse, "right_lse");
    check_cuda_contiguous(merged_out, "merged_out");
    check_cuda_contiguous(merged_lse, "merged_lse");
    for (const auto* tensor : {
             &left_out, &left_lse, &right_out, &right_lse,
             &merged_out, &merged_lse}) {
        TORCH_CHECK(
            tensor->scalar_type() == at::kFloat,
            "canonical merge tensors must have dtype float32");
        TORCH_CHECK(
            tensor->device() == left_out.device(),
            "canonical merge tensors must be on the same CUDA device");
    }
    TORCH_CHECK(
        left_out.dim() == 4,
        "canonical outputs must have shape [B, Q, H, Dv]");
    TORCH_CHECK(
        left_lse.dim() == 3,
        "canonical LSE tensors must have shape [B, H, Q]");
    TORCH_CHECK(
        left_out.sizes() == right_out.sizes() &&
            left_out.sizes() == merged_out.sizes(),
        "canonical output shapes must match");
    TORCH_CHECK(
        left_lse.sizes() == right_lse.sizes() &&
            left_lse.sizes() == merged_lse.sizes(),
        "canonical LSE shapes must match");

    const int64_t batch = left_out.size(0);
    const int64_t q_len = left_out.size(1);
    const int64_t num_heads = left_out.size(2);
    const int64_t head_dim_v = left_out.size(3);
    TORCH_CHECK(
        left_lse.sizes() == at::IntArrayRef({batch, num_heads, q_len}),
        "canonical LSE shape must be [B, H, Q]");
    TORCH_CHECK(
        batch > 0 && q_len > 0 && num_heads > 0,
        "canonical merge dimensions must be positive");
    TORCH_CHECK(
        head_dim_v > 0 && head_dim_v % 4 == 0,
        "canonical merge head_dim_v must be positive and divisible by 4");

    const auto* properties = at::cuda::getDeviceProperties(left_out.device().index());
    TORCH_CHECK(
        properties->major == 9 && properties->minor == 0,
        "canonical pair merge is only supported on SM90");
    const int64_t rows = batch * q_len * num_heads;
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max(),
        "canonical merge grid is too large");

    const c10::cuda::CUDAGuard device_guard(left_out.device());
    const CanonicalMergeParams params{
        left_out.data_ptr<float>(),
        right_out.data_ptr<float>(),
        left_lse.data_ptr<float>(),
        right_lse.data_ptr<float>(),
        merged_out.data_ptr<float>(),
        merged_lse.data_ptr<float>(),
        rows,
        static_cast<int>(q_len),
        static_cast<int>(num_heads),
        static_cast<int>(head_dim_v),
    };
    canonical_merge_pair_kernel<<<
        static_cast<int>(rows),
        kMergeThreads,
        0,
        at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void canonical_peer_tree_prepare(
    const at::Tensor& flag_ptrs,
    int64_t rows,
    int64_t dcp_rank) {
    check_cuda_contiguous(flag_ptrs, "flag_ptrs");
    TORCH_CHECK(
        flag_ptrs.scalar_type() == at::kLong &&
            flag_ptrs.numel() == kPeerWorldSize,
        "flag_ptrs must be a contiguous int64 CUDA tensor with 8 elements");
    TORCH_CHECK(rows > 0, "rows must be positive");
    TORCH_CHECK(
        dcp_rank >= 0 && dcp_rank < kPeerWorldSize,
        "dcp_rank must be in [0, 8)");
    const c10::cuda::CUDAGuard device_guard(flag_ptrs.device());
    const int64_t blocks64 =
        (rows + kPrepareThreads - 1) / kPrepareThreads;
    TORCH_CHECK(
        blocks64 <= std::numeric_limits<int>::max(),
        "canonical peer prepare grid is too large");
    const PeerPrepareParams params{
        flag_ptrs.data_ptr<int64_t>(),
        rows,
        static_cast<int>(dcp_rank),
    };
    canonical_peer_tree_prepare_kernel<<<
        static_cast<int>(blocks64),
        kPrepareThreads,
        0,
        at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void canonical_peer_tree_merge(
    const at::Tensor& output_ptrs,
    const at::Tensor& lse_ptrs,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& local_out,
    const at::Tensor& local_lse) {
    check_cuda_contiguous(output_ptrs, "output_ptrs");
    check_cuda_contiguous(lse_ptrs, "lse_ptrs");
    check_cuda_contiguous(flag_ptrs, "flag_ptrs");
    check_cuda_contiguous(local_out, "local_out");
    check_cuda_contiguous(local_lse, "local_lse");
    for (const auto* ptrs : {&output_ptrs, &lse_ptrs, &flag_ptrs}) {
        TORCH_CHECK(
            ptrs->scalar_type() == at::kLong &&
                ptrs->numel() == kPeerWorldSize,
            "peer pointer tensors must be contiguous int64 CUDA tensors "
            "with 8 elements");
        TORCH_CHECK(
            ptrs->device() == local_out.device(),
            "peer pointer tensors must be on the local output device");
    }
    TORCH_CHECK(
        local_out.scalar_type() == at::kFloat &&
            local_lse.scalar_type() == at::kFloat,
        "canonical peer states must have dtype float32");
    TORCH_CHECK(
        local_out.device() == local_lse.device(),
        "canonical peer output and LSE must share a CUDA device");
    TORCH_CHECK(
        local_out.dim() == 4 && local_lse.dim() == 3,
        "canonical peer state shapes must be [B,Q,H,Dv] and [B,H,Q]");
    const int64_t batch = local_out.size(0);
    const int64_t q_len = local_out.size(1);
    const int64_t num_heads = local_out.size(2);
    const int64_t head_dim_v = local_out.size(3);
    TORCH_CHECK(
        local_lse.sizes() == at::IntArrayRef({batch, num_heads, q_len}),
        "canonical peer LSE shape must be [B,H,Q]");
    TORCH_CHECK(
        batch > 0 && q_len > 0 && num_heads > 0 &&
            head_dim_v > 0 && head_dim_v % 4 == 0,
        "canonical peer tree requires positive dimensions and Dv % 4 == 0");
    TORCH_CHECK(
        dcp_rank >= 0 && dcp_rank < kPeerWorldSize,
        "dcp_rank must be in [0, 8)");
    const int64_t rows = batch * q_len * num_heads;
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max(),
        "canonical peer tree grid is too large");

    const c10::cuda::CUDAGuard device_guard(local_out.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major == 9 && properties->minor == 0,
        "canonical peer tree is only supported on SM90");
    const PeerTreeParams params{
        output_ptrs.data_ptr<int64_t>(),
        lse_ptrs.data_ptr<int64_t>(),
        flag_ptrs.data_ptr<int64_t>(),
        local_out.data_ptr<float>(),
        local_lse.data_ptr<float>(),
        rows,
        static_cast<int>(q_len),
        static_cast<int>(num_heads),
        static_cast<int>(head_dim_v),
        static_cast<int>(dcp_rank),
    };
    canonical_peer_tree_merge_kernel<<<
        static_cast<int>(rows),
        kMergeThreads,
        0,
        at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
