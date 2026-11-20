#include "canonical_peer_reduce_scatter.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kWorldSize = 8;
constexpr int kMergeThreads = 128;
constexpr int kPrepareThreads = 256;
constexpr int kLeafEpoch = 0;
constexpr int kReadEpoch = 3;
constexpr int kFlagsPerRow = 4;
constexpr uint32_t kMaxFenceSpins = 100'000'000;

struct PrepareParams {
    const int64_t* flag_ptrs;
    int64_t rows;
    int dcp_rank;
};

struct ReduceScatterParams {
    const int64_t* output_ptrs;
    const int64_t* lse_ptrs;
    const int64_t* flag_ptrs;
    float* local_out;
    float* local_lse;
    float* compact_out;
    float* compact_lse;
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

__device__ __forceinline__ int64_t* flag_base(
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

__global__ void prepare_kernel(PrepareParams params) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= params.rows) {
        return;
    }
    const int64_t* flags =
        flag_base(params.flag_ptrs, params.dcp_rank) + row * kFlagsPerRow;
    const uint64_t previous_epoch =
        atomic_load_acquire_system(flags + kLeafEpoch);
    wait_for_epoch(flags + kReadEpoch, previous_epoch);
}

__global__ void reduce_scatter_kernel(ReduceScatterParams params) {
    const int64_t row = blockIdx.x;
    const int64_t batch = row / (params.q_len * params.num_heads);
    const int64_t query_head = row - batch * params.q_len * params.num_heads;
    const int query = query_head / params.num_heads;
    const int head = query_head - query * params.num_heads;
    const int heads_per_rank = params.num_heads / kWorldSize;
    const int owner = head / heads_per_rank;
    const int64_t lse_offset =
        (batch * params.num_heads + head) * params.q_len + query;
    int64_t* local_flags =
        flag_base(params.flag_ptrs, params.dcp_rank) + row * kFlagsPerRow;

    __shared__ uint64_t epoch;
    __shared__ float left_weight;
    __shared__ float right_weight;
    __shared__ float final_lse;

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
        const int group_base = (params.dcp_rank / period) * period;
        const int survivor = group_base + (owner & (period - 1));
        if (params.dcp_rank != survivor) {
            return;
        }
        const int peer = survivor ^ stride;
        int64_t* peer_flags =
            flag_base(params.flag_ptrs, peer) + row * kFlagsPerRow;
        const bool local_is_left = (survivor & stride) == 0;
        const bool write_compact = level == 2 && params.compact_out != nullptr;
        const int local_head = head - owner * heads_per_rank;

        if (threadIdx.x == 0) {
            wait_for_epoch(peer_flags + level, epoch);
            const auto* peer_lse =
                reinterpret_cast<const float*>(params.lse_ptrs[peer]);
            const float local = params.local_lse[lse_offset];
            const float remote = peer_lse[lse_offset];
            const float left = local_is_left ? local : remote;
            const float right = local_is_left ? remote : local;
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
            if (write_compact) {
                const int64_t compact_lse_offset =
                    (batch * heads_per_rank + local_head) * params.q_len +
                    query;
                params.compact_lse[compact_lse_offset] = final_lse;
            } else {
                params.local_lse[lse_offset] = final_lse;
            }
        }
        __syncthreads();

        const int64_t vector_base = row * (params.head_dim_v / 4);
        const auto* peer_output = reinterpret_cast<const float4*>(
            params.output_ptrs[peer]);
        auto* local_output = reinterpret_cast<float4*>(params.local_out);
        for (int vector = threadIdx.x;
             vector < params.head_dim_v / 4;
             vector += blockDim.x) {
            const float4 local = local_output[vector_base + vector];
            const float4 remote = peer_output[vector_base + vector];
            const float4 left = local_is_left ? local : remote;
            const float4 right = local_is_left ? remote : local;
            const float4 merged = make_float4(
                left_weight * left.x + right_weight * right.x,
                left_weight * left.y + right_weight * right.y,
                left_weight * left.z + right_weight * right.z,
                left_weight * left.w + right_weight * right.w);
            if (write_compact) {
                const int64_t compact_row =
                    (batch * params.q_len + query) * heads_per_rank +
                    local_head;
                auto* compact_output =
                    reinterpret_cast<float4*>(params.compact_out);
                compact_output[
                    compact_row * (params.head_dim_v / 4) + vector] = merged;
            } else {
                local_output[vector_base + vector] = merged;
            }
        }

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

    // The final owner has no parent; acknowledge its own state so the next
    // replay can overwrite this head shard after downstream stream consumers.
    if (threadIdx.x == 0) {
        atomic_inc_release_system(local_flags + kReadEpoch);
    }
}

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_ptrs(const at::Tensor& ptrs, const at::Tensor& local, const char* name) {
    check_cuda_contiguous(ptrs, name);
    TORCH_CHECK(
        ptrs.scalar_type() == at::kLong && ptrs.numel() == kWorldSize,
        name, " must be a contiguous int64 CUDA tensor with 8 elements");
    TORCH_CHECK(
        ptrs.device() == local.device(),
        name, " must be on the local state device");
}

} // namespace

void canonical_peer_reduce_scatter_prepare(
    const at::Tensor& flag_ptrs,
    int64_t rows,
    int64_t dcp_rank) {
    check_cuda_contiguous(flag_ptrs, "flag_ptrs");
    TORCH_CHECK(
        flag_ptrs.scalar_type() == at::kLong && flag_ptrs.numel() == kWorldSize,
        "flag_ptrs must be a contiguous int64 CUDA tensor with 8 elements");
    TORCH_CHECK(rows > 0, "rows must be positive");
    TORCH_CHECK(
        dcp_rank >= 0 && dcp_rank < kWorldSize,
        "dcp_rank must be in [0, 8)");
    const c10::cuda::CUDAGuard device_guard(flag_ptrs.device());
    const int64_t blocks = (rows + kPrepareThreads - 1) / kPrepareThreads;
    TORCH_CHECK(
        blocks <= std::numeric_limits<int>::max(),
        "canonical reduce-scatter prepare grid is too large");
    const PrepareParams params{
        flag_ptrs.data_ptr<int64_t>(), rows, static_cast<int>(dcp_rank)};
    prepare_kernel<<<
        static_cast<int>(blocks),
        kPrepareThreads,
        0,
        at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void canonical_peer_reduce_scatter(
    const at::Tensor& output_ptrs,
    const at::Tensor& lse_ptrs,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& local_out,
    const at::Tensor& local_lse) {
    check_cuda_contiguous(local_out, "local_out");
    check_cuda_contiguous(local_lse, "local_lse");
    check_ptrs(output_ptrs, local_out, "output_ptrs");
    check_ptrs(lse_ptrs, local_out, "lse_ptrs");
    check_ptrs(flag_ptrs, local_out, "flag_ptrs");
    TORCH_CHECK(
        local_out.scalar_type() == at::kFloat &&
            local_lse.scalar_type() == at::kFloat,
        "canonical reduce-scatter states must have dtype float32");
    TORCH_CHECK(
        local_out.device() == local_lse.device() &&
            local_out.dim() == 4 && local_lse.dim() == 3,
        "canonical reduce-scatter state shapes must be [B,Q,H,Dv] "
        "and [B,H,Q] on one device");
    const int64_t batch = local_out.size(0);
    const int64_t q_len = local_out.size(1);
    const int64_t num_heads = local_out.size(2);
    const int64_t head_dim_v = local_out.size(3);
    TORCH_CHECK(
        local_lse.sizes() == at::IntArrayRef({batch, num_heads, q_len}),
        "canonical reduce-scatter LSE shape must be [B,H,Q]");
    TORCH_CHECK(
        batch > 0 && q_len > 0 && num_heads > 0 &&
            num_heads % kWorldSize == 0 &&
            head_dim_v > 0 && head_dim_v % 4 == 0,
        "canonical reduce-scatter requires H % 8 == 0 and Dv % 4 == 0");
    TORCH_CHECK(
        dcp_rank >= 0 && dcp_rank < kWorldSize,
        "dcp_rank must be in [0, 8)");
    const int64_t rows = batch * q_len * num_heads;
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max(),
        "canonical reduce-scatter grid is too large");

    const c10::cuda::CUDAGuard device_guard(local_out.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major == 9 && properties->minor == 0,
        "canonical peer reduce-scatter is only supported on SM90");
    const ReduceScatterParams params{
        output_ptrs.data_ptr<int64_t>(),
        lse_ptrs.data_ptr<int64_t>(),
        flag_ptrs.data_ptr<int64_t>(),
        local_out.data_ptr<float>(),
        local_lse.data_ptr<float>(),
        nullptr,
        nullptr,
        rows,
        static_cast<int>(q_len),
        static_cast<int>(num_heads),
        static_cast<int>(head_dim_v),
        static_cast<int>(dcp_rank),
    };
    reduce_scatter_kernel<<<
        static_cast<int>(rows),
        kMergeThreads,
        0,
        at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void canonical_peer_reduce_scatter_compact(
    const at::Tensor& output_ptrs,
    const at::Tensor& lse_ptrs,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& local_out,
    const at::Tensor& local_lse,
    const at::Tensor& compact_out,
    const at::Tensor& compact_lse) {
    check_cuda_contiguous(local_out, "local_out");
    check_cuda_contiguous(local_lse, "local_lse");
    check_cuda_contiguous(compact_out, "compact_out");
    check_cuda_contiguous(compact_lse, "compact_lse");
    check_ptrs(output_ptrs, local_out, "output_ptrs");
    check_ptrs(lse_ptrs, local_out, "lse_ptrs");
    check_ptrs(flag_ptrs, local_out, "flag_ptrs");
    TORCH_CHECK(
        local_out.scalar_type() == at::kFloat &&
            local_lse.scalar_type() == at::kFloat &&
            compact_out.scalar_type() == at::kFloat &&
            compact_lse.scalar_type() == at::kFloat,
        "canonical reduce-scatter states must have dtype float32");
    TORCH_CHECK(
        local_out.device() == local_lse.device() &&
            local_out.device() == compact_out.device() &&
            local_out.device() == compact_lse.device() &&
            local_out.dim() == 4 && local_lse.dim() == 3 &&
            compact_out.dim() == 4 && compact_lse.dim() == 3,
        "canonical compact reduce-scatter tensors must be on one device");
    const int64_t batch = local_out.size(0);
    const int64_t q_len = local_out.size(1);
    const int64_t num_heads = local_out.size(2);
    const int64_t head_dim_v = local_out.size(3);
    TORCH_CHECK(
        batch > 0 && q_len > 0 && num_heads > 0 &&
            num_heads % kWorldSize == 0 &&
            head_dim_v > 0 && head_dim_v % 4 == 0,
        "canonical reduce-scatter requires H % 8 == 0 and Dv % 4 == 0");
    const int64_t heads_per_rank = num_heads / kWorldSize;
    TORCH_CHECK(
        local_lse.sizes() == at::IntArrayRef({batch, num_heads, q_len}) &&
            compact_out.sizes() ==
                at::IntArrayRef({batch, q_len, heads_per_rank, head_dim_v}) &&
            compact_lse.sizes() ==
                at::IntArrayRef({batch, heads_per_rank, q_len}),
        "canonical compact output shapes must be [B,Q,H/8,Dv] and [B,H/8,Q]");
    TORCH_CHECK(
        dcp_rank >= 0 && dcp_rank < kWorldSize,
        "dcp_rank must be in [0, 8)");
    const int64_t rows = batch * q_len * num_heads;
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max(),
        "canonical reduce-scatter grid is too large");

    const c10::cuda::CUDAGuard device_guard(local_out.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major == 9 && properties->minor == 0,
        "canonical peer reduce-scatter is only supported on SM90");
    const ReduceScatterParams params{
        output_ptrs.data_ptr<int64_t>(),
        lse_ptrs.data_ptr<int64_t>(),
        flag_ptrs.data_ptr<int64_t>(),
        local_out.data_ptr<float>(),
        local_lse.data_ptr<float>(),
        compact_out.data_ptr<float>(),
        compact_lse.data_ptr<float>(),
        rows,
        static_cast<int>(q_len),
        static_cast<int>(num_heads),
        static_cast<int>(head_dim_v),
        static_cast<int>(dcp_rank),
    };
    reduce_scatter_kernel<<<
        static_cast<int>(rows),
        kMergeThreads,
        0,
        at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
