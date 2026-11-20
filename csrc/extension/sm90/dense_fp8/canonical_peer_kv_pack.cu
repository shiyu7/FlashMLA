#include "canonical_peer_kv_pack.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kWorldSize = 8;
constexpr int kPageSize = 64;
constexpr int kCopyThreads = 256;
constexpr int kWriteEpoch = 0;
constexpr int kReadCount = 1;
constexpr uint32_t kMaxFenceSpins = 100'000'000;

struct PrepareParams {
    const int64_t* flag_ptrs;
    int dcp_rank;
};

struct StageParams {
    const uint4* local_cache;
    const int32_t* block_table;
    const int32_t* local_seqlens;
    uint4* local_staging;
    int64_t total_vectors;
    int block_table_stride;
    int max_local_tokens;
    int vectors_per_token;
};

struct PublishParams {
    const int64_t* flag_ptrs;
    int dcp_rank;
};

struct AssembleParams {
    const int64_t* staging_ptrs;
    uint4* canonical_cache;
    int64_t total_vectors;
    int max_local_tokens;
    int chunk_size;
    int vectors_per_token;
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

__device__ __forceinline__ void wait_for_value(
    const int64_t* ptr,
    uint64_t value) {
    for (uint32_t spin = 0; spin < kMaxFenceSpins; ++spin) {
        if (atomic_load_acquire_system(ptr) >= value) {
            return;
        }
    }
    asm volatile("trap;");
}

__global__ void prepare_kernel(PrepareParams params) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t* flags = flag_base(params.flag_ptrs, params.dcp_rank);
    const uint64_t previous_epoch =
        atomic_load_acquire_system(flags + kWriteEpoch);
    wait_for_value(flags + kReadCount, previous_epoch * kWorldSize);
}

__global__ void stage_kernel(StageParams params) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= params.total_vectors) {
        return;
    }
    const int vector = index % params.vectors_per_token;
    const int64_t token_row = index / params.vectors_per_token;
    const int local_token = token_row % params.max_local_tokens;
    const int batch = token_row / params.max_local_tokens;
    if (local_token >= params.local_seqlens[batch]) {
        return;
    }
    const int page = params.block_table[
        batch * params.block_table_stride + local_token / kPageSize];
    const int page_offset = local_token % kPageSize;
    const int64_t source =
        (static_cast<int64_t>(page) * kPageSize + page_offset) *
            params.vectors_per_token +
        vector;
    params.local_staging[index] = params.local_cache[source];
}

__global__ void publish_kernel(PublishParams params) {
    if (threadIdx.x != 0) {
        return;
    }
    __threadfence_system();
    int64_t* flags = flag_base(params.flag_ptrs, params.dcp_rank);
    atomic_inc_release_system(flags + kWriteEpoch);
}

__global__ void wait_for_peers_kernel(PublishParams params) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t* local_flags = flag_base(params.flag_ptrs, params.dcp_rank);
    const uint64_t epoch =
        atomic_load_acquire_system(local_flags + kWriteEpoch);
#pragma unroll
    for (int peer = 0; peer < kWorldSize; ++peer) {
        wait_for_value(flag_base(params.flag_ptrs, peer) + kWriteEpoch, epoch);
    }
}

__global__ void assemble_kernel(AssembleParams params) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= params.total_vectors) {
        return;
    }
    const int vector = index % params.vectors_per_token;
    const int64_t token_row = index / params.vectors_per_token;
    const int chunk_offset = token_row % params.chunk_size;
    const int batch = token_row / params.chunk_size;
    const int global_token = params.dcp_rank * params.chunk_size + chunk_offset;
    const int source_rank = global_token % kWorldSize;
    const int source_local_token = global_token / kWorldSize;
    const auto* source = reinterpret_cast<const uint4*>(
        params.staging_ptrs[source_rank]);
    const int64_t source_index =
        (static_cast<int64_t>(batch) * params.max_local_tokens +
         source_local_token) *
            params.vectors_per_token +
        vector;
    params.canonical_cache[index] = source[source_index];
}

__global__ void acknowledge_kernel(PublishParams params) {
    if (threadIdx.x != 0) {
        return;
    }
    __threadfence_system();
#pragma unroll
    for (int source = 0; source < kWorldSize; ++source) {
        int64_t* flags = flag_base(params.flag_ptrs, source);
        atomic_inc_release_system(flags + kReadCount);
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
    TORCH_CHECK(ptrs.device() == local.device(), name, " must be on local device");
}

int checked_blocks(int64_t work, int threads, const char* name) {
    const int64_t blocks = (work + threads - 1) / threads;
    TORCH_CHECK(
        blocks > 0 && blocks <= std::numeric_limits<int>::max(),
        name, " grid is invalid or too large");
    return static_cast<int>(blocks);
}

} // namespace

void canonical_peer_pack_token_cyclic_kv(
    const at::Tensor& staging_ptrs,
    const at::Tensor& flag_ptrs,
    int64_t dcp_rank,
    const at::Tensor& local_cache,
    const at::Tensor& block_table,
    const at::Tensor& local_seqlens,
    const at::Tensor& local_staging,
    const at::Tensor& canonical_cache) {
    check_cuda_contiguous(local_cache, "local_cache");
    check_cuda_contiguous(block_table, "block_table");
    check_cuda_contiguous(local_seqlens, "local_seqlens");
    check_cuda_contiguous(local_staging, "local_staging");
    check_cuda_contiguous(canonical_cache, "canonical_cache");
    check_ptrs(staging_ptrs, local_staging, "staging_ptrs");
    check_ptrs(flag_ptrs, local_staging, "flag_ptrs");
    TORCH_CHECK(
        local_cache.scalar_type() == torch::kFloat8_e4m3fn &&
            local_staging.scalar_type() == torch::kFloat8_e4m3fn &&
            canonical_cache.scalar_type() == torch::kFloat8_e4m3fn,
        "canonical peer KV pack requires FP8 E4M3 tensors");
    TORCH_CHECK(
        block_table.scalar_type() == at::kInt &&
            local_seqlens.scalar_type() == at::kInt,
        "block_table and local_seqlens must have dtype int32");
    TORCH_CHECK(
        local_cache.device() == local_staging.device() &&
            local_cache.device() == canonical_cache.device() &&
            local_cache.device() == block_table.device() &&
            local_cache.device() == local_seqlens.device(),
        "canonical peer KV pack tensors must be on one device");
    TORCH_CHECK(
        local_cache.dim() == 4 && local_cache.size(1) == kPageSize &&
            local_cache.size(2) == 1,
        "local_cache must have shape [pages,64,1,D]");
    TORCH_CHECK(
        local_staging.dim() == 3 && canonical_cache.dim() == 4 &&
            canonical_cache.size(2) == 1,
        "local_staging/canonical_cache shapes must be [B,L,D] and [B,C,1,D]");
    const int64_t batch = local_staging.size(0);
    const int64_t max_local_tokens = local_staging.size(1);
    const int64_t head_dim = local_staging.size(2);
    const int64_t chunk_size = canonical_cache.size(1);
    TORCH_CHECK(
        batch > 0 && max_local_tokens >= chunk_size && chunk_size > 0 &&
            head_dim > 0 && head_dim % 16 == 0,
        "canonical peer KV pack requires L >= C > 0 and D % 16 == 0");
    TORCH_CHECK(
        canonical_cache.size(0) == batch &&
            canonical_cache.size(3) == head_dim &&
            block_table.dim() == 2 && block_table.size(0) == batch &&
            local_seqlens.dim() == 1 && local_seqlens.size(0) == batch,
        "canonical peer KV pack batch/head dimensions are inconsistent");
    TORCH_CHECK(
        block_table.size(1) * kPageSize >= max_local_tokens,
        "block_table does not cover local staging length");
    TORCH_CHECK(
        dcp_rank >= 0 && dcp_rank < kWorldSize,
        "dcp_rank must be in [0, 8)");

    const c10::cuda::CUDAGuard device_guard(local_cache.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major == 9 && properties->minor == 0,
        "canonical peer KV pack is only supported on SM90");
    const int vectors_per_token = static_cast<int>(head_dim / 16);
    const int64_t stage_vectors = batch * max_local_tokens * vectors_per_token;
    const int64_t assemble_vectors = batch * chunk_size * vectors_per_token;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const PrepareParams prepare{
        flag_ptrs.data_ptr<int64_t>(), static_cast<int>(dcp_rank)};
    prepare_kernel<<<1, 32, 0, stream>>>(prepare);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const StageParams stage{
        reinterpret_cast<const uint4*>(local_cache.data_ptr()),
        block_table.data_ptr<int32_t>(),
        local_seqlens.data_ptr<int32_t>(),
        reinterpret_cast<uint4*>(local_staging.data_ptr()),
        stage_vectors,
        static_cast<int>(block_table.size(1)),
        static_cast<int>(max_local_tokens),
        vectors_per_token,
    };
    stage_kernel<<<
        checked_blocks(stage_vectors, kCopyThreads, "stage"),
        kCopyThreads,
        0,
        stream>>>(stage);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const PublishParams publish{
        flag_ptrs.data_ptr<int64_t>(), static_cast<int>(dcp_rank)};
    publish_kernel<<<1, 32, 0, stream>>>(publish);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    wait_for_peers_kernel<<<1, 32, 0, stream>>>(publish);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const AssembleParams assemble{
        staging_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<uint4*>(canonical_cache.data_ptr()),
        assemble_vectors,
        static_cast<int>(max_local_tokens),
        static_cast<int>(chunk_size),
        vectors_per_token,
        static_cast<int>(dcp_rank),
    };
    assemble_kernel<<<
        checked_blocks(assemble_vectors, kCopyThreads, "assemble"),
        kCopyThreads,
        0,
        stream>>>(assemble);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    acknowledge_kernel<<<1, 32, 0, stream>>>(publish);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
