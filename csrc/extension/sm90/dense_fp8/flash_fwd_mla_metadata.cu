#include "flash_fwd_mla_kernel.h"

static constexpr int MaxBatchSize = 4096;

__global__ void __launch_bounds__(256, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const Mla_metadata_params params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.batch_size;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    __shared__ int num_blocks_shared[MaxBatchSize];
    __shared__ int num_splits_shared[MaxBatchSize];

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int num_blocks = cutlass::ceil_div(seqlens_k_ptr[i], block_size_n);
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
    }
    for (int offset = 16; offset >= 1; offset /= 2) {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;

        int now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
            tile_scheduler_metadata0[0] = now_idx;
            tile_scheduler_metadata0[1] = now_block * block_size_n;
            tile_scheduler_metadata1 = now_n_split_idx;
            int remain_payload = payload;
            while (now_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_idx;
                    now_block = 0;
                    now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx;
                        remain_payload = 0;
                    }
                    break;
                }
            }
            tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
            tile_scheduler_metadata0[3] = now_block > 0 ? now_block * block_size_n : seqlens_k_ptr[now_idx - 1];
            *reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize) = *reinterpret_cast<int4 *>(tile_scheduler_metadata0);
            tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
        }
        FLASH_DEVICE_ASSERT(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32) {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.batch_size < MaxBatchSize);
    get_mla_metadata_kernel<<<1, 32, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

// Build one scheduler record per canonical chunk. The grid shape is fixed by
// num_chunks, while the current sequence length is read on device so the
// metadata update remains CUDA-Graph friendly and introduces no host sync.
__global__ void __launch_bounds__(256, 1, 1)
get_mla_canonical_metadata_kernel(
    __grid_constant__ const Mla_canonical_metadata_params params) {
    const int partition = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_partitions = params.batch_size * params.num_chunks;
    if (partition < total_partitions) {
        const int batch_idx = partition / params.num_chunks;
        const int chunk_idx = partition - batch_idx * params.num_chunks;
        const int seqlen = max(__ldg(params.seqlens_k_ptr + batch_idx), 0);
        const int chunk_begin = chunk_idx * params.chunk_size;
        const int chunk_end = min(chunk_begin + params.chunk_size, seqlen);

        // For an inactive tail leaf use the first block boundary after the
        // sequence. This makes n_block_min == n_block_max even when seqlen is
        // not page aligned, and the attention kernel emits a neutral state.
        const int empty_boundary =
            cutlass::ceil_div(seqlen, params.block_size_n) * params.block_size_n;
        const bool active = chunk_begin < chunk_end;
        int *metadata =
            params.tile_scheduler_metadata_ptr
            + partition * TileSchedulerMetaDataSize;
        metadata[0] = batch_idx;
        metadata[1] = active ? chunk_begin : empty_boundary;
        metadata[2] = batch_idx;
        metadata[3] = active ? chunk_end : empty_boundary;
        metadata[4] = chunk_idx;
        metadata[5] = 0;
        metadata[6] = 0;
        metadata[7] = 0;
    }

    if (blockIdx.x == 0) {
        for (int batch_idx = threadIdx.x; batch_idx <= params.batch_size;
             batch_idx += blockDim.x) {
            params.num_splits_ptr[batch_idx] =
                batch_idx * params.num_chunks;
        }
    }
}

void get_mla_canonical_metadata_func(
    Mla_canonical_metadata_params &params,
    cudaStream_t stream) {
    const int total_partitions = params.batch_size * params.num_chunks;
    get_mla_canonical_metadata_kernel<<<
        cutlass::ceil_div(total_partitions, 128), 128, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
