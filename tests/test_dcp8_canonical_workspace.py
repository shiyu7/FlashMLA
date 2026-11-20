"""Run with:

torchrun --standalone --nproc-per-node=8 -m pytest -q \
    tests/test_dcp8_canonical_workspace.py
"""

import math
import os

import pytest
import torch
import torch.distributed as dist

import flash_mla


PAGE_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512
HEADS = 128
CHUNK_SIZE = 1216
WORLD_SIZE = 8


@pytest.fixture(scope="session", autouse=True)
def _close_process_group_after_tests():
    yield
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _require_dcp8_sm90():
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        pytest.skip("launch this test with torchrun and eight processes")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("canonical DCP8 workspace requires SM90")
    if not dist.is_initialized():
        device = torch.device("cuda", local_rank)
        try:
            dist.init_process_group("nccl", device_id=device)
        except TypeError:
            dist.init_process_group("nccl")
    return dist.get_rank(), dist.group.WORLD


@torch.inference_mode()
def test_canonical_workspace_matches_cp1_across_8k_decode_window():
    rank, group = _require_dcp8_sm90()
    batch_size, q_len = 1, 8
    capacity = CHUNK_SIZE * WORLD_SIZE

    q_generator = torch.Generator(device="cuda")
    q_generator.manual_seed(1430)
    q = (
        torch.randn(
            batch_size,
            q_len,
            HEADS,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
            generator=q_generator,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    kv_generator = torch.Generator(device="cuda")
    kv_generator.manual_seed(1431)
    global_kv = (
        torch.randn(
            batch_size,
            capacity,
            1,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
            generator=kv_generator,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    local_cache = global_kv[:, rank::WORLD_SIZE].reshape(
        -1, PAGE_SIZE, 1, HEAD_DIM
    ).contiguous()
    pages_per_chunk = CHUNK_SIZE // PAGE_SIZE
    local_block_table = torch.arange(
        batch_size * pages_per_chunk,
        dtype=torch.int32,
        device="cuda",
    ).view(batch_size, pages_per_chunk)
    cp1_block_table = torch.arange(
        batch_size * pages_per_chunk * WORLD_SIZE,
        dtype=torch.int32,
        device="cuda",
    ).view(batch_size, pages_per_chunk * WORLD_SIZE)
    descale_q = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    descale_k = torch.tensor([1.25], dtype=torch.float32, device="cuda")

    workspace = flash_mla.create_canonical_dcp8_workspace(
        max_batch_size=batch_size,
        max_seqlen_q=q_len,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        head_dim_v=HEAD_DIM_V,
        chunk_size=CHUNK_SIZE,
        group=group,
    )
    try:
        for global_length in (8192, 9692):
            global_seqlens = torch.full(
                (batch_size,),
                global_length,
                dtype=torch.int32,
                device="cuda",
            )
            chunk_seqlens = workspace.chunk_cache_seqlens(global_seqlens)
            metadata, num_splits = (
                flash_mla.get_mla_metadata_canonical_dense_fp8(
                    chunk_seqlens,
                    CHUNK_SIZE,
                    1,
                )
            )
            output, lse = workspace.attention(
                q,
                local_cache,
                local_block_table,
                global_seqlens,
                metadata,
                num_splits,
                softmax_scale=1 / math.sqrt(HEAD_DIM),
                descale_q=descale_q,
                descale_k=descale_k,
            )

            cp1_metadata, cp1_splits = (
                flash_mla.get_mla_metadata_canonical_dense_fp8(
                    global_seqlens,
                    CHUNK_SIZE,
                    WORLD_SIZE,
                )
            )
            visible_global = torch.arange(
                global_length - q_len + 1,
                global_length + 1,
                dtype=torch.int32,
                device="cuda",
            ).view(1, q_len)
            reference_output, reference_lse = (
                flash_mla.flash_mla_dense_fp8_canonical(
                    q,
                    global_kv.view(-1, PAGE_SIZE, 1, HEAD_DIM),
                    cp1_block_table,
                    global_seqlens,
                    HEAD_DIM_V,
                    cp1_metadata,
                    cp1_splits,
                    WORLD_SIZE,
                    softmax_scale=1 / math.sqrt(HEAD_DIM),
                    causal=True,
                    causal_seqlens_k=visible_global,
                    descale_q=descale_q,
                    descale_k=descale_k,
                )
            )
            local_heads = HEADS // WORLD_SIZE
            head_begin = rank * local_heads
            head_end = head_begin + local_heads
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output,
                reference_output[:, :, head_begin:head_end],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                lse,
                reference_lse[:, head_begin:head_end],
                rtol=0,
                atol=0,
            )
    finally:
        workspace.close()
