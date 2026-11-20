"""End-to-end distributed checks for SM90 dense FP8 DCP.

Run one DCP group per eight-GPU node, for example:

torchrun --standalone --nproc-per-node=8 -m pytest -q \
    tests/test_dense_fp8_dcp_distributed.py
"""

import math
import os

import pytest
import torch
import torch.distributed as dist

import flash_mla
from dcp_eval.accuracy import (
    HEAD_DIM_V,
    global_attention_reference,
    make_fp8_inputs,
    pack_cyclic_local_cache,
)


@pytest.fixture(scope="session", autouse=True)
def _close_process_group_after_tests():
    yield
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _require_distributed_sm90():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8:
        pytest.skip("launch with torchrun and eight processes")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("dense FP8 DCP requires SM90")
    if not dist.is_initialized():
        device = torch.device("cuda", local_rank)
        try:
            dist.init_process_group("nccl", device_id=device)
        except TypeError:
            dist.init_process_group("nccl")
    return dist.get_rank(), world_size, dist.group.WORLD


@pytest.mark.parametrize("q_len", [1, 2, 4, 8])
@torch.inference_mode()
def test_distributed_dense_fp8_dcp8_matches_global_reference(q_len):
    rank, world_size, group = _require_distributed_sm90()
    num_heads = 48
    descale_q = 0.75
    descale_k = 1.25
    softmax_scale = 1 / math.sqrt(576)
    q, global_kv, global_seqlens = make_fp8_inputs(
        global_seqlens=[q_len, 64 + q_len, 129 + q_len],
        q_len=q_len,
        num_heads=num_heads,
        seed=2026,
    )
    q_lens = torch.tensor(
        [q_len, max(1, q_len // 2), 0],
        dtype=torch.int32,
        device="cuda",
    )
    local_cache, block_table, local_seqlens = pack_cyclic_local_cache(
        global_kv,
        global_seqlens,
        world_size,
        rank,
        permute_pages=True,
        seed=2026,
    )
    local_causal_seqlens = flash_mla.make_cyclic_local_causal_seqlens(
        global_seqlens,
        q_len,
        world_size,
        rank,
        q_lens=q_lens,
    )
    metadata, num_splits = flash_mla.get_mla_metadata_dense_fp8(
        local_seqlens,
        q_len * num_heads,
        1,
    )
    local_out, local_lse_bhq = flash_mla.flash_mla_dense_fp8(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        softmax_scale=softmax_scale,
        causal=True,
        causal_seqlens_k=local_causal_seqlens,
        q_lens=q_lens,
        descale_q=torch.tensor([descale_q], dtype=torch.float32, device="cuda"),
        descale_k=torch.tensor([descale_k], dtype=torch.float32, device="cuda"),
    )
    local_out_fp32, local_lse_fp32_bhq = flash_mla.flash_mla_dense_fp8_fp32(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        softmax_scale=softmax_scale,
        causal=True,
        causal_seqlens_k=local_causal_seqlens,
        q_lens=q_lens,
        descale_q=torch.tensor([descale_q], dtype=torch.float32, device="cuda"),
        descale_k=torch.tensor([descale_k], dtype=torch.float32, device="cuda"),
    )
    assert local_out_fp32.dtype == torch.float32
    torch.testing.assert_close(
        local_out_fp32.to(torch.bfloat16), local_out, rtol=0, atol=0
    )
    torch.testing.assert_close(local_lse_fp32_bhq, local_lse_bhq, rtol=0, atol=0)
    output, lse = flash_mla.nccl_merge_attention(
        local_out,
        local_lse_bhq.transpose(1, 2).contiguous(),
        group,
    )
    output_fp32, lse_fp32 = flash_mla.nccl_merge_attention(
        local_out_fp32,
        local_lse_fp32_bhq.transpose(1, 2).contiguous(),
        group,
    )
    output_ref, lse_ref = global_attention_reference(
        q,
        global_kv,
        global_seqlens,
        q_lens,
        softmax_scale=softmax_scale,
        descale_q=descale_q,
        descale_k=descale_k,
    )

    local_heads = num_heads // world_size
    head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
    torch.testing.assert_close(
        output.float(),
        output_ref[:, :, head_slice].float(),
        rtol=0.08,
        atol=0.02,
    )
    torch.testing.assert_close(
        output_fp32,
        output_ref[:, :, head_slice].float(),
        rtol=0.08,
        atol=0.02,
    )
    torch.testing.assert_close(
        lse,
        lse_ref[:, :, head_slice],
        rtol=0.01,
        atol=0.02,
    )
    torch.testing.assert_close(lse_fp32, lse, rtol=0, atol=0)
    assert torch.count_nonzero(output[2]) == 0
    assert torch.count_nonzero(output_fp32[2]) == 0
    assert torch.isneginf(lse[2]).all()
    assert not torch.isnan(output).any()
    assert not torch.isnan(output_fp32).any()
    assert not torch.isnan(lse).any()
