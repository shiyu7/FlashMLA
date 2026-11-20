"""Distributed NCCL merge checks for DCP2/4/8.

Run one world size at a time, for example:

torchrun --standalone --nproc-per-node=8 -m pytest -q \
    tests/test_dcp_nccl_merge.py
"""

import os

import pytest
import torch
import torch.distributed as dist

import flash_mla
from dcp_eval.accuracy import fp64_merge_reference


@pytest.fixture(scope="session", autouse=True)
def _close_process_group_after_tests():
    yield
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _require_distributed_sm90():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (2, 4, 8):
        pytest.skip("launch with torchrun and 2, 4, or 8 processes")
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


def _make_local_state(rank: int, world_size: int, q_len: int, head_dim_v: int):
    batch = 2
    heads = max(16, world_size * 2)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1237 + rank * 1009 + q_len * 97 + head_dim_v)
    local_out = torch.randn(
        batch,
        q_len,
        heads,
        head_dim_v,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).to(torch.bfloat16)
    local_lse = torch.randn(
        batch,
        q_len,
        heads,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    local_lse[:, 0, rank::world_size] = -torch.inf
    local_lse[-1, -1, :] = -torch.inf
    return local_out, local_lse


def _gather_states(local_out, local_lse, group):
    world_size = dist.get_world_size(group)
    gathered_out = [torch.empty_like(local_out) for _ in range(world_size)]
    gathered_lse = [torch.empty_like(local_lse) for _ in range(world_size)]
    dist.all_gather(gathered_out, local_out, group=group)
    dist.all_gather(gathered_lse, local_lse, group=group)
    return torch.stack(gathered_out), torch.stack(gathered_lse)


@pytest.mark.parametrize(
    "q_len,head_dim_v",
    [(1, 128), (2, 512), (4, 128), (8, 512)],
)
@torch.inference_mode()
def test_nccl_merge_matches_fp64_oracle(q_len, head_dim_v):
    rank, world_size, group = _require_distributed_sm90()
    local_out, local_lse = _make_local_state(
        rank, world_size, q_len, head_dim_v
    )
    gathered_out, gathered_lse = _gather_states(local_out, local_lse, group)
    oracle_out, oracle_lse = fp64_merge_reference(gathered_out, gathered_lse)

    output, lse = flash_mla.nccl_merge_attention(local_out, local_lse, group)
    routed_out, routed_lse = flash_mla.merge_dcp_attention(
        local_out, local_lse, group
    )
    torch.cuda.synchronize()

    local_heads = local_out.shape[2] // world_size
    head_begin = rank * local_heads
    head_end = head_begin + local_heads
    oracle_out = oracle_out[:, :, head_begin:head_end]
    oracle_lse = oracle_lse[:, :, head_begin:head_end]
    torch.testing.assert_close(
        output.float(), oracle_out.float(), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        lse, oracle_lse.float(), rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(routed_out, output, rtol=0, atol=0)
    torch.testing.assert_close(routed_lse, lse, rtol=0, atol=0)

    assert torch.count_nonzero(output[-1, -1]) == 0
    assert torch.isneginf(lse[-1, -1]).all()
    assert not torch.isnan(output).any()
    assert not torch.isnan(lse).any()
