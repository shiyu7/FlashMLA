"""Run with:

torchrun --standalone --nproc-per-node=2 -m pytest -q \
    tests/test_dcp2_peer_merge.py
"""

import os

import pytest
import torch
import torch.distributed as dist

import flash_mla


def test_peer_policy_is_strictly_dcp2_and_q_le_8():
    assert flash_mla.is_dcp2_peer_attention_eligible(1, 8, 16, 512, 2)
    assert not flash_mla.is_dcp2_peer_attention_eligible(1, 9, 16, 512, 2)
    assert not flash_mla.is_dcp2_peer_attention_eligible(1, 8, 16, 512, 4)
    assert not flash_mla.is_dcp2_peer_attention_eligible(1, 8, 16, 512, 8)
    payload = flash_mla.dcp2_attention_payload_bytes(1, 8, 16, 512)
    assert flash_mla.is_dcp2_peer_attention_eligible(
        1, 8, 16, 512, 2, max_peer_payload_bytes=payload
    )
    assert not flash_mla.is_dcp2_peer_attention_eligible(
        1, 8, 16, 512, 2, max_peer_payload_bytes=payload - 1
    )


def _require_dcp2_sm90():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch this test with torchrun and two processes")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("DCP2 peer merge requires SM90")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return local_rank, dist.group.WORLD


def _fill_local_state(
    out: torch.Tensor,
    lse: torch.Tensor,
    rank: int,
    iteration: int,
) -> None:
    generator = torch.Generator(device=out.device)
    generator.manual_seed(1009 + rank * 97 + iteration * 10007)
    out.copy_(
        torch.randn(
            out.shape,
            dtype=torch.float32,
            device=out.device,
            generator=generator,
        ).to(torch.bfloat16)
    )
    lse.copy_(
        torch.randn(
            lse.shape,
            dtype=torch.float32,
            device=lse.device,
            generator=generator,
        )
    )
    # Exercise neutral empty-shard rows on alternating ranks.
    lse[:, 0, rank::2] = -torch.inf


@pytest.mark.parametrize("q_len", [1, 4, 8])
@torch.inference_mode()
def test_dcp2_peer_merge_matches_nccl(q_len):
    rank, group = _require_dcp2_sm90()
    batch, heads, head_dim_v = 3, 16, 512
    workspace = flash_mla.create_dcp2_attention_workspace(
        batch,
        q_len,
        heads,
        head_dim_v,
        group,
    )
    try:
        local_out, local_lse = workspace.prepare(
            batch, q_len, heads, head_dim_v
        )
        _fill_local_state(local_out, local_lse, rank, iteration=q_len)
        source_out = local_out.clone()
        source_lse = local_lse.clone()
        peer_out, peer_lse = workspace.merge(
            batch, q_len, heads, head_dim_v
        )
        nccl_out, nccl_lse = flash_mla.nccl_merge_attention(
            source_out, source_lse, group
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            peer_out.float(),
            nccl_out.float(),
            rtol=0.02,
            atol=0.02,
        )
        torch.testing.assert_close(
            peer_lse,
            nccl_lse,
            rtol=2e-5,
            atol=2e-5,
        )
    finally:
        workspace.close()
        dist.barrier(group=group)


@torch.inference_mode()
def test_dcp2_epoch_protocol_survives_cuda_graph_replay():
    rank, group = _require_dcp2_sm90()
    batch, q_len, heads, head_dim_v = 2, 4, 16, 512
    workspace = flash_mla.create_dcp2_attention_workspace(
        batch,
        q_len,
        heads,
        head_dim_v,
        group,
    )
    graph_out = torch.empty(
        batch,
        q_len,
        heads // 2,
        head_dim_v,
        dtype=torch.bfloat16,
        device="cuda",
    )
    graph_lse = torch.empty(
        batch,
        q_len,
        heads // 2,
        dtype=torch.float32,
        device="cuda",
    )
    source_out = torch.empty(
        batch,
        q_len,
        heads,
        head_dim_v,
        dtype=torch.bfloat16,
        device="cuda",
    )
    source_lse = torch.empty(
        batch,
        q_len,
        heads,
        dtype=torch.float32,
        device="cuda",
    )
    try:
        # Warm up all kernels and establish epoch 1 before capture.
        local_out, local_lse = workspace.prepare(
            batch, q_len, heads, head_dim_v
        )
        _fill_local_state(source_out, source_lse, rank, iteration=0)
        local_out.copy_(source_out)
        local_lse.copy_(source_lse)
        workspace.merge(
            batch,
            q_len,
            heads,
            head_dim_v,
            out=graph_out,
            softmax_lse=graph_lse,
        )
        torch.cuda.synchronize()
        dist.barrier(group=group)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out, captured_lse = workspace.prepare(
                batch, q_len, heads, head_dim_v
            )
            captured_out.copy_(source_out)
            captured_lse.copy_(source_lse)
            workspace.merge(
                batch,
                q_len,
                heads,
                head_dim_v,
                out=graph_out,
                softmax_lse=graph_lse,
            )

        for iteration in range(1, 5):
            _fill_local_state(source_out, source_lse, rank, iteration)
            graph.replay()
            torch.cuda.synchronize()
            reference_out, reference_lse = flash_mla.nccl_merge_attention(
                source_out, source_lse, group
            )
            torch.testing.assert_close(
                graph_out.float(),
                reference_out.float(),
                rtol=0.02,
                atol=0.02,
            )
            torch.testing.assert_close(
                graph_lse,
                reference_lse,
                rtol=2e-5,
                atol=2e-5,
            )
    finally:
        workspace.close()
        dist.barrier(group=group)
