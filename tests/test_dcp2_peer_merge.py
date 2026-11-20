"""Run with:

torchrun --standalone --nproc-per-node=2 -m pytest -q \
    tests/test_dcp2_peer_merge.py
"""

import os

import pytest
import torch
import torch.distributed as dist

import flash_mla


PAGE_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512
NUM_HEADS_K = 1


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
        device = torch.device("cuda", local_rank)
        try:
            dist.init_process_group("nccl", device_id=device)
        except TypeError:
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


def _make_dense_fp8_dcp_inputs(rank: int):
    batch, q_len, heads = 3, 4, 16
    global_seqlens = torch.tensor(
        [72, 65, 64],
        dtype=torch.int32,
        device="cuda",
    )
    local_seqlens = flash_mla.cyclic_local_length(
        global_seqlens,
        cp_size=2,
        cp_rank=rank,
    ).to(torch.int32)

    q_generator = torch.Generator(device="cuda")
    q_generator.manual_seed(2027)
    q = (
        torch.randn(
            batch,
            q_len,
            heads,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
            generator=q_generator,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )

    # Every local sequence fits in one page for this DCP2 test. Give each
    # request a distinct page and use rank-specific cache contents.
    cache_generator = torch.Generator(device="cuda")
    cache_generator.manual_seed(3011 + rank)
    local_cache = (
        torch.randn(
            batch,
            PAGE_SIZE,
            NUM_HEADS_K,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
            generator=cache_generator,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    block_table = torch.arange(
        batch,
        dtype=torch.int32,
        device="cuda",
    ).view(batch, 1)
    metadata, num_splits = flash_mla.get_mla_metadata_dense_fp8(
        local_seqlens,
        q_len * heads,
        NUM_HEADS_K,
    )
    descale_q = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    descale_k = torch.tensor([1.25], dtype=torch.float32, device="cuda")
    return (
        q,
        local_cache,
        block_table,
        local_seqlens,
        global_seqlens,
        metadata,
        num_splits,
        descale_q,
        descale_k,
    )


def _run_local_dense_fp8(
    q: torch.Tensor,
    local_cache: torch.Tensor,
    block_table: torch.Tensor,
    local_seqlens: torch.Tensor,
    metadata: torch.Tensor,
    num_splits: torch.Tensor,
    causal_seqlens_k: torch.Tensor,
    q_lens: torch.Tensor,
    descale_q: torch.Tensor,
    descale_k: torch.Tensor,
):
    local_out, local_lse_bhq = flash_mla.flash_mla_dense_fp8(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        causal=True,
        causal_seqlens_k=causal_seqlens_k,
        q_lens=q_lens,
        descale_q=descale_q,
        descale_k=descale_k,
    )
    return local_out, local_lse_bhq.transpose(1, 2).contiguous()


@pytest.mark.parametrize(
    "q_len,head_dim_v",
    [(1, 512), (4, 512), (8, 512), (2, 510)],
)
@torch.inference_mode()
def test_dcp2_peer_merge_matches_nccl(q_len, head_dim_v):
    rank, group = _require_dcp2_sm90()
    batch, heads = 3, 16
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
        _fill_local_state(
            local_out,
            local_lse,
            rank,
            iteration=q_len + head_dim_v,
        )
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
def test_dcp2_flash_mla_ragged_q_lens_survives_graph_replay():
    rank, group = _require_dcp2_sm90()
    (
        q,
        local_cache,
        block_table,
        local_seqlens,
        global_seqlens,
        metadata,
        num_splits,
        descale_q,
        descale_k,
    ) = _make_dense_fp8_dcp_inputs(rank)
    batch, q_len, heads, _ = q.shape
    q_lens = torch.tensor(
        [q_len, 1, 0],
        dtype=torch.int32,
        device="cuda",
    )
    causal_seqlens_k = flash_mla.make_cyclic_local_causal_seqlens(
        global_seqlens,
        q_len,
        cp_size=2,
        cp_rank=rank,
        q_lens=q_lens,
    )
    workspace = flash_mla.create_dcp2_attention_workspace(
        batch,
        q_len,
        heads,
        HEAD_DIM_V,
        group,
    )
    graph_out = torch.empty(
        batch,
        q_len,
        heads // 2,
        HEAD_DIM_V,
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

    def run_workspace() -> None:
        workspace.run_flash_mla_dense_fp8(
            q,
            local_cache,
            block_table,
            local_seqlens,
            HEAD_DIM_V,
            metadata,
            num_splits,
            causal=True,
            causal_seqlens_k=causal_seqlens_k,
            q_lens=q_lens,
            descale_q=descale_q,
            descale_k=descale_k,
            out=graph_out,
            softmax_lse=graph_lse,
        )

    try:
        # Warm up FlashMLA, symmetric-memory publication, and peer merge before
        # capturing their fixed physical B x Q x H shape.
        run_workspace()
        torch.cuda.synchronize()
        dist.barrier(group=group)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_workspace()

        profiles = (
            (q_len, 1, 0),
            (2, q_len, 1),
            (0, 3, q_len),
        )
        graph_replays = int(
            os.environ.get("FLASH_MLA_DCP2_RAGGED_GRAPH_REPLAYS", len(profiles))
        )
        check_interval = int(
            os.environ.get("FLASH_MLA_DCP2_GRAPH_CHECK_INTERVAL", "1")
        )
        if graph_replays <= 0 or check_interval <= 0:
            raise ValueError("graph replay count and check interval must be positive")
        for replay_idx in range(graph_replays):
            profile = profiles[replay_idx % len(profiles)]
            q_lens.copy_(
                torch.tensor(
                    profile,
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            causal_seqlens_k.copy_(
                flash_mla.make_cyclic_local_causal_seqlens(
                    global_seqlens,
                    q_len,
                    cp_size=2,
                    cp_rank=rank,
                    q_lens=q_lens,
                )
            )
            graph.replay()

            should_check = (
                replay_idx < len(profiles)
                or (replay_idx + 1) % check_interval == 0
                or replay_idx + 1 == graph_replays
            )
            if not should_check:
                continue

            local_ref_out, local_ref_lse = _run_local_dense_fp8(
                q,
                local_cache,
                block_table,
                local_seqlens,
                metadata,
                num_splits,
                causal_seqlens_k,
                q_lens,
                descale_q,
                descale_k,
            )
            nccl_out, nccl_lse = flash_mla.nccl_merge_attention(
                local_ref_out,
                local_ref_lse,
                group,
            )
            torch.cuda.synchronize()

            torch.testing.assert_close(
                graph_out.float(),
                nccl_out.float(),
                rtol=0.02,
                atol=0.02,
            )
            torch.testing.assert_close(
                graph_lse,
                nccl_lse,
                rtol=2e-5,
                atol=2e-5,
            )
            for batch_idx, logical_q_len in enumerate(profile):
                assert (
                    torch.count_nonzero(
                        graph_out[batch_idx, logical_q_len:]
                    )
                    == 0
                )
                assert torch.isneginf(
                    graph_lse[batch_idx, logical_q_len:]
                ).all()
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

        graph_replays = int(
            os.environ.get("FLASH_MLA_DCP2_EPOCH_GRAPH_REPLAYS", "4")
        )
        check_interval = int(
            os.environ.get("FLASH_MLA_DCP2_GRAPH_CHECK_INTERVAL", "1")
        )
        if graph_replays <= 0 or check_interval <= 0:
            raise ValueError("graph replay count and check interval must be positive")
        for iteration in range(1, graph_replays + 1):
            _fill_local_state(source_out, source_lse, rank, iteration)
            graph.replay()
            should_check = (
                iteration <= 4
                or iteration % check_interval == 0
                or iteration == graph_replays
            )
            if not should_check:
                continue
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
