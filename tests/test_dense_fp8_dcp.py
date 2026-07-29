import math

import pytest
import torch

import flash_mla


PAGE_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512
NUM_HEADS_K = 1


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("dense FP8 DCP requires SM90")


def _pack_cyclic_local_cache(
    global_kv: torch.Tensor,
    global_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
):
    batch = global_kv.shape[0]
    pages_per_batch = []
    local_tokens_per_batch = []

    for batch_idx in range(batch):
        seqlen = int(global_seqlens[batch_idx].item())
        local_tokens = global_kv[
            batch_idx, cp_rank:seqlen:cp_size
        ].contiguous()
        local_tokens_per_batch.append(local_tokens)
        pages_per_batch.append(
            (local_tokens.shape[0] + PAGE_SIZE - 1) // PAGE_SIZE
        )

    max_pages = max(max(pages_per_batch), 1)
    block_table = torch.full(
        (batch, max_pages),
        -1,
        dtype=torch.int32,
        device=global_kv.device,
    )
    pages = []
    for batch_idx, local_tokens in enumerate(local_tokens_per_batch):
        for page_idx in range(pages_per_batch[batch_idx]):
            page = torch.zeros(
                PAGE_SIZE,
                NUM_HEADS_K,
                HEAD_DIM,
                dtype=global_kv.dtype,
                device=global_kv.device,
            )
            begin = page_idx * PAGE_SIZE
            end = min(begin + PAGE_SIZE, local_tokens.shape[0])
            page[: end - begin] = local_tokens[begin:end]
            block_table[batch_idx, page_idx] = len(pages)
            pages.append(page)

    if not pages:
        pages.append(
            torch.zeros(
                PAGE_SIZE,
                NUM_HEADS_K,
                HEAD_DIM,
                dtype=global_kv.dtype,
                device=global_kv.device,
            )
        )
    local_cache = torch.stack(pages)
    local_seqlens = flash_mla.cyclic_local_length(
        global_seqlens,
        cp_size,
        cp_rank,
    ).to(torch.int32)
    return local_cache, block_table, local_seqlens


def _reference(
    q: torch.Tensor,
    global_kv: torch.Tensor,
    global_seqlens: torch.Tensor,
    softmax_scale: float,
    descale_q: float,
    descale_k: float,
):
    batch, q_len, num_heads, _ = q.shape
    output = torch.empty(
        batch,
        q_len,
        num_heads,
        HEAD_DIM_V,
        dtype=torch.float32,
        device=q.device,
    )
    lse = torch.empty(
        batch,
        num_heads,
        q_len,
        dtype=torch.float32,
        device=q.device,
    )
    for batch_idx in range(batch):
        seqlen = int(global_seqlens[batch_idx].item())
        query = q[batch_idx].float().transpose(0, 1) * descale_q
        kv = global_kv[batch_idx, :seqlen, 0].float() * descale_k
        scores = torch.matmul(query, kv.transpose(0, 1)) * softmax_scale
        query_indices = torch.arange(q_len, device=q.device)
        visible = seqlen - q_len + query_indices + 1
        key_indices = torch.arange(seqlen, device=q.device)
        scores.masked_fill_(
            key_indices[None, None, :] >= visible[None, :, None],
            -torch.inf,
        )
        lse[batch_idx] = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output[batch_idx] = torch.matmul(
            probs,
            kv[:, :HEAD_DIM_V],
        ).transpose(0, 1)
    return output.to(torch.bfloat16), lse


def _run_one_rank(
    q,
    global_kv,
    global_seqlens,
    q_len,
    num_heads,
    cp_size,
    cp_rank,
    softmax_scale,
    descale_q,
    descale_k,
):
    local_cache, block_table, local_seqlens = _pack_cyclic_local_cache(
        global_kv,
        global_seqlens,
        cp_size,
        cp_rank,
    )
    local_causal = flash_mla.make_cyclic_local_causal_seqlens(
        global_seqlens,
        q_len,
        cp_size,
        cp_rank,
    )
    metadata, num_splits = flash_mla.get_mla_metadata_dense_fp8(
        local_seqlens,
        q_len * num_heads // NUM_HEADS_K,
        NUM_HEADS_K,
    )
    return flash_mla.flash_mla_dense_fp8(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        softmax_scale=softmax_scale,
        causal=True,
        causal_seqlens_k=local_causal,
        descale_q=descale_q,
        descale_k=descale_k,
    )


def test_cyclic_length_and_mtp_boundaries():
    global_seqlens = torch.tensor([1, 2, 7, 8, 9], dtype=torch.int32)
    expected = [
        [1, 1, 2, 2, 3],
        [0, 1, 2, 2, 2],
        [0, 0, 2, 2, 2],
        [0, 0, 1, 2, 2],
    ]
    for rank in range(4):
        actual = flash_mla.cyclic_local_length(
            global_seqlens,
            4,
            rank,
        )
        assert actual.tolist() == expected[rank]

    boundaries = [
        flash_mla.make_cyclic_local_causal_seqlens(
            torch.tensor([9], dtype=torch.int32),
            q_len=4,
            cp_size=4,
            cp_rank=rank,
        ).tolist()[0]
        for rank in range(4)
    ]
    assert boundaries == [
        [2, 2, 2, 3],
        [2, 2, 2, 2],
        [1, 2, 2, 2],
        [1, 1, 2, 2],
    ]

    empty_output, empty_lse = flash_mla.merge_local_attention(
        torch.zeros(4, 1, 2, 3, 5, dtype=torch.bfloat16),
        torch.full((4, 1, 3, 2), -torch.inf),
    )
    assert torch.count_nonzero(empty_output) == 0
    assert torch.isneginf(empty_lse).all()


@pytest.mark.parametrize("q_len", [1, 2, 4, 8])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@torch.inference_mode()
def test_dense_fp8_cyclic_dcp_matches_global_reference(q_len, cp_size):
    _require_sm90()
    torch.manual_seed(17 + q_len * 10 + cp_size)
    num_heads = 16
    global_seqlens = torch.tensor(
        [q_len, 64 + q_len, 129 + q_len],
        dtype=torch.int32,
        device="cuda",
    )
    max_seqlen = int(global_seqlens.max().item())
    q = (
        torch.randn(
            len(global_seqlens),
            q_len,
            num_heads,
            HEAD_DIM,
            device="cuda",
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    global_kv = (
        torch.randn(
            len(global_seqlens),
            max_seqlen,
            NUM_HEADS_K,
            HEAD_DIM,
            device="cuda",
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    softmax_scale = 1 / math.sqrt(HEAD_DIM)
    descale_q_value = 0.75
    descale_k_value = 1.25
    descale_q = torch.tensor(
        [descale_q_value],
        dtype=torch.float32,
        device="cuda",
    )
    descale_k = torch.tensor(
        [descale_k_value],
        dtype=torch.float32,
        device="cuda",
    )

    local_outputs = []
    local_lses = []
    for cp_rank in range(cp_size):
        output, lse = _run_one_rank(
            q,
            global_kv,
            global_seqlens,
            q_len,
            num_heads,
            cp_size,
            cp_rank,
            softmax_scale,
            descale_q,
            descale_k,
        )
        local_outputs.append(output)
        local_lses.append(lse)

    if q_len == 4 and cp_size == 4:
        # The last rank owns one token for the shortest request, but that
        # token is invisible to its first three MTP queries.
        assert torch.count_nonzero(local_outputs[3][0, :3]) == 0
        assert torch.isneginf(local_lses[3][0, :, :3]).all()

    output, lse = flash_mla.merge_local_attention(
        local_outputs,
        local_lses,
    )
    output_ref, lse_ref = _reference(
        q,
        global_kv,
        global_seqlens,
        softmax_scale,
        descale_q_value,
        descale_k_value,
    )
    torch.testing.assert_close(
        output.float(),
        output_ref.float(),
        rtol=0.08,
        atol=0.02,
    )
    torch.testing.assert_close(lse, lse_ref, rtol=0.01, atol=0.02)
