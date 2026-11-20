import math

import pytest
import torch

import flash_mla
from dcp_eval.canonical import canonical_tree_merge


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

    ragged_boundaries = flash_mla.make_cyclic_local_causal_seqlens(
        torch.tensor([9, 9, 9], dtype=torch.int32),
        q_len=4,
        cp_size=1,
        cp_rank=0,
        q_lens=torch.tensor([4, 2, 0], dtype=torch.int32),
    )
    assert ragged_boundaries.tolist() == [
        [6, 7, 8, 9],
        [8, 9, 9, 9],
        [9, 9, 9, 9],
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


@torch.inference_mode()
def test_dense_fp8_preallocated_output_matches_legacy_layout():
    _require_sm90()
    torch.manual_seed(319)
    batch = 2
    q_len = 4
    num_heads = 16
    global_seqlens = torch.tensor(
        [q_len, 96 + q_len],
        dtype=torch.int32,
        device="cuda",
    )
    q = (
        torch.randn(
            batch,
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
            batch,
            int(global_seqlens.max().item()),
            NUM_HEADS_K,
            HEAD_DIM,
            device="cuda",
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    local_cache, block_table, local_seqlens = _pack_cyclic_local_cache(
        global_kv,
        global_seqlens,
        cp_size=2,
        cp_rank=0,
    )
    local_causal = flash_mla.make_cyclic_local_causal_seqlens(
        global_seqlens,
        q_len,
        cp_size=2,
        cp_rank=0,
    )
    metadata, num_splits = flash_mla.get_mla_metadata_dense_fp8(
        local_seqlens,
        q_len * num_heads,
        NUM_HEADS_K,
    )
    descale_q = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    descale_k = torch.tensor([1.25], dtype=torch.float32, device="cuda")
    kwargs = dict(
        softmax_scale=1 / math.sqrt(HEAD_DIM),
        causal=True,
        causal_seqlens_k=local_causal,
        descale_q=descale_q,
        descale_k=descale_k,
    )

    legacy_out, legacy_lse = flash_mla.flash_mla_dense_fp8(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        **kwargs,
    )
    fp32_out, fp32_lse = flash_mla.flash_mla_dense_fp8_fp32(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        **kwargs,
    )
    assert fp32_out.dtype == torch.float32
    assert fp32_lse.dtype == torch.float32
    # The FP32 entry point changes only the final output storage type.  Casting
    # at the old ABI boundary must reproduce the regular BF16 result exactly.
    torch.testing.assert_close(fp32_out.to(torch.bfloat16), legacy_out, rtol=0, atol=0)
    torch.testing.assert_close(fp32_lse, legacy_lse, rtol=0, atol=0)
    assert torch.count_nonzero(fp32_out - legacy_out.float()) > 0
    out = torch.empty_like(legacy_out)
    native_lse = torch.empty(
        batch,
        q_len,
        num_heads,
        dtype=torch.float32,
        device="cuda",
    )
    returned_out, returned_lse = flash_mla.flash_mla_dense_fp8_out(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        out,
        native_lse,
        **kwargs,
    )

    assert returned_out.data_ptr() == out.data_ptr()
    assert returned_lse.data_ptr() == native_lse.data_ptr()
    torch.testing.assert_close(out, legacy_out, rtol=0, atol=0)
    torch.testing.assert_close(
        native_lse.transpose(1, 2),
        legacy_lse,
        rtol=0,
        atol=0,
    )

    q_lens = torch.tensor(
        [q_len, 1],
        dtype=torch.int32,
        device="cuda",
    )
    ragged_causal = flash_mla.make_cyclic_local_causal_seqlens(
        global_seqlens,
        q_len,
        cp_size=2,
        cp_rank=0,
        q_lens=q_lens,
    )
    ragged_kwargs = {
        **kwargs,
        "causal_seqlens_k": ragged_causal,
        "q_lens": q_lens,
    }
    unmasked_out, unmasked_lse = flash_mla.flash_mla_dense_fp8(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        **{**kwargs, "causal_seqlens_k": ragged_causal},
    )
    ragged_legacy_out, ragged_legacy_lse = flash_mla.flash_mla_dense_fp8(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        **ragged_kwargs,
    )
    ragged_out = torch.full_like(out, 1)
    ragged_native_lse = torch.zeros_like(native_lse)
    flash_mla.flash_mla_dense_fp8_out(
        q,
        local_cache,
        block_table,
        local_seqlens,
        HEAD_DIM_V,
        metadata,
        num_splits,
        ragged_out,
        ragged_native_lse,
        **ragged_kwargs,
    )

    torch.testing.assert_close(
        ragged_out,
        ragged_legacy_out,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ragged_native_lse.transpose(1, 2),
        ragged_legacy_lse,
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(ragged_out[1, 1:]) == 0
    assert torch.isneginf(ragged_native_lse[1, 1:]).all()
    torch.testing.assert_close(
        ragged_out[0],
        unmasked_out[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ragged_native_lse[0],
        unmasked_lse[0].transpose(0, 1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ragged_out[1, :1],
        unmasked_out[1, :1],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ragged_native_lse[1, :1],
        unmasked_lse[1, :, :1].transpose(0, 1),
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_dense_fp8_fused_canonical_matches_fp32_and_is_repeatable():
    _require_sm90()
    torch.manual_seed(1414)
    batch = 2
    q_len = 4
    num_heads = 16
    chunk_size = 128
    num_chunks = 4
    global_seqlens = torch.tensor(
        [129, 257], dtype=torch.int32, device="cuda"
    )
    q = (
        torch.randn(
            batch, q_len, num_heads, HEAD_DIM, device="cuda"
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    global_kv = (
        torch.randn(
            batch,
            int(global_seqlens.max().item()),
            NUM_HEADS_K,
            HEAD_DIM,
            device="cuda",
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    cache, block_table, seqlens = _pack_cyclic_local_cache(
        global_kv, global_seqlens, cp_size=1, cp_rank=0
    )
    causal_seqlens = flash_mla.make_cyclic_local_causal_seqlens(
        global_seqlens, q_len, cp_size=1, cp_rank=0
    )
    descale_q = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    descale_k = torch.tensor([1.25], dtype=torch.float32, device="cuda")
    kwargs = dict(
        softmax_scale=1 / math.sqrt(HEAD_DIM),
        causal=True,
        causal_seqlens_k=causal_seqlens,
        descale_q=descale_q,
        descale_k=descale_k,
    )

    canonical_metadata, canonical_splits = (
        flash_mla.get_mla_metadata_canonical_dense_fp8(
            seqlens, chunk_size, num_chunks
        )
    )
    torch.testing.assert_close(
        canonical_splits.cpu(),
        torch.tensor([0, 4, 8], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    metadata_host = canonical_metadata.cpu()
    assert metadata_host[0, :5].tolist() == [0, 0, 0, 128, 0]
    assert metadata_host[1, :5].tolist() == [0, 128, 0, 129, 1]
    assert metadata_host[2, :5].tolist() == [0, 192, 0, 192, 2]
    assert metadata_host[6, :5].tolist() == [1, 256, 1, 257, 2]
    assert metadata_host[7, :5].tolist() == [1, 320, 1, 320, 3]

    canonical_out, canonical_lse = flash_mla.flash_mla_dense_fp8_canonical(
        q,
        cache,
        block_table,
        seqlens,
        HEAD_DIM_V,
        canonical_metadata,
        canonical_splits,
        num_chunks,
        **kwargs,
    )
    repeated_out, repeated_lse = flash_mla.flash_mla_dense_fp8_canonical(
        q,
        cache,
        block_table,
        seqlens,
        HEAD_DIM_V,
        canonical_metadata,
        canonical_splits,
        num_chunks,
        **kwargs,
    )
    assert canonical_out.dtype == torch.float32
    torch.testing.assert_close(repeated_out, canonical_out, rtol=0, atol=0)
    torch.testing.assert_close(repeated_lse, canonical_lse, rtol=0, atol=0)

    regular_metadata, regular_splits = flash_mla.get_mla_metadata_dense_fp8(
        seqlens, q_len * num_heads, NUM_HEADS_K
    )
    regular_out, regular_lse = flash_mla.flash_mla_dense_fp8_fp32(
        q,
        cache,
        block_table,
        seqlens,
        HEAD_DIM_V,
        regular_metadata,
        regular_splits,
        **kwargs,
    )
    torch.testing.assert_close(
        canonical_out, regular_out, rtol=0.02, atol=0.01
    )
    torch.testing.assert_close(
        canonical_lse, regular_lse, rtol=0.01, atol=0.01
    )


@torch.inference_mode()
def test_dense_fp8_canonical_leaves_are_cp_size_invariant():
    _require_sm90()
    torch.manual_seed(1420)
    batch = 1
    q_len = 4
    num_heads = 16
    chunk_size = PAGE_SIZE
    num_global_chunks = 8
    global_seqlen = chunk_size * num_global_chunks
    q = (
        torch.randn(
            batch, q_len, num_heads, HEAD_DIM, device="cuda"
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    global_kv = (
        torch.randn(
            batch,
            global_seqlen,
            NUM_HEADS_K,
            HEAD_DIM,
            device="cuda",
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    descale_q = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    descale_k = torch.tensor([1.25], dtype=torch.float32, device="cuda")

    baseline_leaves = None
    baseline_lses = None
    baseline_tree = None
    baseline_tree_lse = None
    for cp_size in (1, 2, 4, 8):
        global_outputs = [None] * num_global_chunks
        global_lses = [None] * num_global_chunks
        for cp_rank in range(cp_size):
            chunk_ids = list(range(cp_rank, num_global_chunks, cp_size))
            local_tokens = torch.cat(
                [
                    global_kv[
                        :,
                        chunk_id * chunk_size : (chunk_id + 1) * chunk_size,
                    ]
                    for chunk_id in chunk_ids
                ],
                dim=1,
            )
            local_cache = local_tokens.reshape(
                len(chunk_ids), PAGE_SIZE, NUM_HEADS_K, HEAD_DIM
            )
            block_table = torch.arange(
                len(chunk_ids), dtype=torch.int32, device="cuda"
            ).view(1, -1)
            local_seqlens = torch.tensor(
                [len(chunk_ids) * chunk_size],
                dtype=torch.int32,
                device="cuda",
            )
            visible_global = [
                global_seqlen - q_len + query_idx + 1
                for query_idx in range(q_len)
            ]
            local_causal = torch.tensor(
                [[
                    sum(
                        max(
                            0,
                            min((chunk_id + 1) * chunk_size, visible)
                            - chunk_id * chunk_size,
                        )
                        for chunk_id in chunk_ids
                    )
                    for visible in visible_global
                ]],
                dtype=torch.int32,
                device="cuda",
            )
            metadata, splits = (
                flash_mla.get_mla_metadata_canonical_dense_fp8(
                    local_seqlens, chunk_size, len(chunk_ids)
                )
            )
            local_outputs, local_lses = (
                flash_mla.flash_mla_dense_fp8_canonical_leaves(
                    q,
                    local_cache,
                    block_table,
                    local_seqlens,
                    HEAD_DIM_V,
                    metadata,
                    splits,
                    len(chunk_ids),
                    softmax_scale=1 / math.sqrt(HEAD_DIM),
                    causal=True,
                    causal_seqlens_k=local_causal,
                    descale_q=descale_q,
                    descale_k=descale_k,
                )
            )
            assert local_outputs.shape == (
                1, len(chunk_ids), q_len, num_heads, HEAD_DIM_V
            )
            assert local_lses.shape == (
                1, len(chunk_ids), num_heads, q_len
            )
            for local_idx, chunk_id in enumerate(chunk_ids):
                global_outputs[chunk_id] = local_outputs[:, local_idx]
                global_lses[chunk_id] = local_lses[:, local_idx]

        leaves = torch.stack(global_outputs)
        lses = torch.stack(global_lses).transpose(-1, -2)
        tree_output, tree_lse = canonical_tree_merge(leaves, lses)
        if baseline_leaves is None:
            baseline_leaves = leaves
            baseline_lses = lses
            baseline_tree = tree_output
            baseline_tree_lse = tree_lse
        else:
            torch.testing.assert_close(leaves, baseline_leaves, rtol=0, atol=0)
            torch.testing.assert_close(lses, baseline_lses, rtol=0, atol=0)
            torch.testing.assert_close(tree_output, baseline_tree, rtol=0, atol=0)
            torch.testing.assert_close(tree_lse, baseline_tree_lse, rtol=0, atol=0)

    # The production P3 primitive must reproduce the P1 in-kernel tree exactly,
    # including operand order and FP32 rounding at every edge.
    pair_outputs = [leaf.clone() for leaf in baseline_leaves.unbind(0)]
    pair_lses = [
        leaf.transpose(-1, -2).contiguous()
        for leaf in baseline_lses.unbind(0)
    ]
    while len(pair_outputs) > 1:
        next_outputs = []
        next_lses = []
        for index in range(0, len(pair_outputs), 2):
            left_out = pair_outputs[index]
            left_lse = pair_lses[index]
            flash_mla.canonical_merge_pair_out(
                left_out,
                left_lse,
                pair_outputs[index + 1],
                pair_lses[index + 1],
                left_out,
                left_lse,
            )
            next_outputs.append(left_out)
            next_lses.append(left_lse)
        pair_outputs = next_outputs
        pair_lses = next_lses

    direct_cache = global_kv[:, :chunk_size].reshape(
        1, PAGE_SIZE, NUM_HEADS_K, HEAD_DIM
    )
    direct_block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    direct_seqlens = torch.tensor(
        [chunk_size], dtype=torch.int32, device="cuda"
    )
    direct_causal = torch.full(
        (1, q_len), chunk_size, dtype=torch.int32, device="cuda"
    )
    direct_metadata, direct_splits = (
        flash_mla.get_mla_metadata_canonical_dense_fp8(
            direct_seqlens, chunk_size, 1
        )
    )
    allocated_out, allocated_lse = (
        flash_mla.flash_mla_dense_fp8_canonical_leaves(
            q,
            direct_cache,
            direct_block_table,
            direct_seqlens,
            HEAD_DIM_V,
            direct_metadata,
            direct_splits,
            1,
            softmax_scale=1 / math.sqrt(HEAD_DIM),
            causal=True,
            causal_seqlens_k=direct_causal,
            descale_q=descale_q,
            descale_k=descale_k,
        )
    )
    direct_storage = torch.empty_like(allocated_out)
    returned_out, returned_lse = (
        flash_mla.flash_mla_dense_fp8_canonical_leaves_out(
            q,
            direct_cache,
            direct_block_table,
            direct_seqlens,
            HEAD_DIM_V,
            direct_metadata,
            direct_splits,
            1,
            direct_storage,
            softmax_scale=1 / math.sqrt(HEAD_DIM),
            causal=True,
            causal_seqlens_k=direct_causal,
            descale_q=descale_q,
            descale_k=descale_k,
        )
    )
    assert returned_out.data_ptr() == direct_storage.data_ptr()
    torch.testing.assert_close(returned_out, allocated_out, rtol=0, atol=0)
    torch.testing.assert_close(returned_lse, allocated_lse, rtol=0, atol=0)

    cp1_cache = global_kv.reshape(
        num_global_chunks, PAGE_SIZE, NUM_HEADS_K, HEAD_DIM
    )
    cp1_block_table = torch.arange(
        num_global_chunks, dtype=torch.int32, device="cuda"
    ).view(1, -1)
    cp1_seqlens = torch.tensor(
        [global_seqlen], dtype=torch.int32, device="cuda"
    )
    cp1_causal = torch.tensor(
        [[global_seqlen - q_len + query_idx + 1 for query_idx in range(q_len)]],
        dtype=torch.int32,
        device="cuda",
    )
    cp1_metadata, cp1_splits = (
        flash_mla.get_mla_metadata_canonical_dense_fp8(
            cp1_seqlens, chunk_size, num_global_chunks
        )
    )
    fused_out, fused_lse = flash_mla.flash_mla_dense_fp8_canonical(
        q,
        cp1_cache,
        cp1_block_table,
        cp1_seqlens,
        HEAD_DIM_V,
        cp1_metadata,
        cp1_splits,
        num_global_chunks,
        softmax_scale=1 / math.sqrt(HEAD_DIM),
        causal=True,
        causal_seqlens_k=cp1_causal,
        descale_q=descale_q,
        descale_k=descale_k,
    )
    torch.testing.assert_close(pair_outputs[0], fused_out, rtol=0, atol=0)
    torch.testing.assert_close(pair_lses[0], fused_lse, rtol=0, atol=0)
