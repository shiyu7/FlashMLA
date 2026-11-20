import math

import pytest
import torch

from dcp_eval.accuracy import global_attention_reference
from dcp_eval.canonical import (
    canonical_chunk_attention,
    canonical_tree_merge,
    merge_attention_states,
)


def test_canonical_merge_neutral_state():
    output = torch.randn(2, 3, 4, 8)
    lse = torch.randn(2, 3, 4)
    neutral_output = torch.zeros_like(output)
    neutral_lse = torch.full_like(lse, -torch.inf)
    merged_output, merged_lse = merge_attention_states(
        output, lse, neutral_output, neutral_lse
    )
    torch.testing.assert_close(merged_output, output, rtol=0, atol=0)
    torch.testing.assert_close(merged_lse, lse, rtol=0, atol=0)


def test_canonical_tree_has_fixed_adjacent_order():
    outputs = torch.tensor([1.0, 3.0, 5.0, 7.0]).view(4, 1, 1, 1, 1)
    lses = torch.zeros(4, 1, 1, 1)
    output, lse = canonical_tree_merge(outputs, lses)
    torch.testing.assert_close(output, torch.tensor([[[[4.0]]]]))
    torch.testing.assert_close(lse, torch.tensor([[[math.log(4.0)]]]))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_canonical_chunks_are_cp_size_invariant(cp_size):
    generator = torch.Generator().manual_seed(2026)
    q = torch.randn(2, 4, 4, 16, generator=generator).mul_(0.1)
    kv = torch.randn(2, 31, 1, 16, generator=generator).mul_(0.1)
    seqlens = torch.tensor([17, 31], dtype=torch.int32)
    q_lens = torch.tensor([4, 2], dtype=torch.int32)
    kwargs = {
        "chunk_size": 7,
        "softmax_scale": 1 / math.sqrt(q.shape[-1]),
        "descale_q": 0.75,
        "descale_k": 1.25,
        "head_dim_v": 8,
    }
    baseline_output, baseline_lse = canonical_chunk_attention(
        q, kv, seqlens, q_lens, cp_size=1, **kwargs
    )
    output, lse = canonical_chunk_attention(
        q, kv, seqlens, q_lens, cp_size=cp_size, **kwargs
    )
    torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
    torch.testing.assert_close(lse, baseline_lse, rtol=0, atol=0)

    reference_output, reference_lse = global_attention_reference(
        q,
        kv,
        seqlens,
        q_lens,
        softmax_scale=kwargs["softmax_scale"],
        descale_q=kwargs["descale_q"],
        descale_k=kwargs["descale_k"],
        head_dim_v=kwargs["head_dim_v"],
    )
    torch.testing.assert_close(output, reference_output, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(lse, reference_lse, rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(output[1, 2:]) == 0
    assert torch.isneginf(lse[1, 2:]).all()
