import math

import pytest
import torch

from dcp_eval.accuracy import (
    compute_accuracy_metrics,
    fp64_merge_reference,
    make_fp8_inputs,
    run_cyclic_case,
)


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("dense FP8 DCP requires SM90")


def test_fp64_merge_reference_weighting_and_empty_row():
    local_outputs = torch.tensor(
        [
            [[[[1.0, 1.0]], [[7.0, 9.0]]]],
            [[[[3.0, 3.0]], [[11.0, 13.0]]]],
        ],
        dtype=torch.float32,
    )
    local_lses = torch.tensor(
        [
            [[[0.0], [-torch.inf]]],
            [[[math.log(3.0)], [-torch.inf]]],
        ],
        dtype=torch.float32,
    )
    output, lse = fp64_merge_reference(local_outputs, local_lses)
    torch.testing.assert_close(
        output[0, 0, 0], torch.tensor([2.5, 2.5], dtype=torch.float64)
    )
    torch.testing.assert_close(
        lse[0, 0, 0], torch.tensor(math.log(4.0), dtype=torch.float64)
    )
    assert torch.count_nonzero(output[0, 1]) == 0
    assert torch.isneginf(lse[0, 1]).all()


def test_accuracy_metrics_are_zero_for_exact_neutral_result():
    output = torch.zeros(2, 4, 3, 8)
    lse = torch.full((2, 4, 3), -torch.inf)
    q_lens = torch.tensor([0, 0], dtype=torch.int32)
    metrics = compute_accuracy_metrics(output, lse, output, lse, q_lens)
    assert metrics.output_abs_max == 0
    assert metrics.output_nrmse == 0
    assert metrics.lse_inf_mask_mismatches == 0
    assert metrics.output_nan_count == 0
    assert metrics.lse_nan_count == 0


@pytest.mark.parametrize(
    "q_len,cp_size,distribution,descale_q,descale_k,seed",
    [
        (1, 1, "normal", 0.75, 1.25, 101),
        (2, 2, "uniform", 1.0, 0.5, 211),
        (4, 4, "near_zero", 2.0, 1.0, 307),
        (8, 8, "signed", 0.5, 0.5, 401),
    ],
)
@torch.inference_mode()
def test_cyclic_dcp_accuracy_matrix(
    q_len, cp_size, distribution, descale_q, descale_k, seed
):
    _require_sm90()
    global_lengths = [q_len, 64 + q_len, 129 + q_len]
    q, global_kv, global_seqlens = make_fp8_inputs(
        global_seqlens=global_lengths,
        q_len=q_len,
        num_heads=16,
        distribution=distribution,
        seed=seed,
    )
    q_lens = torch.tensor(
        [q_len, max(1, q_len // 2), 0],
        dtype=torch.int32,
        device="cuda",
    )
    output, lse, output_ref, lse_ref = run_cyclic_case(
        q,
        global_kv,
        global_seqlens,
        q_lens,
        cp_size=cp_size,
        softmax_scale=1 / math.sqrt(q.shape[-1]),
        descale_q=descale_q,
        descale_k=descale_k,
        permute_pages=True,
        page_seed=seed,
    )
    torch.testing.assert_close(
        output.float(), output_ref, rtol=0.08, atol=0.02
    )
    torch.testing.assert_close(lse, lse_ref, rtol=0.01, atol=0.02)

    for batch_idx, logical_q in enumerate(q_lens.tolist()):
        assert torch.count_nonzero(output[batch_idx, logical_q:]) == 0
        assert torch.isneginf(lse[batch_idx, logical_q:]).all()
    metrics = compute_accuracy_metrics(
        output, lse, output_ref, lse_ref, q_lens
    )
    assert metrics.output_nan_count == 0
    assert metrics.lse_nan_count == 0
    assert metrics.lse_inf_mask_mismatches == 0


@torch.inference_mode()
def test_physical_page_permutation_is_semantically_invariant():
    _require_sm90()
    q_len = 4
    q, global_kv, global_seqlens = make_fp8_inputs(
        global_seqlens=[68, 133],
        q_len=q_len,
        num_heads=16,
        distribution="normal",
        seed=809,
    )
    q_lens = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
    common = dict(
        cp_size=4,
        softmax_scale=1 / math.sqrt(q.shape[-1]),
        descale_q=0.75,
        descale_k=1.25,
    )
    contiguous_out, contiguous_lse, _, _ = run_cyclic_case(
        q,
        global_kv,
        global_seqlens,
        q_lens,
        permute_pages=False,
        **common,
    )
    permuted_out, permuted_lse, _, _ = run_cyclic_case(
        q,
        global_kv,
        global_seqlens,
        q_lens,
        permute_pages=True,
        page_seed=809,
        **common,
    )
    torch.testing.assert_close(permuted_out, contiguous_out, rtol=0, atol=0)
    torch.testing.assert_close(permuted_lse, contiguous_lse, rtol=0, atol=0)
