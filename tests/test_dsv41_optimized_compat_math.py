# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 optimized operators",
)


@pytest.mark.parametrize("rows,heads", [(1, 32), (17, 8)])
def test_candidate_score_small_shapes_use_general_kernel(rows, heads):
    from flaggems_sglang_plugin.prefill_compat import fp8_index_logits_prefill
    from flaggems_sglang.runtime.backend._nvidia.hopper.ops.dsv41_fp8_prefill_candidates import (
        supports_candidate_prefill,
    )

    length, block_size = 129, 8
    q = torch.randint(-2, 3, (rows, heads, 128), device="cuda").to(
        torch.float8_e4m3fn
    )
    k = torch.randint(-2, 3, (length, 128), device="cuda").to(
        torch.float8_e4m3fn
    )
    w = torch.randint(-2, 3, (rows, heads), device="cuda").bfloat16()
    lens = torch.arange(rows, device="cuda", dtype=torch.int32) * 7 + 12
    ids = torch.tensor([0, 16, -1], device="cuda", dtype=torch.int32)
    ids = ids.repeat(rows, 1)
    assert not supports_candidate_prefill(q, w, k, lens, ids)
    actual = fp8_index_logits_prefill(q, w, k, lens, candidate_blocks=ids)
    logical = ids.long().repeat_interleave(block_size, dim=1) * block_size
    logical += torch.arange(block_size, device="cuda").repeat(3)
    gathered = k.float()[logical.clamp(0, length - 1)]
    dot = torch.einsum("rhd,rnd->rhn", q.float(), gathered)
    expected = (
        (dot.bfloat16().float().clamp_min(0) * w.float()[..., None])
        .bfloat16()
        .float()
        .sum(1)
        .bfloat16()
        .float()
    )
    valid = (logical >= 0) & (logical < lens[:, None]) & (logical < length)
    expected.masked_fill_(~valid, float("-inf"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 17])
def test_mhc_bridge_preserves_matrix_and_output_shapes(rows):
    from flag_gems.fused.dsv41.mhc import hc_mix_stats, hc_split_sinkhorn
    from flaggems_sglang_plugin.compat.dsv41.mhc import hc_mix_stats_sinkhorn

    hc, width = 4, 20480
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, width, device="cuda", dtype=torch.float32) * 0.01
    scale = torch.ones(3, device="cuda", dtype=torch.float32)
    base = torch.zeros(24, device="cuda", dtype=torch.float32)
    actual = hc_mix_stats_sinkhorn(x, fn, scale, base, hc, 20, 1e-6, 1e-6)
    expected = hc_split_sinkhorn(
        hc_mix_stats(x, fn, 1e-6), scale, base, hc, 20, 1e-6
    )
    assert [t.shape for t in actual] == [
        (rows, hc),
        (rows, hc),
        (rows, hc, hc),
    ]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
