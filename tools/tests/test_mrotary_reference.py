# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU controls; real NVIDIA agreement is checked separately."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    spec = importlib.util.spec_from_file_location(
        "_mrotary_test", ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stride", [20, 512])
def test_torch_reference_matches_scalar_axis_oracle(stride):
    ref = load(
        "src/flaggems_sglang/reference/mrotary_embedding_torch.py"
    ).reference
    gen = torch.Generator().manual_seed(17)
    q = torch.randn(20, 1024, generator=gen).bfloat16()
    k = torch.randn(20, 256, generator=gen).bfloat16()
    before = [x.clone() for x in (q, k)]
    positions = torch.empty_strided((3, 20), (stride, 1), dtype=torch.int64)
    positions.copy_(
        torch.arange(20)[None, :] + torch.tensor([0, 7, 14])[:, None]
    )
    cache = torch.randn(64, 64, generator=gen).bfloat16()
    pointers = [x.data_ptr() for x in (q, k)]
    result = ref(
        q, k, cache, positions, [11, 11, 10], 256, 64, True, False, True, None
    )
    assert result is None
    axes = ([0, 1, 2] * 11)[:32]
    for value, original, pointer in zip((q, k), before, pointers):
        expected = original.view(20, -1, 256).clone()
        source = original.view(20, -1, 256).float()
        for j, axis in enumerate(axes):
            c = cache[positions[axis], j].float()[:, None]
            s = cache[positions[axis], j + 32].float()[:, None]
            expected[..., j] = source[..., j] * c - source[..., j + 32] * s
            expected[..., j + 32] = (
                source[..., j + 32] * c + source[..., j] * s
            )
        assert value.data_ptr() == pointer
        torch.testing.assert_close(
            value.view_as(expected), expected, atol=0, rtol=0
        )
        assert torch.equal(
            value.view_as(expected)[..., 64:],
            original.view_as(expected)[..., 64:],
        )


def test_baseline_identity_includes_layout():
    cases = load("src/flaggems_sglang/reference/mrotary_embedding_cases.py")
    keys = [
        cases.baseline_key(cases.build_inputs(c, torch.bfloat16, "meta"))
        for c in cases.WORKLOADS
    ]
    assert len(set(keys)) == 6
    attr = load("benchmark/attri_util.py")
    data = {
        "mrotary_embedding": {
            "shapes": {keys[2]: {"torch.bfloat16": {"latency_ms": 1}}}
        }
    }
    record, reason = attr.lookup_base_record_ex(
        data, "mrotary_embedding", torch.bfloat16, dict(baseline_key=keys[2])
    )
    assert record["latency_ms"] == 1 and reason is None
    record, reason = attr.lookup_base_record_ex(
        data, "mrotary_embedding", torch.bfloat16, dict(baseline_key=keys[4])
    )
    assert record is None and reason
