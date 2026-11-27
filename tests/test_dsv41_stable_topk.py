# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stable equal-score selection for the SGLang v0.5.18 backport."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _canonical(indices):
    return (
        indices.masked_fill(indices < 0, torch.iinfo(torch.int32).max)
        .sort(-1)
        .values
    )


@pytest.mark.parametrize(
    "count", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 1024, 2048]
)
def test_ragged_ties_and_changed_input_graph(count):
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    values = torch.cat(
        (
            torch.full((505,), 2.0),
            torch.full((count,), 0.5),
            torch.full((32,), -4.0),
        )
    )
    n, k = values.numel(), 512
    scores_cpu = torch.full((2, ((n + 7) // 4) * 4), -float("inf"))
    starts = torch.tensor([0, 3], dtype=torch.int32)
    lengths = torch.full((2,), n, dtype=torch.int32)
    offsets = torch.tensor([100, 300], dtype=torch.int32)
    for row in range(2):
        scores_cpu[row, starts[row] : starts[row] + n] = values
    scores, lens, starts_gpu, offsets_gpu = [
        x.cuda() for x in (scores_cpu, lengths, starts, offsets)
    ]
    out = torch.empty((2, k), dtype=torch.int32, device="cuda")

    def expected(data, positions, token_lengths, out_offsets):
        result = torch.full((2, k), -1, dtype=torch.int32)
        for row in range(2):
            pos, length = int(positions[row]), int(token_lengths[row])
            order = torch.argsort(
                data[row, pos : pos + length], descending=True, stable=True
            )[:k]
            result[row, : len(order)] = order.int() + out_offsets[row]
        return result

    for _ in range(10):
        topk_transform_ragged_v2(
            scores,
            lens,
            row_starts=starts_gpu,
            out_offsets=offsets_gpu,
            out_indices=out,
        )
        assert torch.equal(
            _canonical(out.cpu()),
            _canonical(expected(scores_cpu, starts, lengths, offsets)),
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_transform_ragged_v2(
            scores,
            lens,
            row_starts=starts_gpu,
            out_offsets=offsets_gpu,
            out_indices=out,
        )
    changed = torch.full_like(scores_cpu, -float("inf"))
    changed_starts = torch.tensor([1, 2], dtype=torch.int32)
    changed_lengths = torch.tensor([n - 3, n], dtype=torch.int32)
    changed_offsets = offsets + 5
    for row in range(2):
        length, start = int(changed_lengths[row]), int(changed_starts[row])
        changed[row, start : start + length] = values.flip(0)[:length]
    scores.copy_(changed)
    lens.copy_(changed_lengths)
    starts_gpu.copy_(changed_starts)
    offsets_gpu.copy_(changed_offsets)
    out.fill_(-9)
    graph.replay()
    assert torch.equal(
        _canonical(out.cpu()),
        _canonical(
            expected(changed, changed_starts, changed_lengths, changed_offsets)
        ),
    )


@pytest.mark.parametrize("n", [1024, 8192, 16384, 32768])
def test_paged_ties_and_changed_input_graph(n):
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
    )

    values = torch.arange(n, dtype=torch.float32).remainder(2048) / 2048
    scores = values[None].cuda()
    lens = torch.tensor([n], dtype=torch.int32, device="cuda")
    pages = (
        torch.arange((n + 63) // 64, dtype=torch.int32)[None] * 3 + 7
    ).cuda()
    out = torch.empty((1, 511), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(out)
    plan = plan_topk_v2(lens, static_threshold=1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_transform_paged_v2(
            scores, lens, pages, out, 64, plan, out_raw_indices=raw
        )
    changed = values.flip(0)
    length = n - 7
    scores.copy_(changed[None])
    lens.fill_(length)
    pages.add_(11)
    plan.copy_(plan_topk_v2(lens, static_threshold=1))
    out.fill_(-9)
    raw.fill_(-9)
    graph.replay()
    expected_raw = torch.argsort(
        changed[:length], descending=True, stable=True
    )[:511].int()
    actual_raw, actual_out = raw.cpu()[0], out.cpu()[0]
    assert torch.equal(actual_raw.sort().values, expected_raw.sort().values)
    assert torch.equal(
        actual_out,
        pages.cpu()[0, actual_raw.long() // 64] * 64 + actual_raw % 64,
    )
