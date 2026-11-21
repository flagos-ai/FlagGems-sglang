# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
from __future__ import annotations

import math

import pytest
import torch

from flaggems_sglang import qsa_mqa_paged as qwen4_qsa_mqa_paged_dot

from . import conftest as cfg

DEVICE = cfg.device
EPS = 1e-6
pytestmark = pytest.mark.qsa_mqa_paged


def _require_cuda() -> None:
    if not torch.get_device_module(cfg.device).is_available():
        pytest.skip("Qwen4 Triton correctness requires an accelerator")


def _qsa_mqa_reference(
    q, cache, table, token_to_req, positions, lengths, ratio
):
    columns = table.shape[1] * cache.shape[1]
    req = token_to_req.to(torch.int64)
    valid_req = (req >= 0) & (req < table.shape[0])
    safe_req = req.clamp(0, table.shape[0] - 1)
    seq = lengths.index_select(0, safe_req)
    seq = torch.where(valid_req, seq, torch.zeros_like(seq))
    visible = torch.minimum(
        (positions.to(torch.int64) + 1) // ratio, seq // ratio
    )
    col = torch.arange(columns, device=q.device, dtype=torch.int64)
    logical_page = col // cache.shape[1]
    offset = col % cache.shape[1]
    safe_logical = logical_page.clamp(0, table.shape[1] - 1)
    physical = table[safe_req[:, None], safe_logical[None, :]].to(torch.int64)
    valid = (
        valid_req[:, None]
        & (col[None, :] < visible[:, None])
        & (physical >= 0)
        & (physical < cache.shape[0])
    )
    safe_physical = physical.clamp(0, cache.shape[0] - 1)
    keys = cache[safe_physical, offset[None, :], 0].float()
    dots = torch.matmul(q.float(), keys.transpose(1, 2))
    logits = torch.relu(dots).sum(1) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~valid, -float("inf"))
    return logits, visible.to(torch.int32)


@pytest.mark.qwen4_qsa
def test_qwen4_qsa_mqa_paged_dot_matches_torch_and_invalid_pages():
    _require_cuda()
    rows, page_size, pages, columns = 8, 16, 4, 48
    torch.manual_seed(202)
    q = torch.randn((rows, 4, 128), device=DEVICE, dtype=torch.bfloat16)
    cache = torch.randn(
        (pages, page_size, 1, 128), device=DEVICE, dtype=torch.bfloat16
    )
    table = torch.tensor(
        [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
        device=DEVICE,
        dtype=torch.int32,
    )
    table[2, 1] = -1
    token_to_req = torch.tensor(
        [0, 1, 2, 3, 0, 1, -1, 3], device=DEVICE, dtype=torch.int32
    )
    positions = torch.tensor(
        [47, 39, 31, 23, 47, 35, 47, 15], device=DEVICE, dtype=torch.int64
    )
    lengths = torch.tensor([48, 40, 32, 24], device=DEVICE, dtype=torch.int64)

    ref_logits, ref_visible = _qsa_mqa_reference(
        q, cache, table, token_to_req, positions, lengths, 4
    )
    out_logits, out_visible = qwen4_qsa_mqa_paged_dot(
        q,
        cache,
        table,
        token_to_req,
        positions,
        lengths,
        compress_ratio=4,
        num_columns=columns,
    )
    torch.testing.assert_close(out_visible, ref_visible, atol=0, rtol=0)
    torch.testing.assert_close(out_logits, ref_logits, atol=0.2, rtol=0.02)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_paged_graph_replay(rows):
    from flaggems_sglang.reference import get_reference

    device = torch.get_device_module(cfg.device)
    q = torch.randn(rows, 4, 128, device=cfg.device, dtype=torch.bfloat16)
    cache = torch.randn(
        rows * 4, 16, 1, 128, device=cfg.device, dtype=torch.bfloat16
    )
    table = torch.arange(rows * 4, device=cfg.device, dtype=torch.int32).view(
        rows, 4
    )
    requests = torch.arange(rows, device=cfg.device, dtype=torch.int32)
    positions = torch.full((rows,), 255, device=cfg.device, dtype=torch.int32)
    lengths = torch.full((rows,), 256, device=cfg.device, dtype=torch.int32)
    args = [q, cache, table, requests, positions, lengths, 4]
    qwen4_qsa_mqa_paged_dot(*args)
    device.synchronize()
    graph = (
        torch.musa.MUSAGraph()
        if cfg.device == "musa"
        else torch.cuda.CUDAGraph()
    )
    with device.graph(graph):
        logits, visible = qwen4_qsa_mqa_paged_dot(*args)
    for iteration in range(3):
        q.normal_()
        cache.normal_()
        positions.sub_(4)
        logits.fill_(float("nan"))
        visible.fill_(-99)
        graph.replay()
        device.synchronize()
        expected = get_reference("qsa_mqa_paged")(
            *[x.cpu() if isinstance(x, torch.Tensor) else x for x in args]
        )
        torch.testing.assert_close(
            logits.cpu(), expected[0], atol=0.2, rtol=0.02
        )
        torch.testing.assert_close(visible.cpu(), expected[1], atol=0, rtol=0)
