# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
from __future__ import annotations

import pytest
import torch

from flaggems_sglang import (
    qsa_compress_norm_mrope_store_groups as qwen4_compress_norm_mrope_store_groups,
)

from . import conftest as cfg

DEVICE = cfg.device
EPS = 1e-6
pytestmark = pytest.mark.qsa_compress_norm_mrope_store_groups


def _require_cuda() -> None:
    if not torch.get_device_module(cfg.device).is_available():
        pytest.skip("Qwen4 Triton correctness requires an accelerator")


def _compress_reference(
    raw, rope, table, requests, positions, slots, weight, cos_sin, out
):
    ratio = 4
    for row in range(requests.numel()):
        request = int(requests[row].item())
        end = int(positions[row].item())
        slot = int(slots[row].item())
        raw_rows = []
        for offset in range(ratio):
            pos = end - (ratio - 1 - offset)
            page = int(table[request, pos // raw.shape[1]].item())
            raw_rows.append(raw[page, pos % raw.shape[1], 0].float())
        pooled = (torch.stack(raw_rows).sum(0) / ratio).to(torch.bfloat16)
        pooled_fp32 = pooled.float()
        variance = pooled_fp32.square().mean() / 1.0
        normalized = (
            pooled_fp32 * torch.rsqrt(variance + EPS) * (weight.float() + 1.0)
        ).to(torch.bfloat16)
        first_pos = end - ratio + 1
        page = int(table[request, first_pos // raw.shape[1]].item())
        axes = rope[page, first_pos % rope.shape[1], 0, :3]
        freq = torch.arange(32, device=raw.device)
        use_h = (freq % 3 == 1) & (freq < 3 * 11)
        use_w = (freq % 3 == 2) & (freq < 3 * 10)
        axis_pos = torch.where(
            use_h,
            axes[1],
            torch.where(use_w, axes[2], axes[0]),
        )
        cos = cos_sin[axis_pos, freq].to(torch.bfloat16)
        sin = cos_sin[axis_pos, 32 + freq].to(torch.bfloat16)
        first = normalized[:32]
        second = normalized[32:64]
        rotated_first = (first * cos - second * sin).to(torch.bfloat16)
        rotated_second = (second * cos + first * sin).to(torch.bfloat16)
        stored = torch.cat(
            (rotated_first, rotated_second, normalized[64:]), dim=0
        )
        block, token = slot // out.shape[1], slot % out.shape[1]
        out[block, token, 0].copy_(stored)


@pytest.mark.qwen4_qsa
def test_qwen4_qsa_fused_compress_norm_mrope_interleaved_matches_torch():
    _require_cuda()
    page_size, raw_blocks = 8, 2
    torch.manual_seed(404)
    raw = torch.randn(
        (raw_blocks, page_size, 1, 128), device=DEVICE, dtype=torch.bfloat16
    )
    rope = torch.empty(
        (raw_blocks, page_size, 1, 3), device=DEVICE, dtype=torch.int64
    )
    rope[..., 0] = (
        torch.arange(page_size, device=DEVICE).view(1, page_size, 1) % 16
    )
    rope[..., 1] = (
        torch.arange(page_size, device=DEVICE).view(1, page_size, 1) + 1
    ) % 16
    rope[..., 2] = (
        torch.arange(page_size, device=DEVICE).view(1, page_size, 1) + 2
    ) % 16
    table = torch.tensor([[0, 1]], device=DEVICE, dtype=torch.int32)
    requests = torch.tensor([0, 0], device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([3, 7], device=DEVICE, dtype=torch.int64)
    slots = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
    weight = torch.randn((128,), device=DEVICE, dtype=torch.bfloat16) * 0.01
    t = torch.arange(16, device=DEVICE, dtype=torch.float32)
    freq = torch.arange(32, device=DEVICE, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (freq / 32.0))
    angles = t[:, None] * inv_freq[None, :]
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(
        torch.bfloat16
    )
    initial = torch.full_like(raw, 7)
    expected = initial.clone()
    _compress_reference(
        raw, rope, table, requests, positions, slots, weight, cos_sin, expected
    )

    actual = initial.clone()
    qwen4_compress_norm_mrope_store_groups(
        raw,
        table,
        requests,
        positions,
        slots,
        actual,
        weight,
        cos_sin,
        compress_ratio=4,
        norm_eps=EPS,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        mrope_interleaved=True,
        rope_cache=rope,
    )
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)


@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("interleaved", [False, True])
def test_slot_layout_graph_replay(rows, interleaved):
    from flaggems_sglang.reference import get_reference

    device = torch.get_device_module(cfg.device)
    raw = torch.randn(
        rows * 4, 1, 1, 128, device=cfg.device, dtype=torch.bfloat16
    )
    table = torch.arange(rows * 4, device=cfg.device, dtype=torch.int32).view(
        rows, 4
    )
    requests = torch.arange(rows, device=cfg.device, dtype=torch.int32)
    positions = torch.full((rows,), 3, device=cfg.device, dtype=torch.int32)
    slots = torch.arange(rows, device=cfg.device, dtype=torch.int32)
    weight = torch.randn(128, device=cfg.device, dtype=torch.bfloat16) * 0.01
    rope = (
        torch.arange(rows * 4, device=cfg.device, dtype=torch.int64)
        .view(-1, 1, 1, 1)
        .expand(-1, 1, 1, 3)
        .contiguous()
    )
    angle = torch.randn(rows * 4, 32, device=cfg.device, dtype=torch.float32)
    cos_sin = torch.cat((angle.cos(), angle.sin()), -1)
    output = torch.zeros(
        rows + 3, 1, 1, 128, device=cfg.device, dtype=torch.bfloat16
    )
    args = [
        raw,
        table,
        requests,
        positions,
        slots,
        output,
        weight,
        cos_sin,
        4,
        EPS,
        64,
        (11, 11, 10),
        interleaved,
        rope,
    ]
    qwen4_compress_norm_mrope_store_groups(*args)
    device.synchronize()
    graph = (
        torch.musa.MUSAGraph()
        if cfg.device == "musa"
        else torch.cuda.CUDAGraph()
    )
    with device.graph(graph):
        qwen4_compress_norm_mrope_store_groups(*args)
    for iteration in range(3):
        raw.normal_()
        output.fill_(7)
        expected_args = [
            x.cpu().clone() if isinstance(x, torch.Tensor) else x for x in args
        ]
        get_reference("qsa_compress_norm_mrope_store_groups")(*expected_args)
        graph.replay()
        device.synchronize()
        torch.testing.assert_close(
            output.cpu(), expected_args[5], atol=0.03, rtol=0.02
        )
        torch.testing.assert_close(
            output[rows:], torch.full_like(output[rows:], 7), atol=0, rtol=0
        )
