# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts used by the SGLang 0.5.18 compatibility imports."""

import pytest
import torch

from flaggems_sglang_plugin.compat.dsv41.engram_gather import engram_gather
from flaggems_sglang_plugin.compat.dsv41.engram_gate import fused_engram_gate
from flaggems_sglang_plugin.compat.dsv41.fp4_pack import (
    store_fp4_index_k_cache,
)
from flaggems_sglang_plugin.compat.dsv41.native_fp4 import (
    fp8_index_logits_prefill as dense_fp8_index_logits,
    quantize_bf16_index_queries_fp8,
    unpack_fp4_index_keys_to_fp8,
)
from flaggems_sglang_plugin.compat.dsv41.rmsnorm_fp32 import rmsnorm_fp32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 compatibility operators",
)


@pytest.mark.parametrize("rows,width", [(0, 128), (1, 128), (17, 512)])
def test_rmsnorm_fp32(rows, width):
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(width, device="cuda", dtype=torch.float32)
    expected = (
        w
        * (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        )
    ).bfloat16()
    torch.testing.assert_close(
        rmsnorm_fp32(x, w, 1e-6), expected, rtol=0.008, atol=0.016
    )


@pytest.mark.parametrize("rows", [1, 7])
def test_engram_gate(rows):
    hc, d = 4, 128
    x = torch.randn(rows, hc, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(rows, (hc + 1) * d, device="cuda", dtype=torch.bfloat16)
    qw = torch.randn(hc, d, device="cuda")
    kw = torch.randn_like(qw)
    key, value = (
        kv[:, : hc * d].view(rows, hc, d).float(),
        kv[:, hc * d :].float(),
    )
    inv = torch.rsqrt(x.float().square().mean(-1) + 1e-6) * torch.rsqrt(
        key.square().mean(-1) + 1e-6
    )
    score = ((x.float() * (qw * kw)) * key).sum(-1) * inv * d**-0.5
    gate = torch.sigmoid(
        torch.copysign(score.abs().clamp_min(1e-6).sqrt(), score)
    )
    expected = (x.float() + gate[..., None] * value[:, None, :]).bfloat16()
    torch.testing.assert_close(
        fused_engram_gate(x, kv, qw, kw, 1e-6, 1e-6),
        expected,
        rtol=0.008,
        atol=0.016,
    )


def test_engram_gather_global_range_and_exponents():
    width, block = 128, 32
    weight = (
        torch.arange(4 * width, device="cuda")
        .remainder(16)
        .view(4, width)
        .to(torch.float8_e4m3fn)
    )
    scale = torch.tensor(
        [[0, 126, 127, 128]] * 4, device="cuda", dtype=torch.uint8
    )
    ids = torch.tensor([9, 10, 13, 14], device="cuda")
    out = torch.empty(4, width, device="cuda", dtype=torch.bfloat16)
    engram_gather(
        weight.data_ptr(), scale.data_ptr(), ids, out, width, block, 10, 14
    )
    expected = torch.zeros_like(out)
    for row, src in [(1, 0), (2, 3)]:
        expected[row] = (
            weight[src].float()
            * torch.exp2(scale[src].float() - 127).repeat_interleave(block)
        ).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("page_size", [1, 16, 256])
def test_packed_fp4_physical_slot_roundtrip(page_size):
    # Exact E2M1 values with a maximum of six give an encoded scale of one.
    codebook = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    x = codebook.repeat(3, 8).bfloat16()
    slots = torch.tensor([0, page_size, 2 * page_size - 1], device="cuda")
    if page_size == 1:
        slots = torch.tensor([2, 0, 1], device="cuda")
    table = torch.zeros(3, page_size * 68, device="cuda", dtype=torch.uint8)
    store_fp4_index_k_cache(x, table, slots, page_size=page_size, rne=True)
    actual = unpack_fp4_index_keys_to_fp8(slots, table, page_size)
    torch.testing.assert_close(actual.float(), x.float(), rtol=0, atol=0)
    q = x[:, None].contiguous()
    torch.testing.assert_close(
        quantize_bf16_index_queries_fp8(q).float(), q.float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("heads", [8, 32])
@pytest.mark.parametrize("length", [3, 131])
def test_dense_score_rounding_visibility_and_padding(heads, length):
    q = torch.randint(-2, 3, (3, heads, 128), device="cuda").to(
        torch.float8_e4m3fn
    )
    k = torch.randint(-2, 3, (length, 128), device="cuda").to(
        torch.float8_e4m3fn
    )
    w = torch.randint(-2, 3, (3, heads), device="cuda").bfloat16()
    lens = torch.tensor([0, length // 2, length], device="cuda")
    actual = dense_fp8_index_logits(q, w, k, lens)
    dot = (q.float() @ k.float().T).bfloat16().float().clamp_min(0)
    expected = (
        (dot * w.float()[..., None])
        .bfloat16()
        .float()
        .sum(1)
        .bfloat16()
        .float()
    )
    valid = torch.arange(length, device="cuda")[None, :] < lens[:, None]
    expected.masked_fill_(~valid, float("-inf"))
    torch.testing.assert_close(actual[:, :length], expected, rtol=0, atol=0)
    assert torch.isneginf(actual[:, length:]).all()
