# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
from __future__ import annotations

import pytest
import torch

from flaggems_sglang import qsa_store_kv_cache_rows as qwen4_store_qsa_kv_rows

from . import conftest as cfg

DEVICE = cfg.device
EPS = 1e-6
pytestmark = pytest.mark.qsa_store_kv_cache_rows


def _require_cuda() -> None:
    if not torch.get_device_module(cfg.device).is_available():
        pytest.skip("Qwen4 Triton correctness requires an accelerator")


@pytest.mark.qwen4_qsa
def test_qwen4_qsa_kv_store_matches_torch_and_skips_invalid_slots():
    _require_cuda()
    blocks, page_size, heads, dim, rows = 3, 4, 2, 17, 5
    torch.manual_seed(303)
    initial_k = torch.randn(
        (blocks, page_size, heads, dim), device=DEVICE, dtype=torch.bfloat16
    )
    initial_v = torch.randn_like(initial_k)
    key = torch.randn((rows, heads, dim), device=DEVICE, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 3, 4, 11, -1], device=DEVICE, dtype=torch.int64)

    ref_k, ref_v = initial_k.clone(), initial_v.clone()
    valid = (slots >= 0) & (slots < blocks * page_size)
    safe = slots[valid]
    ref_k.index_put_(
        (safe // page_size, safe % page_size), key[valid], accumulate=False
    )
    ref_v.index_put_(
        (safe // page_size, safe % page_size), value[valid], accumulate=False
    )

    out_k, out_v = initial_k.clone(), initial_v.clone()
    qwen4_store_qsa_kv_rows(out_k, out_v, slots, key, value)
    torch.testing.assert_close(out_k, ref_k, atol=0, rtol=0)
    torch.testing.assert_close(out_v, ref_v, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_store_graph_replay_exact(rows):
    from flaggems_sglang import qsa_store_kv_cache_rows
    from flaggems_sglang.reference import get_reference

    device = torch.get_device_module(cfg.device)
    k = torch.zeros(
        rows + 3, 1, 1, 256, device=cfg.device, dtype=torch.bfloat16
    )
    v = torch.zeros_like(k)
    keys = torch.randn(rows, 1, 256, device=cfg.device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    slots = torch.arange(rows, device=cfg.device, dtype=torch.int32)
    args = [k, v, slots, keys, values]
    qsa_store_kv_cache_rows(*args)
    device.synchronize()
    graph = (
        torch.musa.MUSAGraph()
        if cfg.device == "musa"
        else torch.cuda.CUDAGraph()
    )
    with device.graph(graph):
        qsa_store_kv_cache_rows(*args)
    for iteration in range(3):
        k.fill_(7)
        v.fill_(9)
        keys.normal_()
        values.normal_()
        expected = [x.cpu().clone() for x in args]
        get_reference("qsa_store_kv_cache_rows")(*expected)
        graph.replay()
        device.synchronize()
        torch.testing.assert_close(k.cpu(), expected[0], atol=0, rtol=0)
        torch.testing.assert_close(v.cpu(), expected[1], atol=0, rtol=0)
