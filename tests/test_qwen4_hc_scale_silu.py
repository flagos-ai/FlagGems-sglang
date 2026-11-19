# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang as gems
from flaggems_sglang.reference import get_reference

from . import conftest as cfg


def make_inputs(rows, device):
    packed = torch.randn(rows, 324, device=device, dtype=torch.bfloat16)
    return (packed[:, :320], 4)


@pytest.mark.qwen4_hc_scale_silu
@pytest.mark.parametrize("rows", [1, 7, 64, 1024])
def test_reference(rows):
    torch.manual_seed(42 + rows)
    args = make_inputs(rows, cfg.device)
    expected = get_reference("qwen4_hc_scale_silu")(
        *[x.cpu() if isinstance(x, torch.Tensor) else x for x in args]
    )
    actual = gems.qwen4_hc_scale_silu(*args)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), ref, atol=0.016, rtol=0.016)


@pytest.mark.qwen4_hc_scale_silu
@pytest.mark.parametrize("rows", [1, 7, 64])
def test_graph_replay_changed_input(rows):
    device = torch.get_device_module(cfg.device)
    args = make_inputs(rows, cfg.device)
    gems.qwen4_hc_scale_silu(*args)
    device.synchronize()
    graph = (
        torch.musa.MUSAGraph()
        if cfg.device == "musa"
        else torch.cuda.CUDAGraph()
    )
    with device.graph(graph):
        actual = gems.qwen4_hc_scale_silu(*args)
    outputs = actual if isinstance(actual, tuple) else (actual,)
    for iteration in range(3):
        args[0].normal_()
        for out in outputs:
            out.fill_(float("nan"))
        graph.replay()
        device.synchronize()
        expected = get_reference("qwen4_hc_scale_silu")(
            *[x.cpu() if isinstance(x, torch.Tensor) else x for x in args]
        )
        expected = expected if isinstance(expected, tuple) else (expected,)
        for got, ref in zip(outputs, expected):
            torch.testing.assert_close(got.cpu(), ref, atol=0.016, rtol=0.016)
