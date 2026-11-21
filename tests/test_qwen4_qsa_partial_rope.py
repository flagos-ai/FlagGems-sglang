# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang as gems
from flaggems_sglang.reference import get_reference

from . import conftest as cfg


def make_inputs(rows, device):
    storage = torch.randn(rows, 4, 136, device=device, dtype=torch.bfloat16)
    x = storage[..., :128]
    positions = (
        torch.arange(rows, device=device, dtype=torch.int64)
        .expand(3, -1)
        .clone()
    )
    positions[1] += 1
    positions[2] += 2
    angle = torch.randn(rows + 3, 32, device=device)
    cache = torch.cat((angle.cos(), angle.sin()), -1)
    axis = torch.arange(32, device=device, dtype=torch.int32) % 3
    return x, positions, cache, axis, 64, True


@pytest.mark.qwen4_qsa_partial_rope
@pytest.mark.parametrize("rows", [1, 7, 64])
def test_reference_and_graph(rows):
    args = make_inputs(rows, cfg.device)
    reference = get_reference("qwen4_qsa_partial_rope")
    expected = reference(
        *[x.cpu() if isinstance(x, torch.Tensor) else x for x in args]
    )
    torch.testing.assert_close(
        gems.qwen4_qsa_partial_rope(*args).cpu(), expected, atol=0, rtol=0
    )
    device = torch.get_device_module(cfg.device)
    device.synchronize()
    graph = (
        torch.musa.MUSAGraph()
        if cfg.device == "musa"
        else torch.cuda.CUDAGraph()
    )
    with device.graph(graph):
        output = gems.qwen4_qsa_partial_rope(*args)
    args[0].add_(1)
    output.fill_(-99)
    graph.replay()
    device.synchronize()
    expected = reference(
        *[x.cpu() if isinstance(x, torch.Tensor) else x for x in args]
    )
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
