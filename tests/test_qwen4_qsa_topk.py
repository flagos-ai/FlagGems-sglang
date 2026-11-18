# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang as gems
from flaggems_sglang.reference import get_reference

from . import conftest as cfg


def make_inputs(rows, device):
    logits = torch.arange(
        rows * 640, device=device, dtype=torch.float32
    ).reshape(rows, 640)
    starts = torch.arange(rows, device=device, dtype=torch.int32) % 4
    lengths = torch.tensor(
        [513, 0, 1, 511, 512, 600, 636], device=device, dtype=torch.int32
    )
    ends = starts + lengths[torch.arange(rows, device=device) % 7]
    return logits, starts, ends, 512


@pytest.mark.qwen4_qsa_topk
@pytest.mark.parametrize("rows", [1, 7, 64])
def test_reference_and_graph(rows):
    args = make_inputs(rows, cfg.device)
    reference = get_reference("qwen4_qsa_topk")
    expected = reference(
        *[x.cpu() if isinstance(x, torch.Tensor) else x for x in args]
    )
    torch.testing.assert_close(
        gems.qwen4_qsa_topk(*args).cpu(), expected, atol=0, rtol=0
    )
    device = torch.get_device_module(cfg.device)
    device.synchronize()
    graph = (
        torch.musa.MUSAGraph()
        if cfg.device == "musa"
        else torch.cuda.CUDAGraph()
    )
    with device.graph(graph):
        output = gems.qwen4_qsa_topk(*args)
    args[0].neg_()
    output.fill_(-99)
    graph.replay()
    device.synchronize()
    expected = reference(
        *[x.cpu() if isinstance(x, torch.Tensor) else x for x in args]
    )
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
