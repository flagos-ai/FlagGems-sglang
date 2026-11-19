# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch
from flaggems_sglang import qsa_build_decode_metadata
from flaggems_sglang.reference import get_reference

from . import conftest as cfg

pytestmark = pytest.mark.qsa_build_decode_metadata


def make_inputs(rows, ratio=8, stride=1):
    lengths = torch.tensor(
        ([1, 7, 8, 1024, 2048, 2049, 8192] * rows)[:rows],
        dtype=torch.int64,
        device=cfg.device,
    )
    request = torch.arange(1, rows + 1, device=cfg.device, dtype=torch.int64)
    lens_storage = torch.zeros(
        rows * stride, device=cfg.device, dtype=torch.int64
    )
    req_storage = torch.zeros_like(lens_storage)
    lens_storage[::stride] = lengths
    req_storage[::stride] = request
    table = torch.arange(
        (rows + 2) * 8192 * stride, device=cfg.device, dtype=torch.int64
    ).reshape(rows + 2, -1)[:, ::stride]
    table[0].zero_()
    one = lambda: torch.full(
        (rows,), -99, device=cfg.device, dtype=torch.int32
    )
    outputs = [
        one(),
        one(),
        one(),
        one(),
        one(),
        torch.full((rows, 65), -99, device=cfg.device, dtype=torch.int32),
        one(),
        torch.full((rows,), -99, device=cfg.device, dtype=torch.int64),
        torch.full((rows, ratio), -99, device=cfg.device, dtype=torch.int32),
    ]
    return [
        lens_storage[::stride],
        req_storage[::stride],
        table,
        *outputs,
        ratio,
        128,
    ]


@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
def test_qsa_build_decode_metadata_exact_and_graph(rows, padding, stride):
    args = make_inputs(rows, stride=stride) + [int(padding)]
    device = torch.get_device_module(cfg.device)
    qsa_build_decode_metadata(*args)
    device.synchronize()
    for recapture in range(2):
        if recapture:
            args = make_inputs(rows, stride=stride) + [int(padding)]
        graph = (
            device.MUSAGraph() if cfg.device == "musa" else device.CUDAGraph()
        )
        with device.graph(graph):
            qsa_build_decode_metadata(*args)
        for phase in range(3):
            args[0].copy_(
                torch.tensor(
                    ([1024, 2048, 2049, 8192, 1, 8, 9][phase:] + [1, 8, 9])
                    * rows,
                    device=cfg.device,
                )[:rows]
            )
            args[1].copy_(torch.roll(args[1], 1))
            args[2].add_(128)
            args[2][0].zero_()
            for out in args[3:12]:
                out.fill_(-123)
            expected = [
                x.cpu().clone() if isinstance(x, torch.Tensor) else x
                for x in args
            ]
            get_reference("qsa_build_decode_metadata")(*expected)
            graph.replay()
            device.synchronize()
            for actual, ref in zip(args[3:12], expected[3:12]):
                torch.testing.assert_close(actual.cpu(), ref, atol=0, rtol=0)
