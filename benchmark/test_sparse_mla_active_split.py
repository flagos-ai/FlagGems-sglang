# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Component benchmark for the Hy4-preview SM90 active-head candidate."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _inputs(shape, cur_dtype, device):
    rows, slots = shape
    generator = torch.Generator(device=device).manual_seed(17)
    q = (
        torch.randn(
            (rows, 64, 576),
            device=device,
            dtype=cur_dtype,
            generator=generator,
        )
        * 0.125
    )
    kv = (
        torch.randn(
            (slots, 1, 576),
            device=device,
            dtype=cur_dtype,
            generator=generator,
        )
        * 0.125
    )
    indices = torch.randint(
        0,
        slots,
        (rows, 1, 2048),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    sink = torch.zeros((64,), dtype=torch.float32, device=device)
    lengths = torch.full((rows,), 2048, dtype=torch.int32, device=device)
    yield q, kv, indices, 576**-0.5, 4, sink, lengths


def _candidate(q, kv, indices, sm_scale, active_heads, sink, lengths):
    return flaggems_sglang.sparse_mla_active_split(
        q,
        kv,
        indices,
        sm_scale,
        active_heads=active_heads,
        sink=sink,
        lengths=lengths,
    )


@pytest.mark.sparse_mla_active_split
@torch.no_grad()
def test_perf_sparse_mla_active_split():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("this candidate is qualified only for SM90")
    bench = OpBenchmark(
        op_name="sparse_mla_active_split",
        torch_op=get_reference("sparse_mla_active_split"),
        input_fn=_inputs,
        dtypes=[torch.bfloat16],
        shapes=[(1, 2113), (4, 2113)],
        more_shapes=[],
        shape_desc="query rows, physical KV slots",
    )
    bench.set_gems(_candidate)
    bench.run()
