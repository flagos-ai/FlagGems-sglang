# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    rows = shape[0]
    key = torch.randn(rows, 4, 2560, device=device, dtype=cur_dtype)
    query = torch.randn_like(key)
    value = torch.randn(rows, 2560, device=device, dtype=cur_dtype)
    weights = [
        torch.randn(10240, device=device, dtype=cur_dtype) * 0.01
        for _ in range(3)
    ]
    yield key, query, value, *weights, 4, 1e-6


@pytest.mark.ple_gate_norm
def test_perf_ple_gate_norm():
    bench = OpBenchmark(
        op_name="ple_gate_norm",
        torch_op=get_reference("ple_gate_norm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1,), (7,), (64,), (1024,)],
        shape_desc="rows, optional sequence length",
    )
    bench.set_gems(flaggems_sglang.ple_gate_norm)
    bench.run()
