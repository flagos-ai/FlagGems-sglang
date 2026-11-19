# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


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


def _input_fn(shape, cur_dtype, device):
    yield make_inputs(shape[0], device)


@pytest.mark.qwen4_qsa_partial_rope
def test_perf_qwen4_qsa_partial_rope():
    bench = OpBenchmark(
        op_name="qwen4_qsa_partial_rope",
        torch_op=get_reference("qwen4_qsa_partial_rope"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1,), (7,), (64,)],
        shape_desc="rows",
    )
    bench.set_gems(flaggems_sglang.qwen4_qsa_partial_rope)
    bench.run()
