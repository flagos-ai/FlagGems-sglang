# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def make_inputs(rows, device):
    packed = torch.randn(rows, 324, device=device, dtype=torch.bfloat16)
    return (packed[:, :320], 4)


def _input_fn(shape, cur_dtype, device):
    yield make_inputs(shape[0], device)


@pytest.mark.qwen4_hc_scale_silu
def test_perf_qwen4_hc_scale_silu():
    bench = OpBenchmark(
        op_name="qwen4_hc_scale_silu",
        torch_op=get_reference("qwen4_hc_scale_silu"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(n,) for n in (1, 7, 64, 1024)],
        more_shapes=[(16384,)],
        shape_desc="rows",
    )
    bench.set_gems(flaggems_sglang.qwen4_hc_scale_silu)
    bench.run()
