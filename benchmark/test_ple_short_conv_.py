# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    rows = shape[0]
    x = torch.randn(rows, 10240, device=device, dtype=cur_dtype)
    state = torch.randn(rows, 10240, 9, device=device, dtype=cur_dtype)
    weight = torch.randn(10240, 4, device=device, dtype=cur_dtype) * 0.1
    indices = torch.arange(rows, device=device, dtype=torch.int32)
    yield x, torch.empty_like(x), state, weight, indices, {"dilation": 3}


class PLEStateBenchmark(OpBenchmark):
    def get_latency(self, op, *args, **kwargs):
        return self._get_latency_with_input_reset(op, *args, **kwargs)


@pytest.mark.ple_short_conv_
def test_perf_ple_short_conv_():
    bench = PLEStateBenchmark(
        op_name="ple_short_conv_",
        torch_op=get_reference("ple_short_conv_"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1,), (7,), (64,), (1024,)],
        shape_desc="rows, optional sequence length",
        is_inplace=True,
        timing_reset_args=(2,),
    )
    bench.set_gems(flaggems_sglang.ple_short_conv_)
    bench.run()
