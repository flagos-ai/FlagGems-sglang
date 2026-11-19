# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    requests, length = shape
    rows = requests * length
    x = torch.randn(rows, 10240, device=device, dtype=cur_dtype)
    state = torch.randn(requests, 10240, 9, device=device, dtype=cur_dtype)
    weight = torch.randn(10240, 4, device=device, dtype=cur_dtype) * 0.1
    starts = (
        torch.arange(requests + 1, device=device, dtype=torch.int32) * length
    )
    indices = torch.arange(requests, device=device, dtype=torch.int32)
    initial = torch.ones(requests, device=device, dtype=torch.bool)
    yield x, torch.empty_like(x), state, weight, starts, indices, initial, {
        "num_prefills": requests,
        "max_len": length,
        "state_len": 9,
        "kernel_width": 4,
        "dilation": 3,
        "null_block_id": -1,
    }


class PLEStateBenchmark(OpBenchmark):
    def get_latency(self, op, *args, **kwargs):
        return self._get_latency_with_input_reset(op, *args, **kwargs)


@pytest.mark.ple_prefill_short_conv_
def test_perf_ple_prefill_short_conv_():
    bench = PLEStateBenchmark(
        op_name="ple_prefill_short_conv_",
        torch_op=get_reference("ple_prefill_short_conv_"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1, 1024), (4, 1024)],
        shape_desc="rows, optional sequence length",
        is_inplace=True,
        timing_reset_args=(2,),
    )
    bench.set_gems(flaggems_sglang.ple_prefill_short_conv_)
    bench.run()
