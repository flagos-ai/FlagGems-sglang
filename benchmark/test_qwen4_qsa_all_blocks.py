# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def make_inputs(rows, device):
    return torch.arange(rows, device=device, dtype=torch.int32) % 512, 512


def _input_fn(shape, cur_dtype, device):
    yield make_inputs(shape[0], device)


@pytest.mark.qwen4_qsa_all_blocks
def test_perf_qwen4_qsa_all_blocks():
    bench = OpBenchmark(
        op_name="qwen4_qsa_all_blocks",
        torch_op=get_reference("qwen4_qsa_all_blocks"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1,), (7,), (64,)],
        shape_desc="rows",
    )
    bench.set_gems(flaggems_sglang.qwen4_qsa_all_blocks)
    bench.run()
