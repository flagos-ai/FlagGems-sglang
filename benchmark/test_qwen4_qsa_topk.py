# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def make_inputs(rows, device):
    logits = torch.arange(
        rows * 640, device=device, dtype=torch.float32
    ).reshape(rows, 640)
    starts = torch.arange(rows, device=device, dtype=torch.int32) % 4
    ends = starts + torch.arange(rows, device=device, dtype=torch.int32) % 600
    return logits, starts, ends, 512


def _input_fn(shape, cur_dtype, device):
    yield make_inputs(shape[0], device)


@pytest.mark.qwen4_qsa_topk
def test_perf_qwen4_qsa_topk():
    bench = OpBenchmark(
        op_name="qwen4_qsa_topk",
        torch_op=get_reference("qwen4_qsa_topk"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1,), (7,), (64,)],
        shape_desc="rows",
    )
    bench.set_gems(flaggems_sglang.qwen4_qsa_topk)
    bench.run()
