# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    rows, columns = shape
    q = torch.randn(rows, 4, 128, device=device, dtype=cur_dtype)
    cache = torch.randn(
        rows * columns, 1, 1, 128, device=device, dtype=cur_dtype
    )
    table = torch.arange(
        rows * columns, device=device, dtype=torch.int32
    ).view(rows, columns)
    request = torch.arange(rows, device=device, dtype=torch.int32)
    position = torch.full(
        (rows,), columns * 4 - 1, device=device, dtype=torch.int32
    )
    lengths = position + 1
    yield q, cache, table, request, position, lengths, 4, columns


@pytest.mark.qsa_mqa_paged
def test_perf_qsa_mqa_paged():
    bench = OpBenchmark(
        op_name="qsa_mqa_paged",
        torch_op=get_reference("qsa_mqa_paged"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1, 512), (7, 2048), (64, 2048)],
        shape_desc="rows, optional sequence length",
    )
    bench.set_gems(flaggems_sglang.qsa_mqa_paged)
    bench.run()
