# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    rows, selected = shape
    q = torch.randn(rows, 3, 256, device=device, dtype=cur_dtype)
    k = torch.randn(rows * selected, 1, 1, 256, device=device, dtype=cur_dtype)
    v = torch.randn_like(k)
    indices = (
        torch.arange(selected, device=device, dtype=torch.int32)
        .expand(rows, -1)
        .contiguous()
    )
    table = torch.arange(
        rows * selected, device=device, dtype=torch.int32
    ).view(rows, selected)
    requests = torch.arange(rows, device=device, dtype=torch.int32)
    yield q, k, v, indices, table, requests, 256**-0.5, None, torch.randn_like(
        q
    )


@pytest.mark.qsa_sparse_paged_attention
def test_perf_qsa_sparse_paged_attention():
    bench = OpBenchmark(
        op_name="qsa_sparse_paged_attention",
        torch_op=get_reference("qsa_sparse_paged_attention"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1, 512), (7, 2048), (64, 2048)],
        shape_desc="rows, optional sequence length",
    )
    bench.set_gems(flaggems_sglang.qsa_sparse_paged_attention)
    bench.run()
