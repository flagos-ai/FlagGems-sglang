# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    rows = shape[0]
    raw = torch.randn(rows * 4, 1, 1, 128, device=device, dtype=cur_dtype)
    table = torch.arange(rows * 4, device=device, dtype=torch.int32).view(
        rows, 4
    )
    requests = torch.arange(rows, device=device, dtype=torch.int32)
    positions = torch.full((rows,), 3, device=device, dtype=torch.int32)
    slots = torch.arange(rows, device=device, dtype=torch.int32)
    cache = torch.zeros(rows, 1, 1, 128, device=device, dtype=cur_dtype)
    weight = torch.randn(128, device=device, dtype=cur_dtype) * 0.01
    angle = torch.randn(4, 32, device=device)
    cos_sin = torch.cat((angle.cos(), angle.sin()), -1)
    yield raw, table, requests, positions, slots, cache, weight, cos_sin, 4, 1e-6, 64, (
        11,
        11,
        10,
    ), True


@pytest.mark.qsa_compress_norm_mrope_store_groups
def test_perf_qsa_compress_norm_mrope_store_groups():
    bench = OpBenchmark(
        op_name="qsa_compress_norm_mrope_store_groups",
        torch_op=get_reference("qsa_compress_norm_mrope_store_groups"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1,), (7,), (64,), (1024,)],
        shape_desc="rows, optional sequence length",
    )
    bench.set_gems(flaggems_sglang.qsa_compress_norm_mrope_store_groups)
    bench.run()
