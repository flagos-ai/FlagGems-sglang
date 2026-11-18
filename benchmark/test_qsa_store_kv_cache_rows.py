# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    rows = shape[0]
    k = torch.zeros(rows + 1, 1, 1, 256, device=device, dtype=cur_dtype)
    v = torch.zeros_like(k)
    slots = torch.arange(rows, device=device, dtype=torch.int32)
    key = torch.randn(rows, 1, 256, device=device, dtype=cur_dtype)
    yield k, v, slots, key, torch.randn_like(key)


@pytest.mark.qsa_store_kv_cache_rows
def test_perf_qsa_store_kv_cache_rows():
    bench = OpBenchmark(
        op_name="qsa_store_kv_cache_rows",
        torch_op=get_reference("qsa_store_kv_cache_rows"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(1,), (7,), (64,), (1024,)],
        shape_desc="rows, optional sequence length",
    )
    bench.set_gems(flaggems_sglang.qsa_store_kv_cache_rows)
    bench.run()
