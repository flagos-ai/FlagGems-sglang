# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for quantization/quant_store_kv_mxfp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_PAGE = 128
# (t, nkv)
SHAPES = [(t, nkv) for t in (16, 128, 1024, 4096) for nkv in (1, 2)]
MORE_SHAPES = [(16, 1, 128), (512, 1, 512)]


def _input_fn(shape, cur_dtype, device):
    t, nkv = shape
    d = 128
    num_pages = max(2, t // _PAGE + 1)
    g = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(
        t, nkv, d, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    v = torch.randn(
        t, nkv, d, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    slots = torch.randperm(num_pages * _PAGE, generator=g, device=device)[:t]
    loc = slots.to(torch.int32)
    sf = d // 32
    k_cache = torch.zeros(
        (num_pages * _PAGE, nkv, d), dtype=torch.float8_e4m3fn, device=device
    )
    v_cache = torch.zeros_like(k_cache)
    sfk = torch.zeros(
        (num_pages, nkv, 32, _PAGE // 32, sf), dtype=torch.uint8, device=device
    )
    sfv = torch.zeros_like(sfk)
    yield dict(
        k=k.contiguous(),
        v=v.contiguous(),
        loc=loc,
        k_cache=k_cache,
        v_cache=v_cache,
        sfk=sfk,
        sfv=sfv,
        page_size=_PAGE,
    )


@pytest.mark.quant_store_kv_mxfp8
def test_perf_quant_store_kv_mxfp8():
    gems_op = flaggems_sglang.get_op("quant_store_kv_mxfp8")
    if gems_op is None:
        pytest.skip("quantization/quant_store_kv_mxfp8 not implemented yet")
    bench = OpBenchmark(
        op_name="quant_store_kv_mxfp8",
        torch_op=get_reference("quant_store_kv_mxfp8"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="t, nkv",
    )
    bench.set_gems(gems_op)
    bench.run()
