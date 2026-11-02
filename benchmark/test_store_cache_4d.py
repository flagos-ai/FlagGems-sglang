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

"""Benchmark for kvcache/store_cache_4d."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/kvcache/store_cache_4d.
SHAPES = [(n, 8, 128) for n in (1, 8, 64, 512)]
MORE_SHAPES = [(n, 8, 128) for n in (4096,)]

_NUM_PAGES = 4096
_PAGE_SIZE = 64


def _input_fn(shape, cur_dtype, device):
    n, head_num, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    k_view = torch.zeros(
        _NUM_PAGES,
        _PAGE_SIZE,
        head_num,
        head_dim,
        dtype=cur_dtype,
        device=device,
    )
    v_view = torch.zeros_like(k_view)
    cache_k = torch.randn(
        n, head_num, head_dim, device=device, generator=g
    ).to(cur_dtype)
    cache_v = torch.randn(
        n, head_num, head_dim, device=device, generator=g
    ).to(cur_dtype)
    loc = torch.randperm(
        _NUM_PAGES * _PAGE_SIZE, device="cpu", generator=perm_g
    )[:n].to(device=device, dtype=torch.int64)
    yield k_view, v_view, cache_k, cache_v, loc, _PAGE_SIZE


def _case_fn(shape, cur_dtype):
    (n, head_num, head_dim) = shape
    yield BenchmarkCasePlan(
        shape={
            "k_view": (_NUM_PAGES, _PAGE_SIZE, head_num, head_dim),
            "v_view": (_NUM_PAGES, _PAGE_SIZE, head_num, head_dim),
            "cache_k": (n, head_num, head_dim),
            "cache_v": (n, head_num, head_dim),
            "loc": (n,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_NUM_PAGES": _NUM_PAGES,
            "_PAGE_SIZE": _PAGE_SIZE,
            "n": n,
            "head_num": head_num,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.store_cache_4d
def test_perf_store_cache_4d():
    bench = OpBenchmark(
        op_name="store_cache_4d",
        torch_op=get_reference("store_cache_4d"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n, head_num, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
