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

"""Benchmark for mamba/conv_window_scatter_with_mask."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/conv_window_scatter_with_mask.
SHAPES = [(8, 512, requests, 4, 128, 4) for requests in (8, 64)]
MORE_SHAPES = [(8, 512, requests, 4, 128, 4) for requests in (256,)]


def _input_fn(shape, cur_dtype, device):
    layers, cache, requests, draft, dim, k = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    win = k - 1
    dst = torch.randn(layers, cache, dim, win, device=device, generator=g).to(
        cur_dtype
    )
    # src is an as_strided window over a shared buffer, so its per-step
    # windows overlap; that layout is the point of this variant.
    base = torch.randn(
        layers, requests, dim, draft + win - 1, device=device, generator=g
    ).to(cur_dtype)
    src = base.as_strided(
        (layers, requests, draft, dim, win),
        (
            base.stride(0),
            base.stride(1),
            base.stride(3),
            base.stride(2),
            base.stride(3),
        ),
    )
    dst_idx = torch.randperm(cache, device="cpu", generator=perm_g)[
        :requests
    ].to(device=device, dtype=torch.int32)
    step = torch.randint(
        0,
        draft,
        (requests,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    yield dst.contiguous(), src, dst_idx, step


def _case_fn(shape, cur_dtype):
    (layers, cache, requests, draft, dim, k) = shape
    yield BenchmarkCasePlan(
        shape={
            "dst": (layers, cache, dim, k - 1),
            "src": (layers, requests, draft, dim, k - 1),
            "dst_idx": (requests,),
            "step": (requests,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "layers": layers,
            "cache": cache,
            "requests": requests,
            "draft": draft,
            "dim": dim,
            "k": k,
        },
        builder_args=(shape,),
    )


@pytest.mark.conv_window_scatter_with_mask
def test_perf_conv_window_scatter_with_mask():
    bench = OpBenchmark(
        op_name="conv_window_scatter_with_mask",
        torch_op=get_reference("conv_window_scatter_with_mask"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="layers, cache, requests, draft, dim, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
