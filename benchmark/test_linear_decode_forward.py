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

"""Benchmark for attention/linear_decode_forward."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/linear_decode_forward.
SHAPES = [(b, 8, 128) for b in (1, 8, 64)]
MORE_SHAPES = [(b, 8, 128) for b in (256,)]

_BLOCK_SIZE = 32


def _input_fn(shape, cur_dtype, device):
    b, h, d = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    num_slots = b + 2

    def _t(*size):
        return torch.randn(*size, device=device, generator=g).to(cur_dtype)

    slot_idx = torch.randperm(num_slots, device="cpu", generator=perm_g)[
        :b
    ].to(device=device, dtype=torch.int32)
    yield (
        _t(b, h, 1, d).contiguous(),
        _t(b, h, 1, d).contiguous(),
        _t(b, h, 1, d).contiguous(),
        _t(num_slots, h, d, d).contiguous(),
        torch.rand(h, dtype=torch.float32, device=device, generator=g),
        slot_idx,
        _BLOCK_SIZE,
    )


def _case_fn(shape, cur_dtype):
    (b, h, d) = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (b, h, 1, d),
            "k": (b, h, 1, d),
            "v": (b, h, 1, d),
            "state": (b + 2, h, d, d),
            "slope": (h,),
            "slot_idx": (b,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_BLOCK_SIZE": _BLOCK_SIZE,
            "b": b,
            "h": h,
            "d": d,
        },
        builder_args=(shape,),
    )


@pytest.mark.linear_decode_forward
def test_perf_linear_decode_forward():
    bench = OpBenchmark(
        op_name="linear_decode_forward",
        torch_op=get_reference("linear_decode_forward"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="b, h, d",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
