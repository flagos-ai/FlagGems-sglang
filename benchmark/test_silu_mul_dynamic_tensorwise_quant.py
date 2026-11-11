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

"""Benchmark for silu_mul_dynamic_tensorwise_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

OP = "silu_mul_dynamic_tensorwise_quant"

# (rows, intermediate_size, active_rows), the same widths as the static sibling
# so the two benchmarks can be read side by side.
#
# Every case keeps active_rows equal to the row count on purpose. The reference
# selects the active prefix on the device and so computes the activation over
# the whole buffer, which means a partly active case would compare a kernel that
# skips the tail against a baseline that does not: the ratio would be reporting
# the masking asymmetry rather than the fusion. The masked path is covered by
# the correctness tests instead.
SHAPES = [
    # A single decoded token: launch bound, nothing to fill the device with.
    (1, 11008, 1),
    (64, 2048, 64),
    # Prefill batches over the common widths.
    (8, 11008, 8),
    (32, 1408, 32),
    (128, 512, 128),
    (512, 256, 512),
]
MORE_SHAPES = [
    (1024, 512, 1024),
    (256, 4096, 256),
]


def _input_fn(shape, cur_dtype, device):
    rows, intermediate_size, active = shape
    generator = torch.Generator(device=device).manual_seed(11)
    width = 2 * intermediate_size
    # The gate half and the up half come from one draw so the two halves keep
    # realistic relative magnitudes.
    x = (
        torch.randn(
            (rows, width),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(3.0)
        .to(cur_dtype)
    )
    buffer = torch.randn(
        (rows, intermediate_size),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    # Dynamic scaling derives the scale from the data, so this slot is only
    # checked for presence, not read for a value.
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    num_tokens = torch.tensor([active], device=device, dtype=torch.int32)
    yield x, buffer, scale, num_tokens, rows, intermediate_size


def _case_fn(shape, cur_dtype):
    rows, intermediate_size, active = shape
    yield BenchmarkCasePlan(
        shape={
            "input": (rows, 2 * intermediate_size),
            "output": (rows, intermediate_size),
        },
        params={
            "configuration": shape,
            "seed": 11,
            "rows": rows,
            "intermediate_size": intermediate_size,
            "active_rows": active,
        },
        builder_args=(shape,),
    )


@pytest.mark.silu_mul_dynamic_tensorwise_quant
def test_perf_silu_mul_dynamic_tensorwise_quant():
    bench = OpBenchmark(
        op_name=OP,
        torch_op=get_reference(OP),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, intermediate_size, active_rows",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
