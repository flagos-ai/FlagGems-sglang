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

"""Benchmark for quantization/per_group_transpose."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

reference = get_reference("per_group_transpose")

NUM_EXPERTS = 8


class PerGroupTransposeBenchmark(OpBenchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "k, rows_per_expert"
    # Shapes match kernel-comp-baseline/problems/quantization/
    # per_group_transpose; every expert gets ``rows_per_expert`` rows.
    CORE_SHAPES = [(k, n) for k in (128, 512, 4096) for n in (16, 128, 1024)]

    def get_input_iter(self, cur_dtype):
        for k, rows_per_expert in self.shapes:
            counts = [rows_per_expert] * NUM_EXPERTS
            m = sum(counts)
            g = torch.Generator(device=self.device).manual_seed(0)
            a = torch.randn(
                m, k, generator=g, device=self.device, dtype=cur_dtype
            ).contiguous()
            offsets = [0]
            for c in counts:
                offsets.append(offsets[-1] + c)
            expert_offsets = torch.tensor(
                offsets, dtype=torch.int32, device=self.device
            )
            yield a, expert_offsets


@pytest.mark.per_group_transpose
def test_perf_per_group_transpose():
    bench = PerGroupTransposeBenchmark(
        op_name="per_group_transpose",
        torch_op=reference,
    )
    bench.set_gems(flaggems_sglang.per_group_transpose)
    bench.run()
