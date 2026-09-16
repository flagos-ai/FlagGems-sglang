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

"""Benchmark for rope/mrope_fused."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

reference = get_reference("mrope_fused")

MROPE_SECTION = [16, 24, 24]
MAX_POSITION = 4096


class MropeFusedBenchmark(OpBenchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "num_tokens, n_qh, n_kh, head_size, rotary_dim"
    # Shapes match kernel-comp-baseline/problems/rope/mrope_fused.
    CORE_SHAPES = [
        (1, 8, 2, 128, 128),
        (128, 8, 2, 128, 128),
        (2048, 8, 2, 128, 128),
        (8192, 8, 2, 128, 128),
    ]

    def get_input_iter(self, cur_dtype):
        for num_tokens, n_qh, n_kh, head_size, rotary_dim in self.shapes:
            q = torch.randn(
                num_tokens,
                n_qh * head_size,
                device=self.device,
                dtype=cur_dtype,
            )
            k = torch.randn(
                num_tokens,
                n_kh * head_size,
                device=self.device,
                dtype=cur_dtype,
            )
            cos_sin_cache = torch.randn(
                MAX_POSITION, rotary_dim, device=self.device, dtype=cur_dtype
            )
            positions = torch.randint(
                0,
                MAX_POSITION,
                (3, num_tokens),
                device=self.device,
                dtype=torch.int64,
            )
            yield (
                q,
                k,
                cos_sin_cache,
                positions,
                MROPE_SECTION,
                head_size,
                rotary_dim,
            )


@pytest.mark.mrope_fused
def test_perf_mrope_fused():
    bench = MropeFusedBenchmark(
        op_name="mrope_fused",
        torch_op=reference,
    )
    bench.set_gems(flaggems_sglang.mrope_fused)
    bench.run()
