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

"""Benchmark for quantization/per_token_group_quant_fp8_moe_mn_major."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# (n_per_expert, n_experts, k)
SHAPES = [
    (n, e, k) for e, n in ((8, 256), (32, 128)) for k in (2048, 4096, 7168)
]
MORE_SHAPES = [(128, 1, 512), (256, 4, 1024)]
_GROUP = 128


def _input_fn(shape, cur_dtype, device):
    n, e, k = shape
    m_total = n * e
    g = torch.Generator(device=device).manual_seed(0)
    A = torch.randn(
        m_total, k, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    offs = torch.tensor(
        [i * n for i in range(e + 1)], dtype=torch.int32, device=device
    )
    problem_sizes = torch.zeros((e, 3), dtype=torch.int32, device=device)
    problem_sizes[:, 0] = n
    problem_sizes[:, 1] = k
    problem_sizes[:, 2] = k
    yield dict(
        A=A.contiguous(),
        expert_offsets=offs,
        problem_sizes=problem_sizes,
        group_size=_GROUP,
    )


@pytest.mark.per_token_group_quant_fp8_moe_mn_major
def test_perf_per_token_group_quant_fp8_moe_mn_major():
    gems_op = flaggems_sglang.get_op("per_token_group_quant_fp8_moe_mn_major")
    if gems_op is None:
        pytest.skip(
            "quantization/per_token_group_quant_fp8_moe_mn_major not implemented yet"
        )
    bench = OpBenchmark(
        op_name="per_token_group_quant_fp8_moe_mn_major",
        torch_op=get_reference("per_token_group_quant_fp8_moe_mn_major"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n_per_expert, n_experts, k",
    )
    bench.set_gems(gems_op)
    bench.run()
