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

import pytest
import torch

import flaggems_sglang

from .attri_util import MOE_SUM_REDUCE_BENCH_SHAPES


@pytest.mark.parametrize("shape", MOE_SUM_REDUCE_BENCH_SHAPES)
@pytest.mark.moe_sum_reduce
def test_moe_sum_reduce(shape, benchmark):
    input = torch.randn(
        shape, dtype=torch.bfloat16, device=flaggems_sglang.device
    )

    benchmark(flaggems_sglang.moe_sum_reduce, input, 0.3)
