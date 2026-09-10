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

import flaggems_sglang
import pytest
import torch
from benchmark.attri_util import INTERLEAVED_ROPE_BENCH_CASES


@pytest.mark.parametrize("seq_len,mrope_section", INTERLEAVED_ROPE_BENCH_CASES)
@pytest.mark.interleaved_rope
def test_interleaved_rope(seq_len, mrope_section, benchmark):
    dim = sum(mrope_section) * 3
    x = torch.randn(
        (3, seq_len, dim),
        dtype=torch.bfloat16,
        device=flaggems_sglang.device,
    )

    benchmark(flaggems_sglang.interleaved_rope, x, mrope_section)
