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

from . import conftest as cfg

CASES = [
    (1, [1, 1, 1]),
    (65, [16, 8, 8]),
    (127, [7, 17, 19]),
    (2049, [32, 64, 32]),
    (3, [0, 0, 1]),
    (63, [0, 3, 0]),
    (64, [5, 0, 0]),
    (129, [0, 1, 2]),
    (17, [128, 128, 128]),
    (2, [342, 0, 0]),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _reference(x, mrope_section):
    dim = x.shape[-1]
    columns = torch.arange(dim, device=x.device)
    use_height = (columns % 3 == 1) & (columns < mrope_section[1] * 3)
    use_width = (columns % 3 == 2) & (columns < mrope_section[2] * 3)
    output = x[0].clone()
    output[:, use_height] = x[1][:, use_height]
    output[:, use_width] = x[2][:, use_width]
    return output


@pytest.mark.parametrize("seq_len,mrope_section", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.interleaved_rope
def test_interleaved_rope(seq_len, mrope_section, dtype):
    dim = sum(mrope_section) * 3
    torch.manual_seed(seq_len + dim)
    x = torch.randn((3, seq_len, dim), dtype=dtype, device=cfg.device)

    expected = _reference(x, mrope_section)
    actual = flaggems_sglang.interleaved_rope(x, mrope_section)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
