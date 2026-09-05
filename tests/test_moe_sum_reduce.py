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

from . import conftest as cfg

SHAPES = [
    (1, 9, 4096),
    (7, 9, 4096),
    (64, 8, 4096),
    (127, 3, 1536),
    (16, 17, 4103),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def _reference(
    input: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    return input.float().sum(dim=1).mul(routed_scaling_factor).to(input.dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.moe_sum_reduce
def test_moe_sum_reduce(shape, dtype):
    torch.manual_seed(sum(shape))
    input = torch.randn(shape, dtype=dtype, device=cfg.device)
    routed_scaling_factor = 0.3

    expected = _reference(input, routed_scaling_factor)
    actual = flaggems_sglang.moe_sum_reduce(input, routed_scaling_factor)

    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
