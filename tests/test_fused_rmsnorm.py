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
    (1, 512),
    (7, 1024),
    (64, 2048),
    (127, 3072),
    (16, 4103),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def _reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    mean_square = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(mean_square + eps) * weight.float()).to(
        x.dtype
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.fused_rmsnorm
def test_fused_rmsnorm(shape, dtype):
    torch.manual_seed(sum(shape))
    x = torch.randn(shape, dtype=dtype, device=cfg.device)
    weight = torch.randn(shape[-1], dtype=dtype, device=cfg.device)
    eps = 1e-6

    expected = _reference(x, weight, eps)
    actual = flaggems_sglang.fused_rmsnorm(x, weight, eps)

    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
