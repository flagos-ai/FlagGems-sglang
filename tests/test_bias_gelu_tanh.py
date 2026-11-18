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

"""Correctness test for activation/bias_gelu_tanh."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

from . import conftest as cfg

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype, _DEFAULT_TOLERANCE
        )
    )
    tol.update(overrides)
    equal_nan = tol.pop("equal_nan", True)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=equal_nan,
        **tol,
    )


def _case(shape, dtype=torch.bfloat16, seed=0, strided=False):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    n = shape[-1]
    if strided:
        base = torch.randn(
            *shape[:-1],
            3 * n,
            dtype=torch.float32,
            device=cfg.device,
            generator=g,
        ).to(dtype)
        x = base[..., n : 2 * n]
    else:
        x = torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)
    bias = (
        0.5 * torch.randn(n, dtype=torch.float32, device=cfg.device,
                          generator=g)
    ).to(dtype)
    return dict(x=x, bias=bias, check=assert_close)


_CASE_FACTORIES = [
    lambda: _case((4096, 4096)),
    lambda: _case((1, 11008), dtype=torch.float16),
    lambda: _case((32, 11008)),
    lambda: _case((4, 64, 4096), dtype=torch.float32),
    lambda: _case((256, 1024), strided=True),
    lambda: _case((16, 11008), dtype=torch.float16),
    lambda: _case((1, 4096)),
    lambda: _case((33, 100), dtype=torch.float32),
] + [
    (lambda m, n: lambda: _case((m, n)))(m, n)
    for m, n in [
        (512, 4096),
        (4096, 11008),
        (128, 14336),
        (1, 14336),
    ]
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.bias_gelu_tanh
def test_bias_gelu_tanh(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("bias_gelu_tanh")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        from flaggems_sglang import bias_gelu_tanh
    except (ImportError, ModuleNotFoundError):
        pytest.skip("activation/bias_gelu_tanh ops module not found")
        return

    try:
        actual = bias_gelu_tanh(**kwargs)
    except NotImplementedError:
        pytest.skip("activation/bias_gelu_tanh not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    if check is not None:
        case["check"] = check
