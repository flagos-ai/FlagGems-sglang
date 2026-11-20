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

"""Correctness test for activation_norm/grouped_gemma_rmsnorm."""

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


_EPS = 1e-6


def _case(shape, group_size, dtype=torch.bfloat16, seed=0, strided=False,
          eps=_EPS):
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
    weight = (
        0.5
        * torch.randn(n, dtype=torch.float32, device=cfg.device, generator=g)
    ).to(dtype)
    return dict(
        x=x,
        weight=weight,
        group_size=group_size,
        eps=eps,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case((7, 128), 64),
    lambda: _case((83, 1024), 128),
    lambda: _case((4, 32, 2048), 256),
    lambda: _case((64, 512), 128, strided=True),
    lambda: _case((48, 3200), 160, dtype=torch.float16),
    lambda: _case((16, 512), 64, eps=1e-5),
    lambda: _case((33, 2880), 160),
    lambda: _case((128, 2048), 512),
    lambda: _case((5, 1024), 1024),
    lambda: _case((2, 4096), 32, dtype=torch.float32),
] + [
    (lambda m, n, gs: lambda: _case((m, n), gs))(m, n, gs)
    for m, n, gs in [
        (1, 2880, 160),
        (8, 2880, 160),
        (64, 2880, 160),
        (512, 2880, 160),
        (4096, 2880, 160),
        (1, 4096, 128),
        (512, 4096, 128),
        (4096, 4096, 128),
        (256, 8192, 256),
        (2048, 1024, 64),
    ]
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.grouped_gemma_rmsnorm
def test_grouped_gemma_rmsnorm(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("grouped_gemma_rmsnorm")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        from flaggems_sglang import grouped_gemma_rmsnorm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("activation_norm/grouped_gemma_rmsnorm ops module not found")
        return

    try:
        actual = grouped_gemma_rmsnorm(**kwargs)
    except NotImplementedError:
        pytest.skip("activation_norm/grouped_gemma_rmsnorm not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    if check is not None:
        case["check"] = check
