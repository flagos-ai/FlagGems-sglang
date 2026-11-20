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

"""Correctness test for activation_norm/wan_rmsnorm_silu."""

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


def _case(shape, dtype=torch.bfloat16, gamma_dtype=None, bias=True,
          rms_scale=1.0, eps=1e-6, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    b, c = shape[0], shape[1]
    gd = dtype if gamma_dtype is None else gamma_dtype
    x = torch.randn(
        *shape, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    gamma = (
        0.5 * torch.randn(c, dtype=torch.float32, device=cfg.device,
                          generator=g)
        + 1.0
    ).to(gd)
    bz = (
        0.25 * torch.randn(c, dtype=torch.float32, device=cfg.device,
                           generator=g)
    ).to(gd) if bias else None
    return dict(
        x=x, gamma=gamma, bias=bz, rms_scale=rms_scale, eps=eps,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case((2, 128, 1024)),
    lambda: _case((1, 512, 4096)),
    lambda: _case((4, 128, 64)),
    lambda: _case((1, 2048, 2048)),
    lambda: _case((2, 256, 512), bias=False),
    lambda: _case((2, 256, 512), rms_scale=None),
    lambda: _case((1, 128, 1024), gamma_dtype=torch.float32),
    lambda: _case((2, 128, 32, 32)),
    lambda: _case((8, 64, 256), dtype=torch.float32),
    lambda: _case((1, 576, 1024), rms_scale=1.4142135, eps=1e-5),
] + [
    (lambda b, c, l: lambda: _case((b, c, l)))(b, c, l)
    for b, c, l in [
        (16, 128, 2048),
        (1, 5120, 512),
        (8, 512, 1024),
        (64, 128, 512),
    ]
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.wan_rmsnorm_silu
def test_wan_rmsnorm_silu(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("wan_rmsnorm_silu")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        from flaggems_sglang import wan_rmsnorm_silu
    except (ImportError, ModuleNotFoundError):
        pytest.skip("activation_norm/wan_rmsnorm_silu ops module not found")
        return

    try:
        actual = wan_rmsnorm_silu(**kwargs)
    except NotImplementedError:
        pytest.skip("activation_norm/wan_rmsnorm_silu not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    if check is not None:
        case["check"] = check
