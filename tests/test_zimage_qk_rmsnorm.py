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

"""Correctness test for diffusion/zimage_qk_rmsnorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

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
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=True,
        **tol,
    )


_EPS = 1e-6
_HEAD_DIM = 128


def _case(b, s, heads, seed=0, from_qkv=False):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    if from_qkv:
        qkv = torch.randn(
            b,
            s,
            3 * heads,
            _HEAD_DIM,
            dtype=torch.float32,
            device=flaggems_sglang.device,
            generator=g,
        ).to(torch.bfloat16)
        x = qkv[:, :, :heads]
    else:
        x = torch.randn(
            b,
            s,
            heads,
            _HEAD_DIM,
            dtype=torch.float32,
            device=flaggems_sglang.device,
            generator=g,
        ).to(torch.bfloat16)
    weight = torch.randn(
        _HEAD_DIM,
        dtype=torch.float32,
        device=flaggems_sglang.device,
        generator=g,
    ).to(torch.bfloat16)
    return dict(x=x, weight=weight.contiguous(), eps=_EPS, check=assert_close)


_CORRECTNESS_CASES = [
    lambda: _case(1, 64, 8),
    lambda: _case(2, 1024, 16),
    lambda: _case(1, 128, 8, from_qkv=True),
]
_CORRECTNESS_CASES += [
    (lambda s=s: _case(1, s, 32)) for s in (128, 1024, 4360, 16384)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.zimage_qk_rmsnorm
def test_zimage_qk_rmsnorm(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("zimage_qk_rmsnorm")(**kwargs)

    try:
        from flaggems_sglang.ops.zimage_qk_rmsnorm import zimage_qk_rmsnorm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/zimage_qk_rmsnorm ops module not found")
        return

    try:
        actual = zimage_qk_rmsnorm(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/zimage_qk_rmsnorm not yet implemented")
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
