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

"""Correctness test for quantization/mxfp8_block_scaled_matmul."""

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


_FP8_MAX = 448.0


def _quantize_mx(x):
    m, k = x.shape
    xg = x.reshape(m, k // 32, 32).to(torch.float32)
    amax = xg.abs().amax(-1).clamp(min=1e-12)
    exp = torch.ceil(torch.log2(amax / _FP8_MAX)).clamp(-127, 127)
    scale = torch.exp2(exp)
    xq = (
        (xg / scale.unsqueeze(-1))
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    scale_u8 = (exp + 127).to(torch.uint8)
    return xq.reshape(m, k), scale_u8


def _case(m, k, n, output_dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    a_f = torch.randn(
        m, k, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    )
    b_f = torch.randn(
        n, k, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    )
    a, a_scale = _quantize_mx(a_f)
    b, b_scale = _quantize_mx(b_f)
    return dict(
        a=a,
        a_scale=a_scale,
        b=b,
        b_scale=b_scale,
        output_dtype=output_dtype,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual, expected, atol=0.5, rtol=2e-2)


_CORRECTNESS_CASES = [
    lambda: _case(64, 128, 128),
    lambda: _case(130, 256, 384),
    lambda: _case(256, 1024, 512),
]
_CORRECTNESS_CASES += [
    (lambda m=m, n=n: _case(m, 4096, n))
    for m in (128, 1024, 4096)
    for n in (4096, 7168)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.mxfp8_block_scaled_matmul
def test_mxfp8_block_scaled_matmul(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("mxfp8_block_scaled_matmul")(**kwargs)

    try:
        from flaggems_sglang.ops.mxfp8_block_scaled_matmul import (
            mxfp8_block_scaled_matmul,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "quantization/mxfp8_block_scaled_matmul ops module not found"
        )
        return

    try:
        actual = mxfp8_block_scaled_matmul(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "quantization/mxfp8_block_scaled_matmul not yet implemented"
        )
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
