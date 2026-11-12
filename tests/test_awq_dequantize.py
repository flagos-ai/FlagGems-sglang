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

"""Correctness test for quantization/awq_dequantize."""

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


def _case(rows, cols, group_size, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    gs = rows if group_size == -1 else group_size
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (rows, cols),
        dtype=torch.int32,
        device=flaggems_sglang.device,
        generator=g,
    )
    scales = torch.rand(
        rows // gs,
        cols * 8,
        dtype=torch.float16,
        device=flaggems_sglang.device,
        generator=g,
    )
    zeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (rows // gs, cols),
        dtype=torch.int32,
        device=flaggems_sglang.device,
        generator=g,
    )
    return dict(
        qweight=qweight, scales=scales, zeros=zeros, check=assert_close
    )


_CORRECTNESS_CASES = [
    lambda: _case(3584, 448, -1),
    lambda: _case(128, 16, 32),
    lambda: _case(256, 32, 64),
    lambda: _case(1024, 128, 128),
]
_CORRECTNESS_CASES += [
    (lambda rows=rows, cols=cols: _case(rows, cols, 128))
    for rows in (3584, 18944)
    for cols in (448, 4736)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.awq_dequantize
def test_awq_dequantize(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("awq_dequantize")(**kwargs)

    try:
        from flaggems_sglang.ops.awq_dequantize import awq_dequantize
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/awq_dequantize ops module not found")
        return

    try:
        actual = awq_dequantize(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/awq_dequantize not yet implemented")
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
