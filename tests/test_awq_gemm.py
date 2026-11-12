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

"""Correctness test for quantization/awq_gemm."""

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


def _check(actual, expected):
    assert_close(actual.cpu(), expected.cpu(), atol=1e-1, rtol=1e-1)


def _case(m, k, n, group_size, split_k, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    gs = k if group_size == -1 else group_size
    x = torch.rand(
        m, k, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    )
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k, n // 8),
        dtype=torch.int32,
        device=flaggems_sglang.device,
        generator=g,
    )
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k // gs, n // 8),
        dtype=torch.int32,
        device=flaggems_sglang.device,
        generator=g,
    )
    scales = torch.rand(
        k // gs,
        n,
        dtype=torch.float32,
        device=flaggems_sglang.device,
        generator=g,
    )
    return dict(
        input=x,
        qweight=qweight,
        scales=scales,
        qzeros=qzeros,
        split_k_iters=split_k,
        check=_check,
    )


_CORRECTNESS_CASES = [
    lambda: _case(16, 128, 32, 128, 1),
    lambda: _case(24, 128, 64, 32, 8),
    lambda: _case(32, 128, 128, -1, 1),
]
_CORRECTNESS_CASES += [
    (lambda m=m, n=n: _case(m, 4096, n, 128, 8))
    for m in (1, 8, 64, 512)
    for n in (1024, 4096)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.awq_gemm
def test_awq_gemm(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("awq_gemm")(**kwargs)

    try:
        from flaggems_sglang.ops.awq_gemm import awq_gemm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/awq_gemm ops module not found")
        return

    try:
        actual = awq_gemm(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/awq_gemm not yet implemented")
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
