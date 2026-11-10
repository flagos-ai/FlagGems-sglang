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

"""Correctness test for quantization/w8a8_block_fp8_matmul."""

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
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        equal_nan=True,
        **tol,
    )


_BLOCK = [128, 128]
_FP8_DTYPE = torch.float8_e4m3fn


def _case(m, n, k, dtype=torch.bfloat16, seed=0, *, scale_range=(0.0, 1e-2)):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    block_n, block_k = _BLOCK
    fp8_max = torch.finfo(_FP8_DTYPE).max
    A = torch.randn(
        m, k, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    ).clamp(-fp8_max, fp8_max).to(_FP8_DTYPE)
    B = torch.randn(
        n, k, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    ).clamp(-fp8_max, fp8_max).to(_FP8_DTYPE)
    scale_min, scale_max = scale_range
    As = (
        scale_min
        + (scale_max - scale_min)
        * torch.rand(
            m, k // block_k, dtype=torch.float32, device=flaggems_sglang.device, generator=g
        )
    ).contiguous()
    Bs = (
        scale_min
        + (scale_max - scale_min)
        * torch.rand(
            n // block_n, k // block_k, dtype=torch.float32, device=flaggems_sglang.device, generator=g
        )
    ).contiguous()
    return dict(A=A, B=B, As=As, Bs=Bs, block_size=_BLOCK, output_dtype=dtype, check=_check)


def _check(actual, expected):
    assert_close(actual, expected, atol=5e-2, rtol=1e-2)


_CORRECTNESS_CASES = [
    lambda: _case(7, 256, 512),
    lambda: _case(64, 1024, 512),
    lambda: _case(256, 1024, 4096),
]
_NORMAL_SCALE_CASES = [
    lambda: _case(7, 256, 512, seed=11, scale_range=(0.5, 1.5)),
    lambda: _case(64, 1024, 512, seed=12, scale_range=(0.5, 1.5)),
    lambda: _case(256, 1024, 4096, seed=13, scale_range=(0.5, 1.5)),
]
_CORRECTNESS_CASES += _NORMAL_SCALE_CASES
_CORRECTNESS_CASES += [
    (lambda m=m, n=n, k=k: _case(m, n, k))
    for m in (1, 8, 64, 512, 4096)
    for n, k in ((1024, 4096), (4096, 4096), (7168, 4096))
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.w8a8_block_fp8_matmul
def test_w8a8_block_fp8_matmul(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = case.pop("check", None) if isinstance(case, dict) and "check" in case else None
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("w8a8_block_fp8_matmul")(**kwargs)

    try:
        from flaggems_sglang.ops.w8a8_block_fp8_matmul import w8a8_block_fp8_matmul
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/w8a8_block_fp8_matmul ops module not found")
        return

    try:
        actual = w8a8_block_fp8_matmul(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/w8a8_block_fp8_matmul not yet implemented")
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


@pytest.mark.parametrize("case_idx", range(len(_NORMAL_SCALE_CASES)))
def test_w8a8_block_fp8_matmul_rejects_incorrect_scaling(case_idx):
    case = _NORMAL_SCALE_CASES[case_idx]()
    kwargs = {key: value for key, value in case.items() if key != "check"}
    expected = get_reference("w8a8_block_fp8_matmul")(**kwargs)
    check = case["check"]
    for actual in (torch.zeros_like(expected), expected * 0.5):
        with pytest.raises(AssertionError):
            check(actual, expected)
    for missing_scales in [("As",), ("Bs",), ("As", "Bs")]:
        broken_kwargs = dict(kwargs)
        for key in missing_scales:
            broken_kwargs[key] = torch.ones_like(kwargs[key])
        actual = get_reference("w8a8_block_fp8_matmul")(**broken_kwargs)
        with pytest.raises(AssertionError):
            check(actual, expected)
