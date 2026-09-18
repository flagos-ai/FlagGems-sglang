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

"""Correctness test for quantization/w8a8_block_int8_matmul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("w8a8_block_int8_matmul")


# ---------------------------------------------------------------------------
# Tolerance helper (from kernel-comp-baseline/harness/correctness.py)
# ---------------------------------------------------------------------------

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
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/quantization/w8a8_block_int8_matmul/cases.py)
# ---------------------------------------------------------------------------


_BLOCK = [128, 128]


def _case(m, n, k, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    block_n, block_k = _BLOCK
    A = torch.randint(
        -8,
        8,
        (m, k),
        dtype=torch.int8,
        device=flaggems_sglang.device,
        generator=g,
    )
    B = torch.randint(
        -8,
        8,
        (n, k),
        dtype=torch.int8,
        device=flaggems_sglang.device,
        generator=g,
    )
    As = (
        1e-2
        * torch.rand(
            m,
            k // block_k,
            dtype=torch.float32,
            device=flaggems_sglang.device,
            generator=g,
        )
    ).contiguous()
    Bs = (
        1e-2
        * torch.rand(
            n // block_n,
            k // block_k,
            dtype=torch.float32,
            device=flaggems_sglang.device,
            generator=g,
        )
    ).contiguous()
    return dict(
        A=A,
        B=B,
        As=As,
        Bs=Bs,
        block_size=_BLOCK,
        output_dtype=dtype,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual, expected, atol=0.5, rtol=1e-2)


CORRECTNESS_CASES = [
    _case(7, 256, 512),
    _case(64, 1024, 512),
    _case(256, 1024, 4096),
]

BENCH_CASES = [
    _case(m, n, k)
    for m in (1, 8, 64, 512, 4096)
    for n, k in ((1024, 4096), (4096, 4096), (7168, 4096))
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.w8a8_block_int8_matmul
def test_w8a8_block_int8_matmul(case_idx):
    case = CORRECTNESS_CASES[case_idx]
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    # Reference
    expected = reference(**kwargs)

    # Operator under test
    try:
        from flaggems_sglang.ops.w8a8_block_int8_matmul import (
            w8a8_block_int8_matmul,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/w8a8_block_int8_matmul ops module not found")
        return

    try:
        actual = w8a8_block_int8_matmul(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/w8a8_block_int8_matmul not yet implemented")
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
