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

"""Correctness test for gemm/dsv3_fused_a_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from . import conftest as cfg

reference = get_reference("dsv3_fused_a_gemm")


# ---------------------------------------------------------------------------
# Tolerance helper
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
# Cases (from kernel-comp-baseline/problems/gemm/dsv3_fused_a_gemm/cases.py)
# ---------------------------------------------------------------------------


import torch

from . import conftest as cfg


def _case(num_tokens, hd_in=7168, hd_out=2112, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    a = torch.randn(num_tokens, hd_in, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    # Column-major B: build the row-major [hd_out, hd_in] weight and transpose.
    w = torch.randn(hd_out, hd_in, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    return dict(mat_a=a, mat_b=w.t(), check=assert_close)


CORRECTNESS_CASES = [
    _case(1),
    _case(4, seed=1),
    _case(8, hd_in=4096, hd_out=1536, seed=2),
    _case(16, seed=3),
]

BENCH_CASES = [_case(m, seed=9) for m in (1, 2, 4, 8, 16)]

# 关键约定：bench cases 合并进 accuracy cases
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.dsv3_fused_a_gemm
def test_dsv3_fused_a_gemm(case_idx):
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
        from flaggems_sglang import dsv3_fused_a_gemm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("gemm/dsv3_fused_a_gemm ops module not found")
        return

    try:
        actual = dsv3_fused_a_gemm(**kwargs)
    except NotImplementedError:
        pytest.skip("gemm/dsv3_fused_a_gemm not yet implemented")
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
