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

"""Correctness test for fla/chunk_scaled_dot_kkt."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("chunk_scaled_dot_kkt")


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
# Cases (from kernel-comp-baseline/problems/fla/chunk_scaled_dot_kkt/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    assert_close(actual, expected, dtype=torch.float32)


def _case(
    batch,
    nchunks,
    chunk_size,
    hg,
    h,
    k_dim,
    use_g=True,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    t = nchunks * chunk_size
    k = torch.randn(
        batch,
        t,
        hg,
        k_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    beta = torch.rand(
        batch,
        t,
        h,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    )
    g_cumsum = None
    if use_g:
        raw = (
            -torch.rand(
                batch,
                t,
                h,
                generator=g,
                device=flaggems_sglang.device,
                dtype=torch.float32,
            )
            * 0.1
        )
        g_cumsum = (
            raw.view(batch, nchunks, chunk_size, h)
            .cumsum(dim=2)
            .view(batch, t, h)
        )
    return dict(
        k=k, beta=beta, g_cumsum=g_cumsum, chunk_size=chunk_size, check=_check
    )


CORRECTNESS_CASES = [
    _case(1, 1, 16, 2, 2, 32),
    _case(2, 3, 16, 2, 4, 32, use_g=False),
    _case(3, 2, 32, 4, 8, 64),
]

BENCH_CASES = [
    _case(8, 16, 64, 8, 32, 128),
    _case(32, 4, 64, 8, 32, 128),
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.chunk_scaled_dot_kkt
def test_chunk_scaled_dot_kkt(case_idx):
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
        from flaggems_sglang import chunk_scaled_dot_kkt
    except (ImportError, ModuleNotFoundError):
        pytest.skip("fla/chunk_scaled_dot_kkt ops module not found")
        return

    try:
        actual = chunk_scaled_dot_kkt(**kwargs)
    except NotImplementedError:
        pytest.skip("fla/chunk_scaled_dot_kkt not yet implemented")
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
