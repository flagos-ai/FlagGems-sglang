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

"""Correctness test for rope/ernie45_rope_fused."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("ernie45_rope_fused")


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
# Cases (from kernel-comp-baseline/problems/rope/ernie45_rope_fused/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    aq, ak = actual
    eq, ek = expected
    assert_close(aq, eq)
    assert_close(ak, ek)


def _case(
    num_tokens,
    n_qh,
    n_kh,
    head_size,
    rotary_dim,
    mrope_section,
    max_pos=4096,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    dtype = torch.bfloat16

    q = torch.randn(
        num_tokens,
        n_qh * head_size,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    k = torch.randn(
        num_tokens,
        n_kh * head_size,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    cos_sin_cache = torch.randn(
        max_pos,
        rotary_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    positions = torch.randint(
        0,
        max_pos,
        (3, num_tokens),
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.int64,
    )

    return dict(
        q=q,
        k=k,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        mrope_section=mrope_section,
        head_size=head_size,
        rotary_dim=rotary_dim,
        check=_check,
    )


# mrope_section is [section_h, section_w, section_t] with section_h == section_w
# and section_h + section_w + section_t == rotary_dim // 2 (Ernie4.5 layout).
CORRECTNESS_CASES = [
    _case(1, 4, 1, 64, 64, [8, 8, 16]),
    _case(37, 8, 2, 128, 128, [16, 16, 32]),
    _case(129, 16, 2, 128, 64, [8, 8, 16]),
]

BENCH_CASES = [
    _case(t, 8, 2, 128, 128, [16, 16, 32]) for t in (1, 128, 2048, 8192)
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.ernie45_rope_fused
def test_ernie45_rope_fused(case_idx):
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
        from flaggems_sglang.ops.ernie45_rope_fused import ernie45_rope_fused
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rope/ernie45_rope_fused ops module not found")
        return

    try:
        actual = ernie45_rope_fused(**kwargs)
    except NotImplementedError:
        pytest.skip("rope/ernie45_rope_fused not yet implemented")
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
