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

"""Correctness test for mamba/bmm_chunk."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("bmm_chunk")


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
# Cases (from kernel-comp-baseline/problems/mamba/bmm_chunk/cases.py)
# ---------------------------------------------------------------------------


def _check_factory(causal, chunk_size):
    lower_mask = torch.tril(
        torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.bool,
            device=flaggems_sglang.device,
        ),
        diagonal=-1,
    )

    def _check(actual, expected):
        # Output dtype matches the (bf16) inputs, so compare at bf16
        # tolerance rather than the reference's float32 dtype.
        a = actual.clone()
        e = expected.to(actual.dtype)
        if causal:
            # `causal=True` only guarantees i <= j entries; i > j is arbitrary.
            a[..., lower_mask] = 0
            e[..., lower_mask] = 0
        assert_close(a, e, dtype=actual.dtype)

    return _check


def _case(
    batch,
    nchunks,
    chunk_size,
    ngroups,
    k,
    causal=False,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    seqlen = nchunks * chunk_size
    a = torch.randn(
        batch,
        seqlen,
        ngroups,
        k,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    b = torch.randn(
        batch,
        seqlen,
        ngroups,
        k,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    return dict(
        a=a,
        b=b,
        chunk_size=chunk_size,
        causal=causal,
        check=_check_factory(causal, chunk_size),
    )


CORRECTNESS_CASES = [
    _case(1, 1, 8, 2, 16),
    _case(2, 3, 16, 2, 32, causal=True),
    _case(3, 2, 32, 4, 64),
]

BENCH_CASES = [
    _case(8, 16, 256, 8, 64, causal=True),
    _case(32, 4, 256, 8, 64, causal=True),
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.bmm_chunk
def test_bmm_chunk(case_idx):
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
        from flaggems_sglang.ops.bmm_chunk import bmm_chunk
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/bmm_chunk ops module not found")
        return

    try:
        actual = bmm_chunk(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/bmm_chunk not yet implemented")
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
