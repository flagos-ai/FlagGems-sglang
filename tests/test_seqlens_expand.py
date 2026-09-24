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

"""Correctness test for attention/seqlens_expand."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from . import conftest as cfg

reference = get_reference("seqlens_expand")


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
# Cases (from kernel-comp-baseline/problems/attention/seqlens_expand/cases.py)
# ---------------------------------------------------------------------------


import torch

from . import conftest as cfg


def _case(n, max_q=8, seed=0, include_short_kv=True):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    qo = torch.randint(
        1, max_q + 1, (n,), dtype=torch.int32, device=cfg.device, generator=g
    )
    kv = torch.randint(
        1, 8192, (n,), dtype=torch.int32, device=cfg.device, generator=g
    )
    if include_short_kv:
        kv[:: max(n // 4, 1)] = 0
    return dict(
        extend_seq_lens=qo,
        seq_lens=kv,
        total_len=int(qo.sum()),
        max_q_len=int(qo.max()),
        check=assert_close,
    )


CORRECTNESS_CASES = [
    _case(1),
    _case(9, seed=1),
    _case(64, seed=2),
    _case(257, max_q=16, seed=3),
]

BENCH_CASES = [_case(n, max_q=16, seed=7) for n in (1, 8, 64, 512, 4096)]

# 关键约定：bench cases 合并进 accuracy cases
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.seqlens_expand
def test_seqlens_expand(case_idx):
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
        from flaggems_sglang import seqlens_expand
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/seqlens_expand ops module not found")
        return

    try:
        actual = seqlens_expand(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/seqlens_expand not yet implemented")
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
