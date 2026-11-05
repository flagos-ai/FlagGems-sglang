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

"""Correctness test for attention/fixup_zero_kv."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

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
# Cases (from
# kernel-comp-baseline/problems/attention/fixup_zero_kv/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    torch.testing.assert_close(actual[0], expected[0])
    # The zeroed rows carry -inf, so NaN has to compare equal here.
    torch.testing.assert_close(actual[1], expected[1], equal_nan=True)


def _case(bs, num_heads=16, v_head_dim=128, max_q=4, zero_frac=0.3, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    q_lens = torch.randint(
        1, max_q + 1, (bs,), dtype=torch.int32, device=cfg.device, generator=g
    )
    cum = torch.zeros(bs + 1, dtype=torch.int32, device=cfg.device)
    cum[1:] = torch.cumsum(q_lens, dim=0)
    total = int(cum[-1])
    kv_lens = torch.randint(
        1, 4096, (bs,), dtype=torch.int32, device=cfg.device, generator=g
    )
    mask = torch.rand(bs, device=cfg.device, generator=g) < zero_frac
    kv_lens[mask] = 0
    out = torch.randn(
        total, num_heads, v_head_dim, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    lse = torch.randn(total, num_heads, device=cfg.device, generator=g)
    return dict(
        out=out,
        lse=lse,
        kv_lens=kv_lens,
        cum_seq_lens=cum,
        max_seq_len=int(q_lens.max()),
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1, zero_frac=1.0),
    lambda: _case(8, seed=1),
    lambda: _case(64, seed=2),
    lambda: _case(257, max_q=8, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fixup_zero_kv
def test_fixup_zero_kv(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fixup_zero_kv")
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
        candidate = get_candidate("fixup_zero_kv", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/fixup_zero_kv ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/fixup_zero_kv not yet implemented")
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
