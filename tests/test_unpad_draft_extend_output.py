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

"""Correctness test for attention/unpad_draft_extend_output."""

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
# Cases (from kernel-comp-baseline/problems/attention/
# unpad_draft_extend_output/cases.py)
# ---------------------------------------------------------------------------


def _case(bs, token_per_batch=8, num_heads=16, head_dim=128, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    seq_lens = torch.randint(
        1,
        token_per_batch + 1,
        (bs,),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=cfg.device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    raw_out = torch.randn(
        bs,
        token_per_batch,
        num_heads,
        head_dim,
        device=cfg.device,
        generator=g,
    ).to(torch.bfloat16)
    return dict(
        raw_out=raw_out,
        cu_seqlens_q=cu,
        seq_lens_q=seq_lens,
        sum_seq_lens_q=int(cu[-1]),
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(9, seed=1),
    lambda: _case(64, token_per_batch=4, seed=2),
    lambda: _case(33, token_per_batch=16, num_heads=8, head_dim=64, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.unpad_draft_extend_output
def test_unpad_draft_extend_output(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("unpad_draft_extend_output")
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
        candidate = get_candidate(
            "unpad_draft_extend_output", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/unpad_draft_extend_output ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/unpad_draft_extend_output not yet implemented")
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
