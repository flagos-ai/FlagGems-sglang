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

"""Correctness test for attention/pad_sequence_with_mask."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

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
    equal_nan = tol.pop("equal_nan", True)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=equal_nan,
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _check(actual, expected):
    assert actual[0] == expected[0]
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])


def _case(bs, max_len=64, hidden=4096, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    lengths = torch.randint(
        1,
        max_len + 1,
        (bs,),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    offsets = torch.zeros(bs, dtype=torch.int32, device=cfg.device)
    offsets[1:] = torch.cumsum(lengths, dim=0)[:-1]
    total = int(lengths.sum())
    emb = torch.randn(total, hidden, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    return dict(
        input_emb=emb,
        offsets=offsets,
        lengths=lengths,
        max_len=max_len,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1, max_len=8, hidden=512),
    lambda: _case(9, max_len=32, seed=1),
    lambda: _case(64, max_len=64, hidden=2048, seed=2),
    lambda: _case(33, max_len=128, hidden=1024, seed=3),
] + [
    (lambda bs: lambda: _case(bs, max_len=64, hidden=4096, seed=9))(bs)
    for bs in (1, 8, 64, 256)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.pad_sequence_with_mask
def test_pad_sequence_with_mask(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("pad_sequence_with_mask")
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
            "pad_sequence_with_mask", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/pad_sequence_with_mask ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/pad_sequence_with_mask not yet implemented")
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
