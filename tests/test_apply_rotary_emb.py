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

"""Correctness test for rope/apply_rotary_emb."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg
from flaggems_sglang.utils.rng_utils import create_generator

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


def _case(
    tokens,
    heads,
    rope_dim,
    dtype=torch.bfloat16,
    seed=0,
    use_pos=False,
    inverse=False,
):
    g = create_generator(cfg.device, seed=seed)
    shape = (tokens, rope_dim) if heads is None else (tokens, heads, rope_dim)
    x = torch.randn(
        *shape, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    rows = 4096 if use_pos else tokens
    angle = (
        torch.rand(rows, rope_dim // 2, device=cfg.device, generator=g)
        * 6.283185307
    )
    freqs_cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
    positions = (
        torch.randint(
            0,
            rows,
            (tokens,),
            device=cfg.device,
            generator=g,
            dtype=torch.int32,
        )
        if use_pos
        else None
    )
    return dict(
        x=x.contiguous(),
        freqs_cis=freqs_cis,
        positions=positions,
        inverse=inverse,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(7, 8, 64),
    lambda: _case(1024, 16, 128),
    lambda: _case(333, None, 64),
    lambda: _case(512, 8, 64, use_pos=True),
    lambda: _case(512, 8, 64, inverse=True),
] + [
    (lambda t: lambda: _case(t, 16, 64))(t)
    for t in (1, 8, 64, 512, 4096, 16384)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.apply_rotary_emb
def test_apply_rotary_emb(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("apply_rotary_emb")
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
        candidate = get_candidate("apply_rotary_emb", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rope/apply_rotary_emb ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("rope/apply_rotary_emb not yet implemented")
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
