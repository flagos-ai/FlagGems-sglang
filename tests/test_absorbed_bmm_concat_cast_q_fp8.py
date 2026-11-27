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

"""Correctness test for kvcache/absorbed_bmm_concat_cast_q_fp8."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg
from flaggems_sglang.utils.rng_utils import create_generator

# torch_gcu aborts the whole process when casting to these dtypes
# on-device (backend CHECK failure); skip every test in this module.
pytestmark = cfg.cast_skip_mark(torch.float8_e4m3fn)




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


def _make_check(tokens, heads):
    def _check(actual, expected):
        # The rope half is a pure cast: exact.
        assert_close(
            actual[:tokens, :heads, 512:].view(torch.uint8),
            expected[:tokens, :heads, 512:].view(torch.uint8),
        )
        # The nope half comes from a GEMM whose fp32 accumulation order
        # differs from an einsum's, so compare values rather than codes and
        # allow the occasional element to land on the adjacent fp8 codeword.
        a = actual[:tokens, :heads, :512].to(torch.float32)
        e = expected[:tokens, :heads, :512].to(torch.float32)
        atol, rtol = 6e-2, 6e-2
        mismatch = (a - e).abs() > (atol + rtol * e.abs())
        assert (
            mismatch.float().mean() < 2e-2
        ), "too many fp8 boundary mismatches"
        # Padding rows and heads must be left alone.
        if actual.shape[1] > heads:
            assert_close(
                actual[:, heads:].view(torch.uint8),
                expected[:, heads:].view(torch.uint8),
            )

    return _check


def _case(tokens, heads, k, n, rope, pad_heads=None, pad_tokens=None, seed=0):
    g = create_generator(cfg.device, seed=seed)

    def _t(*shape):
        return (
            0.25
            * torch.randn(
                *shape,
                dtype=torch.float32,
                device=cfg.device,
                generator=g,
            )
        ).to(torch.bfloat16)

    pad_heads = pad_heads or heads
    pad_tokens = pad_tokens or tokens
    # A dirty destination: the kernel must not clear the padding.
    q_fp8_pad = torch.full(
        (pad_tokens, pad_heads, n + rope),
        3.0,
        dtype=torch.float32,
        device=cfg.device,
    ).to(torch.float8_e4m3fn)
    return dict(
        q_fp8_pad=q_fp8_pad,
        q_nope=_t(tokens, heads, k).contiguous(),
        w_kc=_t(heads, k, n).contiguous(),
        q_rope=_t(tokens, heads, rope).contiguous(),
        num_heads=heads,
        check=_make_check(tokens, heads),
    )


_CASE_FACTORIES = [
    lambda: _case(16, 8, 128, 512, 64),
    lambda: _case(512, 16, 128, 512, 64),
    lambda: _case(64, 8, 128, 512, 64, pad_heads=16, pad_tokens=128),
] + [
    (lambda t: lambda: _case(t, 16, 128, 512, 64))(t)
    for t in (16, 128, 1024, 4096)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.absorbed_bmm_concat_cast_q_fp8
def test_absorbed_bmm_concat_cast_q_fp8(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("absorbed_bmm_concat_cast_q_fp8")
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
            "absorbed_bmm_concat_cast_q_fp8", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "kvcache/absorbed_bmm_concat_cast_q_fp8 ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "kvcache/absorbed_bmm_concat_cast_q_fp8 not yet implemented"
        )
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
