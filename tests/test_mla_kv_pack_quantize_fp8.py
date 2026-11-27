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

"""Correctness test for attention/mla_kv_pack_quantize_fp8."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.utils.rng_utils import create_generator

from . import conftest as cfg

# torch_gcu aborts the whole process when casting to these dtypes
# on-device (backend CHECK failure); skip every test in this module.
pytestmark = cfg.cast_skip_mark(torch.float8_e4m3fn, torch.float8_e5m2)


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
    for a, e in zip(actual, expected):
        assert_close(a.view(torch.uint8), e.view(torch.uint8))


def _case(
    s,
    heads,
    qk_nope=512,
    qk_rope=64,
    v_head=512,
    dtype=torch.bfloat16,
    seed=0,
    pe_3d=False,
    fp8_dtype=torch.float8_e4m3fn,
    scales=(1.0, 1.0),
):
    g = create_generator(cfg.device, seed=seed)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)

    k_pe = _t(s, 1, qk_rope) if pe_3d else _t(s, qk_rope)
    return dict(
        k_nope=_t(s, heads, qk_nope).contiguous(),
        k_pe=k_pe.contiguous(),
        v=_t(s, heads, v_head).contiguous(),
        k_scale_inv=scales[0],
        v_scale_inv=scales[1],
        fp8_dtype=fp8_dtype,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(7, 8),
    # Crosses the baseline's v0/v1 kernel pick, which keys off (s, heads).
    lambda: _case(1024, 16),
    lambda: _case(333, 1, pe_3d=True),
    lambda: _case(256, 8, scales=(0.5, 2.0)),
    lambda: _case(128, 4, fp8_dtype=torch.float8_e5m2),
] + [
    (lambda s, h: lambda: _case(s, h))(s, h)
    for s in (1, 8, 64, 512, 4096)
    for h in (1, 16)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.mla_kv_pack_quantize_fp8
def test_mla_kv_pack_quantize_fp8(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("mla_kv_pack_quantize_fp8")
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
            "mla_kv_pack_quantize_fp8", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/mla_kv_pack_quantize_fp8 ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/mla_kv_pack_quantize_fp8 not yet implemented")
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
