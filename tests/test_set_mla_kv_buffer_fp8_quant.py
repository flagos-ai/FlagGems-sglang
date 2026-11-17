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

"""Correctness test for kvcache/set_mla_kv_buffer_fp8_quant."""

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


_NOPE = 512
_ROPE = 64


def _check(actual, expected):
    assert_close(actual.view(torch.uint8), expected.view(torch.uint8))


def _case(tokens, slots, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)

    # The buffer is allocated in the source dtype and viewed as fp8 inside
    # the kernel, so its last dim is half as wide in elements.
    kv_buffer = _t(slots, (_NOPE + _ROPE) // 2).contiguous()
    # loc values are distinct; duplicates would make the result depend on
    # write order.
    loc = torch.randperm(slots, generator=g, device=cfg.device)[:tokens].to(
        torch.int64
    )
    return dict(
        kv_buffer=kv_buffer,
        loc=loc,
        cache_k_nope=_t(tokens, _NOPE).contiguous(),
        cache_k_rope=_t(tokens, _ROPE).contiguous(),
        fp8_dtype=torch.float8_e4m3fn,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(16, 256),
    lambda: _case(1024, 4096),
    lambda: _case(333, 1024),
] + [
    (lambda t: lambda: _case(t, max(t * 2, 64)))(t)
    for t in (16, 128, 1024, 4096, 16384)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_set_mla_kv_buffer_fp8_quant(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("set_mla_kv_buffer_fp8_quant")
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
            "set_mla_kv_buffer_fp8_quant", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("kvcache/set_mla_kv_buffer_fp8_quant ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("kvcache/set_mla_kv_buffer_fp8_quant not yet implemented")
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
