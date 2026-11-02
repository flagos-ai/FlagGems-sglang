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

"""Correctness test for kvcache/set_mla_kv_buffer."""

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
# kernel-comp-baseline/problems/kvcache/set_mla_kv_buffer/cases.py)
# ---------------------------------------------------------------------------


def _case(n_loc, num_slots=65536, nope_dim=512, rope_dim=64, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    kv_buffer = torch.zeros(
        num_slots,
        nope_dim + rope_dim,
        dtype=torch.bfloat16,
        device=cfg.device,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    loc = (
        torch.randperm(
            num_slots,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )[:n_loc]
        .to(torch.int64)
        .to(cfg.device)
    )
    nope = torch.randn(n_loc, nope_dim, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    rope = torch.randn(n_loc, rope_dim, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    return dict(
        kv_buffer=kv_buffer,
        loc=loc,
        cache_k_nope=nope,
        cache_k_rope=rope,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(37, seed=1),
    lambda: _case(512, seed=2),
    lambda: _case(4096, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.set_mla_kv_buffer
def test_set_mla_kv_buffer(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("set_mla_kv_buffer")
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
        candidate = get_candidate("set_mla_kv_buffer", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("kvcache/set_mla_kv_buffer ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("kvcache/set_mla_kv_buffer not yet implemented")
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
