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

"""Correctness test for moe/silu_and_mul_masked_post_quant."""

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


def _check_factory(masked_m):
    def _check(actual, expected):
        aq, asc = actual
        eq, esc = expected
        for e in range(masked_m.shape[0]):
            n = int(masked_m[e].item())
            if n <= 0:
                continue
            assert_close(asc[e, :n], esc[e, :n], dtype=torch.float32)
            ratio = aq.shape[-1] // asc.shape[-1]
            a_deq = aq[e, :n].to(torch.float32) * asc[e, :n].repeat_interleave(
                ratio, dim=-1
            )
            e_deq = eq[e, :n].to(torch.float32) * esc[e, :n].repeat_interleave(
                ratio, dim=-1
            )
            atol, rtol = 2e-2, 2e-2
            mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
            # A small n lets one misrounded element dominate the fraction,
            # so this needs more headroom than the flat 1e-2 used elsewhere.
            assert (
                mismatch.float().mean() < 2e-2
            ), "too many fp8 rounding-boundary mismatches"

    return _check


def _case(
    expert_num,
    token_num_padded,
    hidden_dim,
    group_size,
    counts,
    dtype=torch.bfloat16,
    seed=0,
):
    g = create_generator(cfg.device, seed=seed)
    inp = torch.randn(
        expert_num,
        token_num_padded,
        hidden_dim * 2,
        generator=g,
        device=cfg.device,
        dtype=torch.float32,
    ).to(dtype)
    masked_m = torch.tensor(counts, dtype=torch.int32, device=cfg.device)
    return dict(
        input=inp,
        quant_group_size=group_size,
        masked_m=masked_m,
        check=_check_factory(masked_m),
    )


_CASE_FACTORIES = [
    lambda: _case(4, 16, 32, 16, [0, 3, 16, 9]),
    lambda: _case(8, 128, 256, 128, [128, 0, 64, 1, 17, 128, 5, 90]),
] + [
    (lambda e: lambda: _case(e, 256, 4096, 128, [256] * e))(e)
    for e in (4, 8, 32)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.silu_and_mul_masked_post_quant
def test_silu_and_mul_masked_post_quant(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("silu_and_mul_masked_post_quant")
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
            "silu_and_mul_masked_post_quant", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/silu_and_mul_masked_post_quant ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/silu_and_mul_masked_post_quant not yet implemented")
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
