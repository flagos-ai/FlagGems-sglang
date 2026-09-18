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

"""Benchmark for activation_norm/l2norm."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("l2norm")

device = flaggems_sglang.device


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
# Cases (from kernel-comp-baseline/problems/activation_norm/l2norm/cases.py)
# ---------------------------------------------------------------------------


def _x(t, d, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    return torch.randn(
        t, d, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)


def _case(t, d):
    return dict(x=_x(t, d))


CORRECTNESS_CASES = [
    _case(1, 64),
    _case(83, 128),
    _case(256, 256),
    _case(3, 1024),
]

BENCH_CASES = [_case(t, d) for t in (1, 32, 512, 4096) for d in (64, 128, 256)]

BENCH_IDS = [f"t{t}_d{d}" for t in (1, 32, 512, 4096) for d in (64, 128, 256)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.l2norm
def test_l2norm_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    case = case() if callable(case) else case
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.l2norm import l2norm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("activation_norm/l2norm ops module not found")
        return

    try:
        l2norm(**kwargs)
    except NotImplementedError:
        pytest.skip("activation_norm/l2norm not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: l2norm(**kwargs))

    record_case(
        "activation_norm/l2norm", BENCH_IDS[case_idx], ref_us, triton_us
    )
