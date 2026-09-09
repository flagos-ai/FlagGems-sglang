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

"""Benchmark for rope/interleaved_rope."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("interleaved_rope")

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
# Cases (from kernel-comp-baseline/problems/rope/interleaved_rope/cases.py)
# ---------------------------------------------------------------------------


def _x(s, d, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    return torch.randn(
        3,
        s,
        d,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)


def _case(s, d, mrope_section):
    return dict(x=_x(s, d), mrope_section=mrope_section)


CORRECTNESS_CASES = [
    _case(1, 48, [8, 8, 8]),
    _case(37, 96, [16, 16, 16]),
    _case(257, 128, [16, 24, 24]),
]

BENCH_CASES = [_case(s, 128, [16, 24, 24]) for s in (1, 128, 2048, 8192)]

BENCH_IDS = [f"s{s}" for s in (1, 128, 2048, 8192)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.interleaved_rope
def test_interleaved_rope_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = case

    try:
        from flaggems_sglang.ops.interleaved_rope import interleaved_rope
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rope/interleaved_rope ops module not found")
        return

    try:
        interleaved_rope(**kwargs)
    except NotImplementedError:
        pytest.skip("rope/interleaved_rope not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: interleaved_rope(**kwargs))

    record_case(
        "rope/interleaved_rope", BENCH_IDS[case_idx], ref_us, triton_us
    )
