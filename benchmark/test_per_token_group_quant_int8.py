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

"""Benchmark for quantization/per_token_group_quant_int8."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("per_token_group_quant_int8")

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
# Cases (from kernel-comp-baseline/problems/quantization/per_token_group_quant_int8/cases.py)
# ---------------------------------------------------------------------------


def _x(m, k, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    return (
        (
            torch.rand(
                m,
                k,
                generator=g,
                device=flaggems_sglang.device,
                dtype=torch.float32,
            )
            * 2
            - 1
        )
        .to(dtype)
        .contiguous()
    )


def _check(actual, expected):
    aq, asc = actual
    eq, esc = expected
    assert_close(asc, esc, dtype=torch.float32)
    a_deq = aq.to(torch.float32) * asc.repeat_interleave(
        aq.shape[-1] // asc.shape[-1], dim=-1
    )
    e_deq = eq.to(torch.float32) * esc.repeat_interleave(
        eq.shape[-1] // esc.shape[-1], dim=-1
    )
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert (
        mismatch.float().mean() < 1e-2
    ), "too many int8 rounding-boundary mismatches"


CORRECTNESS_CASES = [
    dict(x=_x(7, 128), group_size=128, check=_check),
    dict(x=_x(83, 512), group_size=128, check=_check),
    dict(x=_x(256, 4096), group_size=128, check=_check),
    dict(x=_x(3, 256), group_size=64, check=_check),
]

BENCH_CASES = [
    dict(x=_x(m, k), group_size=128, check=_check)
    for m in (1, 8, 64, 512, 4096)
    for k in (2048, 4096, 8192)
]

BENCH_IDS = [
    f"m{m}_k{k}" for m in (1, 8, 64, 512, 4096) for k in (2048, 4096, 8192)
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.per_token_group_quant_int8
def test_per_token_group_quant_int8_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.per_token_group_quant_int8 import (
            per_token_group_quant_int8,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "quantization/per_token_group_quant_int8 ops module not found"
        )
        return

    try:
        per_token_group_quant_int8(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "quantization/per_token_group_quant_int8 not yet implemented"
        )
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: per_token_group_quant_int8(**kwargs))

    record_case(
        "quantization/per_token_group_quant_int8",
        BENCH_IDS[case_idx],
        ref_us,
        triton_us,
    )
