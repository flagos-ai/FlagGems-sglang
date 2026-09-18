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

"""Benchmark for fla/layernorm_gated."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("fla_layernorm_gated")

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
# Cases (from kernel-comp-baseline/problems/fla/layernorm_gated/cases.py)
# ---------------------------------------------------------------------------


def _case(
    t,
    d,
    activation="swish",
    is_rms_norm=True,
    has_bias=True,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    x = torch.randn(
        t, d, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)
    gate = torch.randn(
        t, d, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)
    weight = torch.randn(
        d, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)
    bias = None
    if has_bias:
        bias = torch.randn(
            d, generator=g, device=flaggems_sglang.device, dtype=torch.float32
        ).to(dtype)
    return dict(
        x=x,
        g=gate,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=1e-5,
        is_rms_norm=is_rms_norm,
    )


CORRECTNESS_CASES = [
    _case(1, 64),
    _case(37, 256, activation="sigmoid"),
    _case(83, 512, is_rms_norm=False, has_bias=False),
]

BENCH_CASES = [_case(t, 2048) for t in (1, 8, 64, 512, 4096)]

BENCH_IDS = [f"t{t}" for t in (1, 8, 64, 512, 4096)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.fla_layernorm_gated
def test_fla_layernorm_gated_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    case = case() if callable(case) else case
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.fla_layernorm_gated import fla_layernorm_gated
    except (ImportError, ModuleNotFoundError):
        pytest.skip("fla/layernorm_gated ops module not found")
        return

    try:
        fla_layernorm_gated(**kwargs)
    except NotImplementedError:
        pytest.skip("fla/layernorm_gated not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: fla_layernorm_gated(**kwargs))

    record_case("fla/layernorm_gated", BENCH_IDS[case_idx], ref_us, triton_us)
