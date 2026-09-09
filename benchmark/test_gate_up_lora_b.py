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

"""Benchmark for lora/gate_up_lora_b."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("gate_up_lora_b")

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
        equal_nan=True,
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/lora/gate_up_lora_b/cases.py)
# ---------------------------------------------------------------------------


from flaggems_sglang.reference._lora_batch_utils import make_batch_info


def _case(
    seg_lens,
    num_lora,
    r,
    output_dim,
    permutation="none",
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    s = sum(seg_lens)

    x = torch.randn(
        s,
        2 * r,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    gate_up_lora_b = torch.randn(
        num_lora,
        2 * output_dim,
        r,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    base_output = torch.randn(
        s,
        2 * output_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens,
        weight_indices,
        lora_ranks=[r] * num_lora,
        scalings=scalings,
        permutation=permutation,
    )
    return dict(
        x=x,
        gate_up_lora_b=gate_up_lora_b,
        batch_info=batch_info,
        output_dim=output_dim,
        base_output=base_output,
    )


CORRECTNESS_CASES = [
    _case([5], 1, 16, 64),
    _case([3, 7, 0, 12], 2, 16, 128),
    _case([9, 4], 2, 32, 256, permutation="shuffled"),
]

BENCH_CASES = [
    _case([64] * 8, 4, 32, 4096),
    _case([256] * 4, 2, 64, 4096),
]

BENCH_IDS = [
    "seg64x8_nlora4_r32_out4096",
    "seg256x4_nlora2_r64_out4096",
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.gate_up_lora_b
def test_gate_up_lora_b_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = case

    try:
        from flaggems_sglang.ops.gate_up_lora_b import gate_up_lora_b
    except (ImportError, ModuleNotFoundError):
        pytest.skip("lora/gate_up_lora_b ops module not found")
        return

    try:
        gate_up_lora_b(**kwargs)
    except NotImplementedError:
        pytest.skip("lora/gate_up_lora_b not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: gate_up_lora_b(**kwargs))

    record_case("lora/gate_up_lora_b", BENCH_IDS[case_idx], ref_us, triton_us)
