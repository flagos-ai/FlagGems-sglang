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

"""Benchmark for attention/context_attention."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("context_attention")


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
# Cases (from kernel-comp-baseline/problems/attention/context_attention/cases.py)
# ---------------------------------------------------------------------------


def _case(
    seq_lens, num_heads, head_dim, is_causal, dtype=torch.float32, seed=0
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    total = sum(seq_lens)
    q = torch.randn(
        total,
        num_heads,
        head_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=dtype,
    )
    k = torch.randn(
        total,
        num_heads,
        head_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=dtype,
    )
    v = torch.randn(
        total,
        num_heads,
        head_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=dtype,
    )

    b_start_loc = torch.zeros(
        len(seq_lens), dtype=torch.int32, device=flaggems_sglang.device
    )
    b_start_loc[1:] = torch.cumsum(
        torch.tensor(
            seq_lens[:-1], dtype=torch.int32, device=flaggems_sglang.device
        ),
        0,
    )
    b_seq_len = torch.tensor(
        seq_lens, dtype=torch.int32, device=flaggems_sglang.device
    )
    max_input_len = max(seq_lens)

    return dict(
        q=q,
        k=k,
        v=v,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=max_input_len,
        is_causal=is_causal,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual.to(torch.float32), expected, atol=1e-2, rtol=1e-2)


CORRECTNESS_CASES = [
    _case([8, 12], 4, 128, True),
    _case([8, 12], 4, 128, False),
    _case([5, 30, 7], 4, 96, True),
    _case([20], 4, 80, True),
    _case([9], 4, 13, True),
]

BENCH_CASES = [
    _case([seq_len] * bs, 32, 128, True)
    for bs in (1, 8, 64)
    for seq_len in (128, 2048)
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.context_attention
def test_context_attention_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.context_attention import context_attention
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/context_attention ops module not found")
        return

    try:
        context_attention(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/context_attention not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: context_attention(**kwargs))
    record_case(
        "attention/context_attention", f"case{case_idx}", ref_us, triton_us
    )
