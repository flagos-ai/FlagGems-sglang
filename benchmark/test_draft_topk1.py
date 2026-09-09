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

"""Benchmark for speculative/draft_topk1."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("draft_topk1")

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
# Cases (from kernel-comp-baseline/problems/speculative/draft_topk1/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_p, a_idx, a_pos, a_dt = actual
    e_p, e_idx, e_pos, e_dt = expected
    assert_close(a_p, e_p, dtype=torch.float32)
    assert torch.equal(a_idx, e_idx), "argmax index mismatch"
    assert torch.equal(a_pos, e_pos), "positions mismatch"
    if a_dt is not None:
        assert torch.equal(a_dt, e_dt), "draft_tokens mismatch"


def _case(
    bs,
    vocab_size,
    with_draft_tokens=False,
    draft_token_column=0,
    num_draft_cols=4,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    next_token_logits = torch.randn(
        bs,
        vocab_size,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    )
    positions = torch.randint(
        0,
        4096,
        (bs,),
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.int64,
    )

    draft_tokens = None
    if with_draft_tokens:
        draft_tokens = torch.randint(
            0,
            vocab_size,
            (bs, num_draft_cols),
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.int64,
        )

    return dict(
        next_token_logits=next_token_logits,
        positions=positions,
        draft_tokens=draft_tokens,
        draft_token_column=draft_token_column,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case(1, 32000),
    _case(37, 12000, with_draft_tokens=True, draft_token_column=2),
    _case(129, 8192 + 500),
    _case(8, 151936, with_draft_tokens=True, draft_token_column=0),
]

BENCH_CASES = [
    _case(bs, 151936, with_draft_tokens=True) for bs in (1, 8, 64, 512)
]

BENCH_IDS = [f"bs{bs}" for bs in (1, 8, 64, 512)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.draft_topk1
def test_draft_topk1_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.draft_topk1 import draft_topk1
    except (ImportError, ModuleNotFoundError):
        pytest.skip("speculative/draft_topk1 ops module not found")
        return

    try:
        draft_topk1(**kwargs)
    except NotImplementedError:
        pytest.skip("speculative/draft_topk1 not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: draft_topk1(**kwargs))

    record_case(
        "speculative/draft_topk1", BENCH_IDS[case_idx], ref_us, triton_us
    )
