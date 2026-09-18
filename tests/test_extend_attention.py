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

"""Correctness test for attention/extend_attention."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("extend_attention")


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
# Cases (from kernel-comp-baseline/problems/attention/extend_attention/cases.py)
# ---------------------------------------------------------------------------


def _case(B, n_ctx, H_Q, H_KV, D, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)

    def rnd_int(lo, hi, n):
        return torch.randint(
            lo,
            hi,
            (n,),
            dtype=torch.int32,
            device=flaggems_sglang.device,
            generator=g,
        )

    b_seq_len_prefix = rnd_int(1, max(2, n_ctx // 2), B)
    b_seq_len_extend = rnd_int(1, max(2, n_ctx // 2), B)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros(
        (B,), dtype=torch.int32, device=flaggems_sglang.device
    )
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros(
        (B,), dtype=torch.int32, device=flaggems_sglang.device
    )
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros(
        (B + 1,), dtype=torch.int32, device=flaggems_sglang.device
    )
    kv_indptr[1:] = torch.cumsum(b_seq_len_prefix, 0)
    kv_indices = torch.zeros(
        (int(b_seq_len_prefix.sum()),),
        dtype=torch.int32,
        device=flaggems_sglang.device,
    )
    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i].item(),
            (b_start_loc[i] + b_seq_len_prefix[i]).item(),
            device=flaggems_sglang.device,
        )

    total_token_num = int(b_seq_len.sum())
    extend_token_num = int(b_seq_len_extend.sum())
    k_buffer = torch.randn(
        total_token_num,
        H_KV,
        D,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    v_buffer = torch.randn(
        total_token_num,
        H_KV,
        D,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)

    k_extend = torch.empty(
        (extend_token_num, H_KV, D), dtype=dtype, device=flaggems_sglang.device
    )
    v_extend = torch.empty(
        (extend_token_num, H_KV, D), dtype=dtype, device=flaggems_sglang.device
    )
    q_extend = torch.empty(
        (extend_token_num, H_Q, D), dtype=dtype, device=flaggems_sglang.device
    )
    for i in range(B):
        eib = (b_start_loc[i] + b_seq_len_prefix[i]).item()
        eie = (b_start_loc[i] + b_seq_len[i]).item()
        es = b_start_loc_extend[i].item()
        ee = (b_start_loc_extend[i] + b_seq_len_extend[i]).item()
        k_extend[es:ee] = k_buffer[eib:eie]
        v_extend[es:ee] = v_buffer[eib:eie]
        q_extend[es:ee] = torch.randn(
            (ee - es, H_Q, D),
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        ).to(dtype)

    qo_indptr = torch.zeros(
        (B + 1,), dtype=torch.int32, device=flaggems_sglang.device
    )
    qo_indptr[1:] = torch.cumsum(b_seq_len_extend, 0)
    max_len_extend = int(b_seq_len_extend.max())

    return dict(
        q_extend=q_extend,
        k_extend=k_extend,
        v_extend=v_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        max_len_extend=max_len_extend,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual.to(torch.float32), expected, atol=1e-2, rtol=1e-2)


CORRECTNESS_CASES = [
    _case(4, 256, 12, 4, 128),
    _case(4, 256, 12, 4, 80),
    _case(2, 128, 8, 8, 64),
]

# Lazy (zero-arg callables): total token count scales with B * n_ctx, so
# cases are generated one at a time by harness.bench.run_bench_cases rather
# than all held in memory at once.
BENCH_CASES = [
    (lambda B=B, n_ctx=n_ctx: _case(B, n_ctx, 32, 8, 128))
    for B, n_ctx in ((1, 2048), (8, 2048), (64, 512), (256, 256))
]
CORRECTNESS_CASES = CORRECTNESS_CASES + [c() for c in BENCH_CASES]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.extend_attention
def test_extend_attention(case_idx):
    case = CORRECTNESS_CASES[case_idx]
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
        from flaggems_sglang import extend_attention
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/extend_attention ops module not found")
        return

    try:
        actual = extend_attention(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/extend_attention not yet implemented")
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
