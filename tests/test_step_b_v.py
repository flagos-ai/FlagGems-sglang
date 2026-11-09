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

"""Correctness test for lora/step_b_v."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from . import conftest as cfg

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


def _case(
    seg_lens,
    num_lora,
    h,
    rank,
    qk_nope_head_dim,
    v_head_dim,
    permutation="none",
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sum(seg_lens)
    full_k = qk_nope_head_dim + v_head_dim

    attn_lora_a = torch.randn(
        s, h, rank, generator=g, device=cfg.device, dtype=torch.float32
    ).to(dtype)
    b_buf = torch.randn(
        num_lora,
        h * full_k,
        rank,
        generator=g,
        device=cfg.device,
        dtype=torch.float32,
    ).to(dtype)
    base_output = torch.randn(
        s, h, v_head_dim, generator=g, device=cfg.device, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens,
        weight_indices,
        lora_ranks=[rank] * num_lora,
        scalings=scalings,
        permutation=permutation,
    )
    return dict(
        attn_lora_a=attn_lora_a,
        B_buf=b_buf,
        batch_info=batch_info,
        base_output=base_output,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
    )


_CASE_FACTORIES = [
    lambda: _case([5], 1, 4, 16, 128, 128),
    lambda: _case([3, 7, 0, 12], 2, 8, 32, 128, 128),
    lambda: _case([9, 4], 2, 16, 16, 64, 128, permutation="shuffled"),
    lambda: _case([64] * 8, 4, 128, 32, 128, 128),
    lambda: _case([256] * 4, 2, 128, 16, 64, 128),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.step_b_v
def test_step_b_v(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("step_b_v")
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
        from flaggems_sglang import step_b_v
    except (ImportError, ModuleNotFoundError):
        pytest.skip("lora/step_b_v ops module not found")
        return

    try:
        actual = step_b_v(**kwargs)
    except NotImplementedError:
        pytest.skip("lora/step_b_v not yet implemented")
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
