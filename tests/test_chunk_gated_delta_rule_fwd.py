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

"""Correctness test for linear-attn/chunk_gated_delta_rule_fwd."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

# ---------------------------------------------------------------------------
# Tolerance helper
# ---------------------------------------------------------------------------

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=2e-2, rtol=2e-2)


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
# Cases
# ---------------------------------------------------------------------------


def _check(actual, expected):
    act_o = actual[0] if isinstance(actual, (tuple, list)) else actual
    exp_o = expected[0] if isinstance(expected, (tuple, list)) else expected
    assert_close(act_o, exp_o)
    if isinstance(expected, (tuple, list)) and len(expected) > 1:
        exp_s = expected[1]
        act_s = actual[1]
        if isinstance(exp_s, torch.Tensor):
            # final_state is fp32 but accumulated from bf16 inputs, so hold it
            # to the bf16 tolerance rather than the fp32 default.
            assert_close(act_s, exp_s, atol=2e-2, rtol=2e-2)


def _case(
    B,
    T,
    H,
    K,
    V,
    use_qk_l2norm_in_kernel=True,
    output_final_state=False,
    seed=0,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    q = torch.randn(B, T, H, K, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    k = torch.randn(B, T, H, K, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    v = torch.randn(B, T, H, V, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    # g is logsigmoid(u), matching the operator's own data generation, so the
    # decay horizon segmentation the kernel relies on stays valid.
    u = torch.rand(B, T, H, device=cfg.device, generator=g)
    gate = torch.nn.functional.logsigmoid(u)
    beta = torch.rand(B, T, H, device=cfg.device, generator=g)
    scale = K**-0.5
    return dict(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=output_final_state,
        cu_seqlens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 64, 4, 128, 128),
    lambda: _case(2, 128, 4, 128, 128, seed=1),
    lambda: _case(2, 256, 2, 128, 128, output_final_state=True, seed=2),
    lambda: _case(1, 512, 8, 64, 64, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.chunk_gated_delta_rule_fwd
def test_chunk_gated_delta_rule_fwd(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("chunk_gated_delta_rule_fwd")
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
            "chunk_gated_delta_rule_fwd", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "linear-attn/chunk_gated_delta_rule_fwd ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "linear-attn/chunk_gated_delta_rule_fwd not yet implemented"
        )
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
