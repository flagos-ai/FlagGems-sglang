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

"""Check that swiglu_limit actually clips, which the official cases do not.

``tests/test_act_and_mul.py`` (the task's own harness) uses ``swiglu_limit=7.0``
and ``10.0``, while the largest ``|up|`` either case produces is 3.500 and
5.312. The clamp never executes there, so a kernel that ignored the limit
entirely would still pass both official cases.

These inputs drive ``up`` past both ends of the bound and ``gate`` past its
one-sided one. Besides matching the reference, the cases assert that the
unclamped expression gives a *different* answer: requiring disagreement is
what makes a dropped clamp fail here instead of passing quietly.
"""

import pytest
import torch
import torch.nn.functional as F

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype,
            dict(atol=1e-2, rtol=1e-2),
        )
    )
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32), expected.to(torch.float32), **tol
    )


def _act(x, kind):
    if kind == "silu":
        return F.silu(x)
    if kind == "gelu":
        return F.gelu(x, approximate="tanh")
    raise ValueError(kind)


def _split(gateup):
    half = gateup.shape[1] // 2
    return gateup[:, :half].float(), gateup[:, half:].float()


def _clamped(gateup, kind, limit):
    """The documented math: gate clipped at ``limit``, up clipped at both ends."""
    gate, up = _split(gateup)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return _act(gate, kind) * up


def _unclamped(gateup, kind, limit):
    """What a kernel that forgot ``swiglu_limit`` returns."""
    gate, up = _split(gateup)
    return _act(gate, kind) * up


def _past_limit(limit, dtype, seed=7):
    """Build a ``[64, 256]`` input with both ends of both bounds exceeded."""
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(seed)
    rows, half = 64, 128
    # gate spans -1..6 x limit so only its positive side needs clipping.
    gate = (
        torch.rand(rows, half, generator=g, device=dev) * 7.0 - 1.0
    ) * limit
    # up spans -4..4 x limit, so both ends of its bound need clipping.
    up = (torch.rand(rows, half, generator=g, device=dev) * 8.0 - 4.0) * limit
    return torch.cat([gate, up], dim=1).to(dtype)


@pytest.mark.parametrize("kind", ["silu", "gelu"])
@pytest.mark.parametrize("limit", [1.0, 0.5])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16]
)
@pytest.mark.act_and_mul
def test_swiglu_limit_clips(kind, limit, dtype):
    """The clamp must execute, match the math, and differ from no clamping."""
    gateup = _past_limit(limit, dtype)

    candidate = get_candidate("act_and_mul")
    actual = candidate(gateup, kind, limit)

    expected = get_reference("act_and_mul")(gateup, kind, limit)
    assert_close(actual, expected, dtype=dtype)

    # The assertion that matters: a kernel ignoring the limit would agree with
    # the unclamped expression instead of this one.
    unclamped = _unclamped(gateup, kind, limit).to(dtype)
    moved = (actual.to(torch.float32) - unclamped.to(torch.float32)).abs()
    assert moved.max().item() > 0.5, (
        "clamped output matches the unclamped expression, so the limit was "
        f"not applied (max delta {moved.max().item():.6f})"
    )


@pytest.mark.parametrize("kind", ["silu", "gelu"])
@pytest.mark.act_and_mul
def test_clamp_is_asymmetric(kind):
    """``gate`` clips only at ``+limit``; ``up`` clips at both ends."""
    dev = flaggems_sglang.device
    limit = 1.0
    # gate (first half) and up (second half) share these two columns.
    gate = torch.tensor([[-4.0, 4.0]], device=dev, dtype=torch.float32)
    up = torch.tensor([[-4.0, 4.0]], device=dev, dtype=torch.float32)
    gateup = torch.cat([gate, up], dim=1)

    candidate = get_candidate("act_and_mul")
    actual = candidate(gateup, kind, limit)

    expected = _clamped(gateup, kind, limit)
    assert_close(actual, expected, dtype=torch.float32)

    # gate = -4 must survive: only its positive side is bounded. Clipping it
    # at -limit would move the gate to -1, whose tanh-GELU differs from the
    # x=-4 value by orders of magnitude. Comparing the two distances proves
    # which one the kernel used without demanding an exact value, because the
    # tanh approximation itself differs slightly between Triton and torch.
    gate_kept = _act(torch.tensor([[-4.0]], device=dev), kind).item()
    gate_clipped = _act(torch.tensor([[-1.0]], device=dev), kind).item()
    value = actual[0, 0].item()
    assert abs(value - gate_kept * -1.0) < abs(value - gate_clipped * -1.0), (
        "the gate half was clipped at -limit, but swiglu_limit only bounds "
        "the gate from above"
    )

    # up = +4 must be clamped to +limit before multiplying. Same reasoning:
    # the clamped product sits next to activation(1)*1, and the unclamped one
    # next to activation(1)*4, which is four times larger.
    gate_at_limit = _act(torch.tensor([[1.0]], device=dev), kind).item()
    value_pos = actual[0, 1].item()
    assert abs(value_pos - gate_at_limit * 1.0) < abs(
        value_pos - gate_at_limit * 4.0
    ), "the up half was not clamped to +limit before multiplying"
