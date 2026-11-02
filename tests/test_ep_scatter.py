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

"""Correctness test for moe/ep_scatter."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

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

_BLOCK_E = 128  # per-expert token counts are padded to this
_QUANT_BLOCK = 128


def _case(t, topk, hidden, num_experts, seed=0, drop=0.0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    ids = torch.randint(
        0,
        num_experts,
        (t, topk),
        device=cfg.device,
        generator=g,
        dtype=torch.int32,
    )
    if drop:
        mask = torch.rand(ids.shape, device=cfg.device, generator=g) < drop
        ids = torch.where(mask, torch.full_like(ids, -1), ids)

    valid = torch.zeros(num_experts, dtype=torch.int32, device=cfg.device)
    for e in range(num_experts):
        valid[e] = int((ids == e).sum())
    # Each expert's slab is padded up to a multiple of BLOCK_E.
    padded = ((valid + _BLOCK_E - 1) // _BLOCK_E * _BLOCK_E).to(torch.int32)
    total = int(padded.sum())

    # fp8 keeps the payload bit-exact through the copy, which is all this
    # operator does to it; the scales carry the magnitudes.
    recv_x = torch.randn(
        t, hidden, dtype=torch.float32, device=cfg.device, generator=g
    ).to(torch.float8_e4m3fn)
    recv_x_scale = torch.rand(
        t,
        hidden // _QUANT_BLOCK,
        dtype=torch.float32,
        device=cfg.device,
        generator=g,
    )

    return dict(
        recv_x=recv_x.contiguous(),
        recv_x_scale=recv_x_scale.contiguous(),
        recv_topk=ids.contiguous(),
        num_recv_tokens_per_expert=padded,
        num_valid_tokens_per_expert=valid,
        expert_start_loc=torch.zeros(
            num_experts, dtype=torch.int32, device=cfg.device
        ),
        output_tensor=torch.zeros(
            (total, hidden), dtype=torch.float8_e4m3fn, device=cfg.device
        ),
        output_tensor_scale=torch.zeros(
            (total, hidden // _QUANT_BLOCK),
            dtype=torch.float32,
            device=cfg.device,
        ),
        m_indices=torch.zeros(total, dtype=torch.int32, device=cfg.device),
        output_index=torch.full(
            (t, topk), -1, dtype=torch.int32, device=cfg.device
        ),
        check=_make_check(ids, recv_x, recv_x_scale, padded, valid),
    )


def _make_check(ids, recv_x, recv_x_scale, padded, valid):
    """The kernel hands out destination rows with atomic_add, so the order
    inside one expert's slab is not part of the contract. Check the structure
    instead of the exact permutation."""

    def _check(actual, expected):
        a_loc, a_out, a_scale, a_m, a_idx = actual
        e_loc, _, _, e_m, _ = expected

        # Deterministic parts: the post-scatter cursor and the expert map.
        assert_close(a_loc, e_loc)
        assert_close(a_m, e_m)

        starts = torch.cumsum(padded.to(torch.int64), 0) - padded.to(
            torch.int64
        )
        flat_ids = ids.reshape(-1).to(torch.int64)
        flat_slots = a_idx.reshape(-1).to(torch.int64)
        src = torch.arange(ids.shape[0], device=ids.device).repeat_interleave(
            ids.shape[1]
        )

        live = flat_ids >= 0
        # 1. Every dispatched slot lands inside its expert's valid range.
        lo = starts[flat_ids[live]]
        hi = lo + valid.to(torch.int64)[flat_ids[live]]
        assert bool(
            ((flat_slots[live] >= lo) & (flat_slots[live] < hi)).all()
        ), "a slot landed outside its expert's valid range"
        # 2. No two dispatched slots share a destination row.
        assert flat_slots[live].unique().numel() == int(
            live.sum()
        ), "destination rows are not unique"
        # 3. The row actually written matches that token's payload and scale.
        assert_close(
            a_out[flat_slots[live]].to(torch.float32),
            recv_x[src[live]].to(torch.float32),
            dtype=torch.float32,
        )
        assert_close(
            a_scale[flat_slots[live]],
            recv_x_scale[src[live]],
            dtype=torch.float32,
        )
        # 4. Undispatched slots keep the sentinel.
        assert bool(
            (flat_slots[~live] == -1).all()
        ), "an undispatched slot was written"

    return _check


_CASE_FACTORIES = [
    lambda: _case(64, 4, 512, num_experts=4),
    lambda: _case(512, 8, 1024, num_experts=8, drop=0.25),
    lambda: _case(333, 2, 512, num_experts=16),
] + [
    (lambda t: lambda: _case(t, 8, 2048, num_experts=8))(t)
    for t in (64, 512, 4096)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.ep_scatter
def test_ep_scatter(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("ep_scatter")
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
        from flaggems_sglang import ep_scatter
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/ep_scatter ops module not found")
        return

    try:
        actual = ep_scatter(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/ep_scatter not yet implemented")
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
