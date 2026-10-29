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

import pytest
import torch
import triton

pytest.importorskip("triton.experimental.tle.language")
from flaggems_sglang.runtime.backend._nvidia.hopper.ops import (  # noqa: E402
    _dsv41_fp4_tle_indexer as native,
)
from flaggems_sglang.runtime.backend._nvidia.hopper.ops import (  # noqa: E402
    dsv41_fp4_indexer as control,
)

from .dsv41_helpers import assert_exact, fp4_inputs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0)
    or triton.__version__.split("+")[0] != "3.6.0",
    reason="SM90 packed serial qualification for FlagTree 3.6.0",
)


class _Capture:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.args = args
            self.compiled = self.kernel[grid](*args, **kwargs)
            return self.compiled

        return launch


class _NoLaunch:
    def __getitem__(self, grid):
        return lambda *args, **kwargs: None


def _args(width, ratio):
    xs = list(fp4_inputs(12, width, ratio, page=64))
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    codes = torch.arange(xs[0].numel(), device="cuda").reshape_as(xs[0]) % 16
    xs[0].copy_(lut[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0))
    return xs


def _call(module, xs):
    return module.fp4_index_logits_req_to_token(
        *xs,
        candidate_block_size=8 if xs[-2] == 1 else 0,
        write_logits=xs[-2] != 1,
        query_group_size=6,
    )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("width", [1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 257])
def test_native_only_edges_and_instruction_evidence(monkeypatch, ratio, width):
    xs = _args(width, ratio)
    capture = _Capture(native._native)
    monkeypatch.setattr(native, "_native", capture)
    monkeypatch.setattr(native, "_fallback", _NoLaunch())
    assert_exact(_call(native, xs), _call(control, xs))
    flags, needed = capture.args[10:12]
    assert bool((flags == 1).all()) and bool((needed == 0).all())
    ptx = capture.compiled.asm["ptx"]
    assert "wgmma.mma_async" in ptx and ".e4m3.e4m3" in ptx
    assert ".bf16.bf16" not in ptx


@pytest.mark.parametrize("ratio", [1, 2])
def test_graph_replay_refreshes_eligibility_requests_lengths(
    monkeypatch, ratio
):
    xs = _args(257, ratio)
    capture = _Capture(native._native)
    monkeypatch.setattr(native, "_native", capture)
    original_q, original_req = xs[0].clone(), xs[3].clone()
    original_table = xs[5].clone()
    for _ in range(3):
        _call(native, xs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(native, xs)
    needed = capture.args[11]
    for step in range(6):
        xs[0].copy_(original_q)
        xs[3].copy_(original_req)
        xs[5].copy_(original_table)
        if step == 1:
            xs[0][:, 0, 0] = 1.1015625
        if step == 2:
            xs[3].copy_(original_req.roll(1))
        if step == 3:
            scales = xs[5][:, 64 * 64 :].view(xs[5].shape[0], 64, 4)
            assert (
                scales.untyped_storage().data_ptr()
                == xs[5].untyped_storage().data_ptr()
            )
            scales[:, :, 0] = 80
            scales[:, :, 1] = 120
        if step == 4:
            xs[5][:, : 64 * 64].bitwise_xor_(0x88)
        if step == 5:
            xs[1].mul_(0.9375)
        xs[4].copy_(torch.tensor([0, 1, 7, 65, 129, 257] * 2, device="cuda"))
        graph.replay()
        assert_exact(out, _call(control, xs))
        assert bool((needed == (1 if step in (1, 2, 3) else 0)).all())


@pytest.mark.parametrize("odd_stride", [False, True])
def test_unaligned_tables_keep_byte_addressed_bf16(monkeypatch, odd_stride):
    xs = _args(257, 1)
    table = xs[5]
    if odd_stride:
        storage = torch.empty(
            (table.shape[0], table.shape[1] + 1),
            dtype=torch.uint8,
            device="cuda",
        )
        xs[5] = storage[:, :-1]
    else:
        storage = torch.empty(
            table.numel() + 1, dtype=torch.uint8, device="cuda"
        )
        xs[5] = storage[1:].reshape_as(table)
    xs[5].copy_(table)
    xs[-1] = 655627
    with monkeypatch.context() as baseline:
        baseline.setattr(control, "_use_native_fp8", lambda *args: False)
        expected = _call(control, xs)

    def unexpected(*args, **kwargs):
        raise AssertionError("unaligned storage entered uint16 native loader")

    monkeypatch.setattr(native, "fp4_index_logits_req_to_token", unexpected)
    assert_exact(_call(control, xs), expected)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("kind", ["query_nan", "query_inf", "scale_nan"])
def test_nonfinite_operands_keep_bf16_bytes(monkeypatch, ratio, kind):
    xs = _args(257, ratio)
    if kind == "query_nan":
        xs[0][:, 0, 0] = float("nan")
    elif kind == "query_inf":
        xs[0][:, 0, 0] = float("inf")
    else:
        xs[5][:, 64 * 64 :].fill_(255)
    capture = _Capture(native._native)
    monkeypatch.setattr(native, "_native", capture)
    assert_exact(_call(native, xs), _call(control, xs))
    assert bool((capture.args[11] == 1).all())


def test_dense_graph_replay_defines_allocated_row_padding():
    xs = _args(129, 2)
    for _ in range(3):
        _call(native, xs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(native, xs)
    full = torch.as_strided(
        out, (out.shape[0], out.stride(0)), (out.stride(0), 1)
    )
    for length in [129, 0, 1, 129]:
        full.fill_(float("nan"))
        xs[4].fill_(length)
        graph.replay()
        expected = _call(control, xs)
        assert_exact(out, expected)
        # The small-width BF16 path leaves allocation padding unspecified.
        # Native aligned readers must still receive defined padding values.
        assert bool(torch.isneginf(full[:, xs[-1] :]).all())


def test_previous_optional_math_abi_preserves_bf16(monkeypatch):
    import flag_gems.fused.dsv41.fp4_fp8 as math

    xs = _args(257, 1)
    xs[-1] = 655627
    with monkeypatch.context() as baseline:
        baseline.setattr(control, "_use_native_fp8", lambda *args: False)
        expected = _call(control, xs)
    monkeypatch.setattr(math, "FP4_FP8_SCORE_ABI", 1)

    def unexpected(*args, **kwargs):
        raise AssertionError("previous math ABI entered the new native module")

    monkeypatch.setattr(native, "fp4_index_logits_req_to_token", unexpected)
    assert_exact(_call(control, xs), expected)


def test_unhinted_multiple_of_six_does_not_enter_native_tle(monkeypatch):
    xs = _args(257, 1)
    xs[-1] = 655627
    expected = control.fp4_index_logits_req_to_token(
        *xs, candidate_block_size=8, write_logits=False
    )

    def unexpected():
        raise AssertionError(
            "a batch multiple of six is not a semantic group hint"
        )

    monkeypatch.setattr(control, "_tle_native_fp8", unexpected)
    assert_exact(
        control.fp4_index_logits_req_to_token(
            *xs, candidate_block_size=8, write_logits=False
        ),
        expected,
    )


def test_missing_tle_helper_retains_native_gluon(monkeypatch):
    from flaggems_sglang.runtime.backend._nvidia.hopper.ops import (  # noqa: E402
        _dsv41_fp4_fp8_indexer as gluon,
    )

    xs = _args(257, 2)
    xs[-1] = 327813
    with monkeypatch.context() as baseline:
        baseline.setattr(control, "_use_tle_native_fp8", lambda *args: False)
        expected = _call(control, xs)
    monkeypatch.setattr(control, "_tle_native_fp8", lambda: None)
    capture = _Capture(gluon._native)
    monkeypatch.setattr(gluon, "_native", capture)
    assert_exact(_call(control, xs), expected)
    assert bool((capture.args[11] == 0).all())
    assert ".e4m3.e4m3" in capture.compiled.asm["ptx"]
