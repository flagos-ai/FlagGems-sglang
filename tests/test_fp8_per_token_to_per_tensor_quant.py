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

"""Tests for moe/fp8_per_token_to_per_tensor_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

FP8 = torch.float8_e4m3fn
GROUP = 128
OUTPUT_SCALE = 2.0


def _build(
    num_experts, m, hidden, seed, column_major_scales=False, device="cuda"
):
    """Integer payload in [-8, 8] with power-of-two group scales.

    Both choices keep every intermediate exactly representable in e4m3, so a
    correct kernel agrees with the reference byte for byte rather than to a
    tolerance.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(-8, 9, (num_experts, m, hidden), generator=g).float()
    exps = torch.randint(-1, 2, (num_experts, m, hidden // GROUP), generator=g)
    x_scale = torch.pow(2.0, exps.float()).to(device)
    if column_major_scales:
        # DeepEP returns the last two scale dims column-major, for TMA.
        x_scale = x_scale.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    return x.to(device).to(FP8), x_scale


def _run(num_experts, m, hidden, masked, column_major_scales=False):
    x, x_scale = _build(
        num_experts,
        m,
        hidden,
        seed=hidden + num_experts,
        column_major_scales=column_major_scales,
    )
    masked_m = torch.tensor(masked, dtype=torch.int32, device=x.device)
    output_scale = torch.tensor(
        [OUTPUT_SCALE], dtype=torch.float32, device=x.device
    )

    out = flaggems_sglang.fp8_per_token_to_per_tensor_quant(
        x, x_scale, masked_m, output_scale
    )
    ref = get_reference("fp8_per_token_to_per_tensor_quant")(
        x, x_scale, masked_m, output_scale
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0)
    return out


@pytest.mark.parametrize("column_major_scales", [False, True])
@pytest.mark.parametrize("hidden", [7168, 3584, 1152])
def test_group_tail_and_scale_layout(hidden, column_major_scales):
    """7168 fills every tile; 3584 and 1152 leave the last group tile partial.

    1152 has nine groups, so the per-expert group count is not a power of two.
    """
    masked = [0, 1, 17, 48]
    out = _run(4, 48, hidden, masked, column_major_scales)
    # The band past masked_m is zero, and it genuinely was written: ``out`` is
    # a fresh allocation, so anything non-zero there came from the kernel.
    for e, valid in enumerate(masked):
        band = out[e, valid:].float()
        assert torch.equal(
            band, torch.zeros_like(band)
        ), f"expert {e} left rows past masked_m non-zero"


@pytest.mark.parametrize(
    "masked", [[0, 0, 0, 0], [0, 5, 0, 7], [9, 0, 0, 0], [0, 0, 0, 9]]
)
def test_experts_with_no_rows_are_skipped(masked):
    """An expert with zero live rows must come back entirely zeroed."""
    out = _run(4, 48, 3584, masked, column_major_scales=True)
    for e, valid in enumerate(masked):
        if valid == 0:
            assert torch.equal(
                out[e].float(), torch.zeros_like(out[e].float())
            ), f"expert {e} has no live rows but is not zero"


@pytest.mark.parametrize("num_experts", [1, 8, 16])
def test_expert_counts(num_experts):
    """One expert, a DeepSeek-V3 EP8 expert count, and a wider local set."""
    m = 48
    masked = [(e * 7) % (m + 1) for e in range(num_experts)]
    _run(num_experts, m, 3584, masked, column_major_scales=True)


def test_every_row_live_and_none_live():
    """The two ends: a full batch and an empty dispatch."""
    _run(4, 48, 3584, [48, 48, 48, 48])
    out = _run(4, 48, 3584, [0, 0, 0, 0], column_major_scales=True)
    assert torch.equal(out.float(), torch.zeros_like(out.float()))


def test_masked_m_past_m_keeps_every_row_live():
    """A live count past m simply selects all rows; nothing is read out of range."""
    x, x_scale = _build(4, 16, 3584, seed=7)
    masked_m = torch.tensor([16, 99, 0, 5], dtype=torch.int32, device=x.device)
    output_scale = torch.tensor(
        [OUTPUT_SCALE], dtype=torch.float32, device=x.device
    )
    out = flaggems_sglang.fp8_per_token_to_per_tensor_quant(
        x, x_scale, masked_m, output_scale
    )
    ref = get_reference("fp8_per_token_to_per_tensor_quant")(
        x, x_scale, masked_m, output_scale
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0)


def test_bfloat16_payload_is_accepted():
    """The group scales are what makes the payload fp8; a bf16 payload is fine."""
    x, x_scale = _build(4, 16, 3584, seed=11)
    x = x.float().to(torch.bfloat16)
    masked_m = torch.tensor([16, 3, 0, 9], dtype=torch.int32, device=x.device)
    output_scale = torch.tensor(
        [OUTPUT_SCALE], dtype=torch.float32, device=x.device
    )
    out = flaggems_sglang.fp8_per_token_to_per_tensor_quant(
        x, x_scale, masked_m, output_scale
    )
    ref = get_reference("fp8_per_token_to_per_tensor_quant")(
        x, x_scale, masked_m, output_scale
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0)


def test_rejects_bad_hidden_size():
    x = torch.zeros(2, 4, 1000, dtype=FP8, device="cuda")
    x_scale = torch.ones(
        2, 4, 1000 // GROUP + 1, dtype=torch.float32, device="cuda"
    )
    masked_m = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    output_scale = torch.tensor(
        [OUTPUT_SCALE], dtype=torch.float32, device="cuda"
    )
    with pytest.raises(ValueError, match="multiple of the 128 scale group"):
        flaggems_sglang.fp8_per_token_to_per_tensor_quant(
            x, x_scale, masked_m, output_scale
        )


def test_rejects_scale_shape_mismatch():
    x = torch.zeros(2, 4, 512, dtype=FP8, device="cuda")
    x_scale = torch.ones(2, 3, 4, dtype=torch.float32, device="cuda")
    masked_m = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    output_scale = torch.tensor(
        [OUTPUT_SCALE], dtype=torch.float32, device="cuda"
    )
    with pytest.raises(ValueError, match="x_scale must be"):
        flaggems_sglang.fp8_per_token_to_per_tensor_quant(
            x, x_scale, masked_m, output_scale
        )


def test_rejects_non_float32_scale():
    x = torch.zeros(2, 4, 512, dtype=FP8, device="cuda")
    x_scale = torch.ones(2, 4, 4, dtype=torch.bfloat16, device="cuda")
    masked_m = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    output_scale = torch.tensor(
        [OUTPUT_SCALE], dtype=torch.float32, device="cuda"
    )
    with pytest.raises(ValueError, match="x_scale must be float32"):
        flaggems_sglang.fp8_per_token_to_per_tensor_quant(
            x, x_scale, masked_m, output_scale
        )


def test_rejects_non_scalar_output_scale():
    x = torch.zeros(2, 4, 512, dtype=FP8, device="cuda")
    x_scale = torch.ones(2, 4, 4, dtype=torch.float32, device="cuda")
    masked_m = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    output_scale = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="output_scale must hold one element"):
        flaggems_sglang.fp8_per_token_to_per_tensor_quant(
            x, x_scale, masked_m, output_scale
        )
