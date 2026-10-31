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

# Adapted from flagos-ai/sglang-plugin-FL PR #125 at 68b414959bbf6ce89501c149293294d83d742e91.
"""S5000 M4/H2048 gate-tail with BF16 logits and sigmoid boundaries.

Output/debug buffers must not overlap inputs or one another. No fallback
executes inside this operator. Caller-owned output permits graph replay.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._tuning import launch_options as _launch_options

PRODUCT_M, PRODUCT_H = 4, 2048


def _storage_span(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return a conservative storage interval for fail-closed alias checks."""

    try:
        storage = tensor.untyped_storage()
        start = int(storage.data_ptr())
        return start, start + int(storage.nbytes())
    except (
        AttributeError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def _may_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively detect overlap, including views sharing one storage."""

    if left.device != right.device:
        return False
    left_span = _storage_span(left)
    right_span = _storage_span(right)
    if left_span is None or right_span is None:
        return True
    return max(left_span[0], right_span[0]) < min(left_span[1], right_span[1])


def candidate_guard_reason(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    gate_logits_out: torch.Tensor | None = None,
    gate_prob_out: torch.Tensor | None = None,
    enabled: bool | None = None,
    capturing: bool | None = None,
) -> str:
    """Explain why the candidate is or is not eligible.

    The reasons are intentionally conservative and stable enough for a
    harness to record.  ``enabled`` is injectable for CPU policy tests; the
    production default enables the guarded S5000 path unless explicitly disabled.
    """

    if enabled is None:
        enabled = True
    if not enabled:
        return "disabled"

    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (hidden, gate_weight, shared_output)
    ):
        return "non_tensor_input"
    if hidden.ndim != 2 or shared_output.ndim != 2 or gate_weight.ndim != 2:
        return "rank_not_2"
    if hidden.shape != (PRODUCT_M, PRODUCT_H):
        return "hidden_shape_not_M4_H2048"
    if gate_weight.shape != (1, PRODUCT_H):
        return "gate_weight_shape_not_1xH"
    if shared_output.shape != (PRODUCT_M, PRODUCT_H):
        return "shared_output_shape_not_M4_H2048"
    if hidden.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
        return "input_dtype_not_bfloat16"
    if shared_output.dtype != torch.bfloat16:
        return "shared_dtype_not_bfloat16"
    if (
        hidden.device != gate_weight.device
        or hidden.device != shared_output.device
    ):
        return "device_mismatch"
    if (
        not hidden.is_contiguous()
        or not gate_weight.is_contiguous()
        or not shared_output.is_contiguous()
    ):
        return "input_layout_not_contiguous"
    if tuple(hidden.stride()) != (PRODUCT_H, 1):
        return "hidden_stride_not_MajorH"
    if tuple(gate_weight.stride()) != (PRODUCT_H, 1):
        return "gate_stride_not_H_major"
    if tuple(shared_output.stride()) != (PRODUCT_H, 1):
        return "shared_stride_not_MajorH"
    if _may_alias(hidden, gate_weight):
        return "hidden_aliases_gate_weight"
    if _may_alias(hidden, shared_output):
        return "hidden_aliases_shared_output"
    if _may_alias(gate_weight, shared_output):
        return "gate_weight_aliases_shared_output"

    if out is not None:
        if not isinstance(out, torch.Tensor):
            return "output_not_tensor"
        if out.shape != (PRODUCT_M, PRODUCT_H):
            return "output_shape_not_M4_H2048"
        if out.dtype != torch.bfloat16:
            return "output_dtype_not_bfloat16"
        if out.device != hidden.device:
            return "output_device_mismatch"
        if not out.is_contiguous() or tuple(out.stride()) != (PRODUCT_H, 1):
            return "output_layout_not_contiguous"
        if any(
            _may_alias(out, tensor)
            for tensor in (hidden, gate_weight, shared_output)
        ):
            return "output_aliases_input"
    elif capturing is True:
        return "output_required_during_capture"

    debug_outputs = (
        ("gate_logits_debug", gate_logits_out),
        ("gate_prob_debug", gate_prob_out),
    )
    for debug_name, debug_output in debug_outputs:
        if debug_output is None:
            continue
        if not isinstance(debug_output, torch.Tensor):
            return f"{debug_name}_not_tensor"
        if debug_output.shape != (PRODUCT_M, 1):
            return f"{debug_name}_shape_not_Mx1"
        if debug_output.dtype != torch.bfloat16:
            return f"{debug_name}_dtype_not_bfloat16"
        if debug_output.device != hidden.device:
            return f"{debug_name}_device_mismatch"
        if not debug_output.is_contiguous() or tuple(
            debug_output.stride()
        ) != (1, 1):
            return f"{debug_name}_layout_not_contiguous"
        if any(
            _may_alias(debug_output, tensor)
            for tensor in (hidden, gate_weight, shared_output, out)
            if tensor is not None
        ):
            return f"{debug_name}_aliases_input_or_output"
    if (
        gate_logits_out is not None
        and gate_prob_out is not None
        and _may_alias(gate_logits_out, gate_prob_out)
    ):
        return "debug_outputs_alias"

    if hidden.device.type != "musa":
        return "device_not_musa"
    return "eligible"


@triton.jit
def _shared_expert_gate_tail_kernel(
    hidden_ptr,
    gate_ptr,
    shared_ptr,
    out_ptr,
    gate_logits_out_ptr,
    gate_prob_out_ptr,
    hidden_stride_m,
    hidden_stride_h,
    shared_stride_m,
    shared_stride_h,
    out_stride_m,
    out_stride_h,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    WRITE_DEBUG_LOGITS: tl.constexpr,
    WRITE_DEBUG_PROB: tl.constexpr,
):
    """One row program; deliberately no temporary gate/probability tensor."""

    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_H)
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # This is an explicit FP32 accumulation, not a claim about the
    # reduction order used by MUSA F.linear.  Bitwise validation must
    # compare the resulting BF16 gate logits before accepting this path.
    for start in range(0, H, BLOCK_H):
        columns = start + offsets
        mask = columns < H
        x = tl.load(
            hidden_ptr + row * hidden_stride_m + columns * hidden_stride_h,
            mask=mask,
            other=0,
        )
        weight = tl.load(gate_ptr + columns, mask=mask, other=0)
        accumulator += x.to(tl.float32) * weight.to(tl.float32)

    # Preserve the first eager materialization boundary before sigmoid.
    gate_logits_bf16 = tl.sum(accumulator, axis=0).to(tl.bfloat16)
    if WRITE_DEBUG_LOGITS:
        tl.store(gate_logits_out_ptr + row, gate_logits_bf16)
    gate_prob_bf16 = tl.sigmoid(gate_logits_bf16.to(tl.float32)).to(
        tl.bfloat16
    )
    if WRITE_DEBUG_PROB:
        tl.store(gate_prob_out_ptr + row, gate_prob_bf16)

    # Preserve a BF16 broadcast multiply and the final BF16 output store.
    for start in range(0, H, BLOCK_H):
        columns = start + offsets
        mask = columns < H
        shared = tl.load(
            shared_ptr + row * shared_stride_m + columns * shared_stride_h,
            mask=mask,
            other=0,
        )
        scaled = shared * gate_prob_bf16
        tl.store(
            out_ptr + row * out_stride_m + columns * out_stride_h,
            scaled,
            mask=mask,
        )


def fused_shared_expert_gate_tail(
    hidden,
    gate_weight,
    shared_output,
    out=None,
    *,
    gate_logits_out=None,
    gate_prob_out=None,
    launch_options=None,
):
    reason = candidate_guard_reason(
        hidden,
        gate_weight,
        shared_output,
        out,
        gate_logits_out=gate_logits_out,
        gate_prob_out=gate_prob_out,
        enabled=True,
        capturing=False,
    )
    if reason != "eligible":
        raise ValueError(f"unsupported shared gate-tail input: {reason}")
    if out is None:
        out = torch.empty_like(shared_output)
    options = _launch_options(
        "fused_shared_expert_gate_tail", "m4_h2048", launch_options
    )
    if (
        set(options) - {"BLOCK_H", "num_warps", "num_stages"}
        or options["BLOCK_H"] > PRODUCT_H
    ):
        raise ValueError("unsupported shared gate-tail launch options")
    with torch.musa.device(hidden.device):
        _shared_expert_gate_tail_kernel[(PRODUCT_M,)](
            hidden,
            gate_weight,
            shared_output,
            out,
            gate_logits_out if gate_logits_out is not None else out,
            gate_prob_out if gate_prob_out is not None else out,
            *hidden.stride(),
            *shared_output.stride(),
            *out.stride(),
            H=PRODUCT_H,
            **options,
            WRITE_DEBUG_LOGITS=gate_logits_out is not None,
            WRITE_DEBUG_PROB=gate_prob_out is not None,
        )
    return out


__all__ = ["fused_shared_expert_gate_tail"]
