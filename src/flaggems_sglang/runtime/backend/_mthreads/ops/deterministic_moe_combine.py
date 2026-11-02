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
"""Ordered routed BF16 reduction plus unmaterialized shared gate tail.

Caller joins all input producers before launch; the operator owns no stream
or persistent state. Each tensor is addressed using its own strides.
"""

import torch
import triton
import triton.language as tl

from ._tuning import launch_options as _launch_options


@triton.jit
def deterministic_moe_combine_kernel(
    routed_ptr,
    routed_stride_0,
    routed_stride_1,
    routed_stride_2,
    gate_logits_ptr,
    gate_stride_0,
    gate_stride_1,
    shared_ptr,
    shared_stride_0,
    shared_stride_1,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    routed_stride_0 = tl.cast(routed_stride_0, dtype=tl.int64)
    routed_stride_1 = tl.cast(routed_stride_1, dtype=tl.int64)
    routed_stride_2 = tl.cast(routed_stride_2, dtype=tl.int64)
    gate_stride_0 = tl.cast(gate_stride_0, dtype=tl.int64)
    gate_stride_1 = tl.cast(gate_stride_1, dtype=tl.int64)
    shared_stride_0 = tl.cast(shared_stride_0, dtype=tl.int64)
    shared_stride_1 = tl.cast(shared_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)
    output_stride_1 = tl.cast(output_stride_1, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)
    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim
    mask = mask_token[:, None] & mask_dim[None, :]

    routed_base = (
        routed_ptr
        + offs_token[:, None] * routed_stride_0
        + offs_dim[None, :] * routed_stride_2
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
    for topk_idx in tl.range(0, TOPK, num_stages=1):
        routed_tile = tl.load(
            routed_base + topk_idx * routed_stride_1,
            mask=mask,
            other=0.0,
        )
        accumulator += routed_tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    gate_logits = tl.load(
        gate_logits_ptr + offs_token * gate_stride_0 + 0 * gate_stride_1,
        mask=mask_token,
        other=0.0,
    ).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate_logits))
    shared_ptrs = (
        shared_ptr
        + offs_token[:, None] * shared_stride_0
        + offs_dim[None, :] * shared_stride_1
    )
    accumulator += gate[:, None] * tl.load(
        shared_ptrs, mask=mask, other=0.0
    ).to(tl.float32)

    output_ptrs = (
        output_ptr
        + offs_token[:, None] * output_stride_0
        + offs_dim[None, :] * output_stride_1
    )
    tl.store(
        output_ptrs,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def deterministic_moe_combine(
    routed, gate_logits, shared, out, *, launch_options=None
):
    tokens = routed.shape[0]
    if routed.ndim != 3 or tuple(routed.shape[1:]) != (8, 2048):
        raise ValueError("routed must have shape [tokens, 8, 2048]")
    if tokens not in (2048, 4096, 6144, 8192, 16384):
        raise ValueError("unsupported deterministic combine token count")
    if (
        gate_logits.shape != (tokens, 1)
        or shared.shape != (tokens, 2048)
        or out.shape != shared.shape
    ):
        raise ValueError("gate/shared/output shape mismatch")
    tensors = (routed, gate_logits, shared, out)
    if routed.device.type != "musa" or any(
        t.device != routed.device or t.dtype != torch.bfloat16 for t in tensors
    ):
        raise ValueError("combine requires BF16 tensors on one MUSA device")
    if any(torch._C._overlaps(out, t) for t in tensors[:-1]):
        raise ValueError("combine output must not alias its inputs")
    if not out.is_contiguous():
        raise ValueError("combine output must be contiguous")
    options = _launch_options(
        "deterministic_moe_combine",
        "m2048" if tokens == 2048 else "large_prefill",
        launch_options,
    )
    if (
        set(options) - {"BLOCK_M", "BLOCK_DIM", "num_warps", "num_stages"}
        or options["BLOCK_DIM"] > 2048
    ):
        raise ValueError("unsupported deterministic combine launch options")
    block_m, block_dim = options["BLOCK_M"], options["BLOCK_DIM"]
    with torch.musa.device(routed.device):
        deterministic_moe_combine_kernel[
            (triton.cdiv(tokens, block_m), 2048 // block_dim)
        ](
            routed,
            *routed.stride(),
            gate_logits,
            *gate_logits.stride(),
            shared,
            *shared.stride(),
            out,
            *out.stride(),
            token_num=tokens,
            hidden_dim=2048,
            routed_scaling_factor=1.0,
            TOPK=8,
            **options,
        )
    return out


__all__ = ["deterministic_moe_combine"]
