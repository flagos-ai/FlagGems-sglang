# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _encode(v):
    bits = tl.minimum(v.to(tl.int32, bitcast=True) & 2147483647, 1138753536)
    magnitude = bits.to(tl.float32, bitcast=True)
    bias = tl.maximum(bits & 2139095040, 1015021568) + 167772160
    biased = magnitude + bias.to(tl.float32, bitcast=True)
    code = biased.to(tl.int32, bitcast=True) - bias + (bias >> 20) - 1128
    code = tl.where(v != v, 127, code)
    return code


@triton.jit
def _activate(
    X,
    A: tl.pointer_type(tl.float32),
    P: tl.pointer_type(tl.float32),
    NT,
    D: tl.constexpr,
    T: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    B: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.load(NT)
    lane = tl.arange(0, B)
    C: tl.constexpr = triton.cdiv(D, B)
    maximum = tl.full((B,), 0, tl.float32)
    for tile in range(tl.program_id(0), n * C, tl.num_programs(0)):
        row = tile // C
        col = tile % C * B + lane
        if D % B == 0:
            gate = tl.load(X + row * SX + col * CX).to(tl.float32)
            up = tl.load(X + row * SX + (col + D) * CX).to(tl.float32)
        else:
            safe = tl.minimum(col, D - 1)
            gate = tl.load(X + row * SX + safe * CX).to(tl.float32)
            up = tl.load(X + row * SX + (safe + D) * CX).to(tl.float32)
        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
        value = tl.where(col < D, value, 0.0)
        maximum = tl.maximum(maximum, tl.abs(value))
        tl.store(A + row * D + col, value, col < D)
    tl.store(P + tl.program_id(0), tl.max(maximum, 0))
    padding = tl.program_id(0) + tl.num_programs(0)
    tl.store(P + padding, 0.0, padding < BK)


@triton.jit
def _scale(P, S, K: tl.constexpr, B: tl.constexpr):
    lane = tl.arange(0, B)
    maximum = tl.load(P + tl.minimum(lane, K - 1))
    maximum = tl.where(lane < K, maximum, 0.0)
    scale = tl.maximum(tl.max(maximum, 0), 1e-10) / 448.0
    tl.store(S, scale)


@triton.jit
def _quant(
    A: tl.pointer_type(tl.float32),
    Q: tl.pointer_type(tl.uint8),
    P: tl.pointer_type(tl.float32),
    S,
    NT,
    D: tl.constexpr,
    T: tl.constexpr,
    SQ: tl.constexpr,
    CQ: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.load(NT)
    reduction_lane = tl.arange(0, BK)
    partial = tl.load(P + reduction_lane)
    scale = tl.maximum(tl.max(partial, 0), 1e-10) / 448.0
    if tl.program_id(0) == 0:
        tl.store(S, scale)
    lane = tl.arange(0, B)
    C: tl.constexpr = triton.cdiv(D, B)
    for tile in range(tl.program_id(0), n * C, tl.num_programs(0)):
        row = tile // C
        col = tile % C * B + lane
        if D % B == 0:
            value = tl.load(A + row * D + col)
        else:
            safe = tl.minimum(col, D - 1)
            value = tl.load(A + row * D + safe)
        quantized = value / scale
        sign = quantized.to(tl.int32, bitcast=True) >> 24 & 128
        code = (_encode(tl.abs(quantized)) | sign).to(tl.uint8)
        tl.store(Q + row * SQ + col * CQ, code, col < D)


def _ordinary(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    rows = input.shape[0]
    width = intermediate_size
    result = output
    result_bytes = result
    result_scale = scale
    sx, cx = input.stride()
    so, co = result_bytes.stride()
    block = min(2048, width and 1 << (width - 1).bit_length())
    quant_block = min(block, 32) if co != 1 else block
    programs = min(40, rows * ((width + block - 1) // block))
    partial_block = programs and 1 << (programs - 1).bit_length()
    activation = input.new_empty(
        (rows * width + partial_block,), dtype=torch.float32
    )
    activation_ptr = activation.data_ptr()
    partial = activation_ptr + 4 * rows * width
    _activate.run(
        input,
        activation_ptr,
        partial,
        num_tokens_tensor,
        width,
        rows,
        sx,
        cx,
        block,
        partial_block,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    _quant.run(
        activation_ptr,
        result_bytes,
        partial,
        result_scale,
        num_tokens_tensor,
        width,
        rows,
        so,
        co,
        quant_block,
        programs,
        partial_block,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return (result, result_scale)


@triton.jit
def _activate_rows(
    X,
    A: tl.pointer_type(tl.float32),
    P: tl.pointer_type(tl.float32),
    NT,
    D: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.load(NT)
    row_lane = tl.arange(0, 2)
    col = tl.arange(0, D)
    maximum = tl.full((2, D), 0, tl.float32)
    for tile in range(tl.program_id(0), (n + 1) // 2, tl.num_programs(0)):
        row = 2 * tile + row_lane
        offsets = row[:, None] * SX + col[None, :] * CX
        if tile * 2 + 1 < n:
            gate = tl.load(X + offsets).to(tl.float32)
            up = tl.load(X + offsets + D * CX).to(tl.float32)
        else:
            gate = tl.load(X + offsets, row[:, None] < n, other=0).to(
                tl.float32
            )
            up = tl.load(X + offsets + D * CX, row[:, None] < n, other=0).to(
                tl.float32
            )
        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
        maximum = tl.maximum(maximum, tl.abs(value))
        if tile * 2 + 1 < n:
            tl.store(A + row[:, None] * D + col[None, :], value)
        else:
            tl.store(
                A + row[:, None] * D + col[None, :], value, row[:, None] < n
            )
    tl.store(P + tl.program_id(0), tl.max(tl.max(maximum, 1), 0))
    padding = tl.program_id(0) + tl.num_programs(0)
    tl.store(P + padding, 0.0, padding < BK)


@triton.jit
def _quant_rows(
    A: tl.pointer_type(tl.float32),
    Q: tl.pointer_type(tl.uint8),
    P: tl.pointer_type(tl.float32),
    S,
    NT,
    D: tl.constexpr,
    SQ: tl.constexpr,
    CQ: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.load(NT)
    partial = tl.load(P + tl.arange(0, BK))
    scale = tl.maximum(tl.max(partial, 0), 1e-10) / 448.0
    if tl.program_id(0) == 0:
        tl.store(S, scale)
    B: tl.constexpr = 2048 if D >= 1024 else 2 * D
    lane = tl.arange(0, B)
    for start in range(tl.program_id(0) * B, n * D, tl.num_programs(0) * B):
        index = start + lane
        if start + B <= n * D:
            value = tl.load(A + index)
        else:
            value = tl.load(A + index, index < n * D, other=0)
        quantized = value / scale
        sign = quantized.to(tl.int32, bitcast=True) >> 24 & 128
        code = (_encode(tl.abs(quantized)) | sign).to(tl.uint8)
        if SQ == D and CQ == 1:
            offset = index
        else:
            offset = index // D * SQ + index % D * CQ
        if start + B <= n * D:
            tl.store(Q + offset, code)
        else:
            tl.store(Q + offset, code, index < n * D)


def silu_mul_dynamic_tensorwise_quant(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    rows = input.shape[0]
    width = intermediate_size
    if width <= 0 or width > 2048 or width & width - 1:
        return _ordinary(
            input, output, scale, num_tokens_tensor, expected_num_tokens, width
        )
    programs = min(40, (rows + 1) // 2)
    partial_block = 1 << (programs - 1).bit_length()
    workspace = input.new_empty(
        (rows * width + partial_block,), dtype=torch.float32
    )
    activation = workspace.data_ptr()
    partial = activation + 4 * rows * width
    sx, cx = input.stride()
    sq, cq = output.stride()
    _activate_rows.run(
        input,
        activation,
        partial,
        num_tokens_tensor,
        width,
        sx,
        cx,
        partial_block,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    _quant_rows.run(
        activation,
        output,
        partial,
        scale,
        num_tokens_tensor,
        width,
        sq,
        cq,
        partial_block,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return (output, scale)


__all__ = ["silu_mul_dynamic_tensorwise_quant"]
