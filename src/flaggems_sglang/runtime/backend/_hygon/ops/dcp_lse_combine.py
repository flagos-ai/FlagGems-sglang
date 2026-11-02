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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl

_NEG_INF = tl.constexpr(float("-inf"))
_POS_INF = tl.constexpr(float("inf"))

_LARGE_ROW_THRESHOLD = 4096

_MAX_HOISTED_SHARDS = 8


@triton.jit
def _sanitize_lse(lse):
    return tl.where((lse != lse) | (lse == _POS_INF), _NEG_INF, lse)


@triton.jit
def _load_lse_masked(ptr, offs_m, mask_m, MASK_M: tl.constexpr):
    if MASK_M:
        return tl.load(ptr + offs_m, mask=mask_m, other=_NEG_INF)
    else:
        return tl.load(ptr + offs_m)


@triton.jit
def _load_output_masked(ptr, off, mask_m, MASK_M: tl.constexpr):
    if MASK_M:
        return tl.load(ptr + off, mask=mask_m[:, None], other=0.0)
    else:
        return tl.load(ptr + off)


@triton.jit
def _dcp_lse_combine_small_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    M,
    stride_on,
    stride_ln,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SHARDS: tl.constexpr,
    BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    MASK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    off = offs_m[:, None] * BLOCK_D + offs_d[None, :]

    lse0 = _sanitize_lse(
        _load_lse_masked(recv_lse_ptr, offs_m, mask_m, MASK_M).to(tl.float32)
    )
    row_max = lse0
    if NUM_SHARDS > 1:
        lse1 = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        row_max = tl.maximum(row_max, lse1)
    if NUM_SHARDS > 2:
        lse2 = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + 2 * stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        row_max = tl.maximum(row_max, lse2)
    if NUM_SHARDS > 3:
        lse3 = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + 3 * stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        row_max = tl.maximum(row_max, lse3)
    if NUM_SHARDS > 4:
        lse4 = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + 4 * stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        row_max = tl.maximum(row_max, lse4)
    if NUM_SHARDS > 5:
        lse5 = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + 5 * stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        row_max = tl.maximum(row_max, lse5)
    if NUM_SHARDS > 6:
        lse6 = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + 6 * stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        row_max = tl.maximum(row_max, lse6)
    if NUM_SHARDS > 7:
        lse7 = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + 7 * stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        row_max = tl.maximum(row_max, lse7)
    row_max = tl.where(row_max == _NEG_INF, 0.0, row_max)

    weight_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    if NUM_SHARDS > 0:
        c = lse0 - row_max
        w = tl.where(c == _NEG_INF, 0.0, tl.exp(c) if BASE_E else tl.exp2(c))
        weight_sum += w
        x = _load_output_masked(recv_output_ptr, off, mask_m, MASK_M)
        acc += x.to(tl.float32) * w[:, None]
    if NUM_SHARDS > 1:
        c = lse1 - row_max
        w = tl.where(c == _NEG_INF, 0.0, tl.exp(c) if BASE_E else tl.exp2(c))
        weight_sum += w
        x = _load_output_masked(
            recv_output_ptr + stride_on, off, mask_m, MASK_M
        )
        acc += x.to(tl.float32) * w[:, None]
    if NUM_SHARDS > 2:
        c = lse2 - row_max
        w = tl.where(c == _NEG_INF, 0.0, tl.exp(c) if BASE_E else tl.exp2(c))
        weight_sum += w
        x = _load_output_masked(
            recv_output_ptr + 2 * stride_on, off, mask_m, MASK_M
        )
        acc += x.to(tl.float32) * w[:, None]
    if NUM_SHARDS > 3:
        c = lse3 - row_max
        w = tl.where(c == _NEG_INF, 0.0, tl.exp(c) if BASE_E else tl.exp2(c))
        weight_sum += w
        x = _load_output_masked(
            recv_output_ptr + 3 * stride_on, off, mask_m, MASK_M
        )
        acc += x.to(tl.float32) * w[:, None]
    if NUM_SHARDS > 4:
        c = lse4 - row_max
        w = tl.where(c == _NEG_INF, 0.0, tl.exp(c) if BASE_E else tl.exp2(c))
        weight_sum += w
        x = _load_output_masked(
            recv_output_ptr + 4 * stride_on, off, mask_m, MASK_M
        )
        acc += x.to(tl.float32) * w[:, None]
    if NUM_SHARDS > 5:
        c = lse5 - row_max
        w = tl.where(c == _NEG_INF, 0.0, tl.exp(c) if BASE_E else tl.exp2(c))
        weight_sum += w
        x = _load_output_masked(
            recv_output_ptr + 5 * stride_on, off, mask_m, MASK_M
        )
        acc += x.to(tl.float32) * w[:, None]
    if NUM_SHARDS > 6:
        c = lse6 - row_max
        w = tl.where(c == _NEG_INF, 0.0, tl.exp(c) if BASE_E else tl.exp2(c))
        weight_sum += w
        x = _load_output_masked(
            recv_output_ptr + 6 * stride_on, off, mask_m, MASK_M
        )
        acc += x.to(tl.float32) * w[:, None]
    if NUM_SHARDS > 7:
        c = lse7 - row_max
        w = tl.where(c == _NEG_INF, 0.0, tl.exp(c) if BASE_E else tl.exp2(c))
        weight_sum += w
        x = _load_output_masked(
            recv_output_ptr + 7 * stride_on, off, mask_m, MASK_M
        )
        acc += x.to(tl.float32) * w[:, None]

    inv_sum = 1.0 / weight_sum
    out = (acc * inv_sum[:, None]).to(out_ptr.dtype.element_ty)
    if MASK_M:
        tl.store(out_ptr + off, out, mask=mask_m[:, None])
    else:
        tl.store(out_ptr + off, out)

    if RETURN_LSE:
        if BASE_E:
            merged_lse = tl.log(weight_sum) + row_max
        else:
            merged_lse = tl.log2(weight_sum) + row_max
        if MASK_M:
            tl.store(out_lse_ptr + offs_m, merged_lse, mask=mask_m)
        else:
            tl.store(out_lse_ptr + offs_m, merged_lse)


_LARGE_CONFIGS = [
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
]


@triton.autotune(configs=_LARGE_CONFIGS, key=["M"])
@triton.jit
def _dcp_lse_combine_large_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    M,
    stride_on,
    stride_ln,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SHARDS: tl.constexpr,
    BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    MASK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    off = offs_m[:, None] * BLOCK_D + offs_d[None, :]

    row_max = tl.full([BLOCK_M], _NEG_INF, tl.float32)
    for i in tl.range(0, NUM_SHARDS, num_stages=2):
        lse_i = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + i * stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        row_max = tl.maximum(row_max, lse_i)
    row_max = tl.where(row_max == _NEG_INF, 0.0, row_max)

    weight_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    for i in tl.static_range(0, NUM_SHARDS):
        lse_i = _sanitize_lse(
            _load_lse_masked(
                recv_lse_ptr + i * stride_ln, offs_m, mask_m, MASK_M
            ).to(tl.float32)
        )
        centered = lse_i - row_max
        w_i = tl.exp(centered) if BASE_E else tl.exp2(centered)
        w_i = tl.where(centered == _NEG_INF, 0.0, w_i)
        weight_sum += w_i

        x = _load_output_masked(
            recv_output_ptr + i * stride_on, off, mask_m, MASK_M
        )
        acc += x.to(tl.float32) * w_i[:, None]

    if RETURN_LSE:
        if BASE_E:
            merged_lse = tl.log(weight_sum) + row_max
        else:
            merged_lse = tl.log2(weight_sum) + row_max
        if MASK_M:
            tl.store(out_lse_ptr + offs_m, merged_lse, mask=mask_m)
        else:
            tl.store(out_lse_ptr + offs_m, merged_lse)

    inv_sum = 1.0 / weight_sum
    out = (acc * inv_sum[:, None]).to(out_ptr.dtype.element_ty)
    if MASK_M:
        tl.store(out_ptr + off, out, mask=mask_m[:, None])
    else:
        tl.store(out_ptr + off, out)


@triton.jit
def _dcp_lse_combine_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    M,
    H,
    D,
    stride_on,
    stride_ob,
    stride_oh,
    stride_od,
    stride_ln,
    stride_lb,
    stride_lh,
    stride_outb,
    stride_outh,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SHARDS: tl.constexpr,
    BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    b = offs_m // H
    h = offs_m % H
    row_ob = b * stride_ob + h * stride_oh
    row_lb = b * stride_lb + h * stride_lh
    row_out = b * stride_outb + h * stride_outh

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    out_mask = mask_m[:, None] & mask_d[None, :]

    row_max = tl.full([BLOCK_M], _NEG_INF, tl.float32)
    for i in tl.range(0, NUM_SHARDS):
        lse_i = tl.load(
            recv_lse_ptr + i * stride_ln + row_lb, mask=mask_m, other=_NEG_INF
        ).to(tl.float32)
        lse_i = _sanitize_lse(lse_i)
        row_max = tl.maximum(row_max, lse_i)
    row_max = tl.where(row_max == _NEG_INF, 0.0, row_max)

    weight_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    for i in tl.range(0, NUM_SHARDS):
        lse_i = tl.load(
            recv_lse_ptr + i * stride_ln + row_lb, mask=mask_m, other=_NEG_INF
        ).to(tl.float32)
        lse_i = _sanitize_lse(lse_i)
        centered = lse_i - row_max
        if BASE_E:
            w_i = tl.exp(centered)
        else:
            w_i = tl.exp2(centered)
        w_i = tl.where(centered == _NEG_INF, 0.0, w_i)
        weight_sum += w_i

        x = tl.load(
            recv_output_ptr
            + i * stride_on
            + row_ob[:, None]
            + offs_d[None, :] * stride_od,
            mask=out_mask,
            other=0.0,
        ).to(tl.float32)
        acc += x * w_i[:, None]

    out = acc * (1.0 / weight_sum)[:, None]
    tl.store(
        out_ptr + row_out[:, None] + offs_d[None, :] * stride_od,
        out,
        mask=out_mask,
    )

    if RETURN_LSE:
        if BASE_E:
            merged_lse = tl.log(weight_sum) + row_max
        else:
            merged_lse = tl.log2(weight_sum) + row_max
        tl.store(out_lse_ptr + row_lb, merged_lse, mask=mask_m)


def _can_flatten(recv_output, recv_lse):
    n, b, h, d = recv_output.shape
    if (
        recv_lse.stride(0) != b * h
        or recv_lse.stride(1) != h
        or recv_lse.stride(2) != 1
    ):
        return False
    return (
        recv_output.stride(1) == h * d
        and recv_output.stride(2) == d
        and recv_output.stride(3) == 1
    )


def _launch_config(d, bh):
    block_d = triton.next_power_of_2(max(d, 1))
    block_m = 16
    block_m = min(block_m, triton.next_power_of_2(max(bh, 1)))
    return block_m, block_d


def dcp_lse_combine(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    n, b, h, d = recv_output.shape

    out = torch.empty(
        (b, h, d), dtype=recv_output.dtype, device=recv_output.device
    )
    out_lse = (
        torch.empty((b, h), dtype=torch.float32, device=recv_lse.device)
        if return_lse
        else None
    )
    if b * h == 0:
        return out, out_lse

    m = b * h
    block_m, block_d = _launch_config(d, m)

    if d == block_d and _can_flatten(recv_output, recv_lse):
        rag = m % block_m != 0
        ro = recv_output.view(n, m, d)
        rl = recv_lse.view(n, m)
        o = out.view(m, d)
        ol = out_lse.view(m) if out_lse is not None else rl
        if m >= _LARGE_ROW_THRESHOLD or n > _MAX_HOISTED_SHARDS:
            _dcp_lse_combine_large_kernel[(triton.cdiv(m, block_m),)](
                ro,
                rl,
                o,
                ol,
                m,
                ro.stride(0),
                rl.stride(0),
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                NUM_SHARDS=n,
                BASE_E=bool(is_lse_base_on_e),
                RETURN_LSE=bool(return_lse),
                MASK_M=rag,
            )
        else:
            _dcp_lse_combine_small_kernel[(triton.cdiv(m, block_m),)](
                ro,
                rl,
                o,
                ol,
                m,
                ro.stride(0),
                rl.stride(0),
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                NUM_SHARDS=n,
                BASE_E=bool(is_lse_base_on_e),
                RETURN_LSE=bool(return_lse),
                MASK_M=rag,
                num_warps=4,
            )
        return out, out_lse

    grid = (triton.cdiv(m, block_m),)

    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse if out_lse is not None else recv_lse,
        m,
        h,
        d,
        recv_output.stride(0),
        recv_output.stride(1),
        recv_output.stride(2),
        recv_output.stride(3),
        recv_lse.stride(0),
        recv_lse.stride(1),
        recv_lse.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        NUM_SHARDS=n,
        BASE_E=bool(is_lse_base_on_e),
        RETURN_LSE=bool(return_lse),
        num_warps=2,
    )
    return out, out_lse


__all__ = ["dcp_lse_combine"]
