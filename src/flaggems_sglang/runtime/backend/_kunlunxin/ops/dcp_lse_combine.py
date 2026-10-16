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


@triton.jit
def _dcp_lse_combine_kernel(
    recv_out_ptr,
    recv_lse_ptr,
    out_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOTAL: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_B: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    L_STRIDE_N: tl.constexpr,
    L_STRIDE_B: tl.constexpr,
    L_STRIDE_H: tl.constexpr,
    BASE_E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bh = i // D
    d = i % D
    b = bh // H
    h = bh % H
    lo = b * L_STRIDE_B + h * L_STRIDE_H
    xo = b * X_STRIDE_B + h * X_STRIDE_H + d * X_STRIDE_D
    m = tl.full((BLOCK,), -float("inf"), tl.float32)
    for n in tl.static_range(N):
        lse_value = tl.load(
            recv_lse_ptr + n * L_STRIDE_N + lo, i < TOTAL, other=-float("inf")
        )
        lse_value = tl.where(
            (lse_value != lse_value) | (lse_value == float("inf")),
            -float("inf"),
            lse_value,
        )
        m = tl.maximum(m, lse_value)
    m = tl.where(m == -float("inf"), 0.0, m)
    z = tl.full((BLOCK,), 0.0, tl.float32)
    a = tl.full((BLOCK,), 0.0, tl.float32)
    for n in tl.static_range(N):
        lse_value = tl.load(
            recv_lse_ptr + n * L_STRIDE_N + lo, i < TOTAL, other=-float("inf")
        )
        lse_value = tl.where(
            (lse_value != lse_value) | (lse_value == float("inf")),
            -float("inf"),
            lse_value,
        )
        if BASE_E:
            w = tl.exp(lse_value - m)
        else:
            w = tl.exp((lse_value - m) * 0.6931471805599453)
        x = tl.load(recv_out_ptr + n * X_STRIDE_N + xo, i < TOTAL, other=0).to(
            tl.float32
        )
        a += x * w
        z += w
    tl.store(out_ptr + i, a / z, i < TOTAL)


@triton.jit
def _dcp_lse_kernel(
    recv_lse_ptr,
    O,
    N: tl.constexpr,
    H: tl.constexpr,
    TOTAL: tl.constexpr,
    L_STRIDE_N: tl.constexpr,
    L_STRIDE_B: tl.constexpr,
    L_STRIDE_H: tl.constexpr,
    BASE_E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = (i // H) * L_STRIDE_B + (i % H) * L_STRIDE_H
    m = tl.full((BLOCK,), -float("inf"), tl.float32)
    for n in tl.static_range(N):
        lse_value = tl.load(
            recv_lse_ptr + n * L_STRIDE_N + lo, i < TOTAL, other=-float("inf")
        )
        lse_value = tl.where(
            (lse_value != lse_value) | (lse_value == float("inf")),
            -float("inf"),
            lse_value,
        )
        m = tl.maximum(m, lse_value)
    m = tl.where(m == -float("inf"), 0.0, m)
    z = tl.full((BLOCK,), 0.0, tl.float32)
    for n in tl.static_range(N):
        lse_value = tl.load(
            recv_lse_ptr + n * L_STRIDE_N + lo, i < TOTAL, other=-float("inf")
        )
        lse_value = tl.where(
            (lse_value != lse_value) | (lse_value == float("inf")),
            -float("inf"),
            lse_value,
        )
        if BASE_E:
            w = tl.exp(lse_value - m)
        else:
            w = tl.exp((lse_value - m) * 0.6931471805599453)
        z += w
    if BASE_E:
        o = tl.log(z) + m
    else:
        o = tl.log(z) * 1.4426950408889634 + m
    tl.store(O + i, o, i < TOTAL)


@triton.jit
def _dcp_lse_weights_kernel(
    recv_lse_ptr,
    w_ptr,
    O,
    N: tl.constexpr,
    H: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    L_STRIDE_N: tl.constexpr,
    L_STRIDE_B: tl.constexpr,
    L_STRIDE_H: tl.constexpr,
    BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < NUM_ROWS
    lo = (i // H) * L_STRIDE_B + (i % H) * L_STRIDE_H
    m = tl.full((BLOCK,), -float("inf"), tl.float32)
    for n in tl.static_range(N):
        lse_value = tl.load(
            recv_lse_ptr + n * L_STRIDE_N + lo, valid, other=-float("inf")
        )
        lse_value = tl.where(
            (lse_value != lse_value) | (lse_value == float("inf")),
            -float("inf"),
            lse_value,
        )
        m = tl.maximum(m, lse_value)
    m = tl.where(m == -float("inf"), 0.0, m)
    z = tl.full((BLOCK,), 0.0, tl.float32)
    for n in tl.static_range(N):
        lse_value = tl.load(
            recv_lse_ptr + n * L_STRIDE_N + lo, valid, other=-float("inf")
        )
        lse_value = tl.where(
            (lse_value != lse_value) | (lse_value == float("inf")),
            -float("inf"),
            lse_value,
        )
        if BASE_E:
            w = tl.exp(lse_value - m)
        else:
            w = tl.exp((lse_value - m) * 0.6931471805599453)
        z += w
        tl.store(w_ptr + n * NUM_ROWS + i, w, valid)
    tl.store(w_ptr + N * NUM_ROWS + i, z, valid)
    if RETURN_LSE:
        if BASE_E:
            o = tl.log(z) + m
        else:
            o = tl.log(z) * 1.4426950408889634 + m
        tl.store(O + i, o, valid)


@triton.jit
def _dcp_lse_apply_kernel(
    recv_out_ptr,
    w_ptr,
    out_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    TOTAL: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_B: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    bh = i // D
    d = i % D
    b = bh // H
    h = bh % H
    xo = b * X_STRIDE_B + h * X_STRIDE_H + d * X_STRIDE_D
    a = tl.full((BLOCK,), 0.0, tl.float32)
    for n in tl.static_range(N):
        w = tl.load(w_ptr + n * NUM_ROWS + bh, valid, other=0)
        x = tl.load(recv_out_ptr + n * X_STRIDE_N + xo, valid, other=0).to(
            tl.float32
        )
        a += x * w
    z = tl.load(w_ptr + N * NUM_ROWS + bh, valid, other=1)
    tl.store(out_ptr + i, a / z, valid)


@triton.jit
def _dcp_lse_weights_reciprocal_kernel(
    recv_lse_ptr,
    w_ptr,
    O,
    N: tl.constexpr,
    H: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    L_STRIDE_N: tl.constexpr,
    L_STRIDE_B: tl.constexpr,
    L_STRIDE_H: tl.constexpr,
    BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < NUM_ROWS
    lo = (i // H) * L_STRIDE_B + (i % H) * L_STRIDE_H
    m = tl.full((BLOCK,), -float("inf"), tl.float32)
    for n in tl.static_range(N):
        lse_value = tl.load(
            recv_lse_ptr + n * L_STRIDE_N + lo, valid, other=-float("inf")
        )
        lse_value = tl.where(
            (lse_value != lse_value) | (lse_value == float("inf")),
            -float("inf"),
            lse_value,
        )
        m = tl.maximum(m, lse_value)
    m = tl.where(m == -float("inf"), 0.0, m)
    z = tl.full((BLOCK,), 0.0, tl.float32)
    for n in tl.static_range(N):
        lse_value = tl.load(
            recv_lse_ptr + n * L_STRIDE_N + lo, valid, other=-float("inf")
        )
        lse_value = tl.where(
            (lse_value != lse_value) | (lse_value == float("inf")),
            -float("inf"),
            lse_value,
        )
        if BASE_E:
            w = tl.exp(lse_value - m)
        else:
            w = tl.exp((lse_value - m) * 0.6931471805599453)
        z += w
        tl.store(w_ptr + n * NUM_ROWS + i, w, valid)
    tl.store(w_ptr + N * NUM_ROWS + i, 1.0 / z, valid)
    if RETURN_LSE:
        if BASE_E:
            o = tl.log(z) + m
        else:
            o = tl.log(z) * 1.4426950408889634 + m
        tl.store(O + i, o, valid)


@triton.jit
def _dcp_lse_apply_reciprocal_kernel(
    recv_out_ptr,
    w_ptr,
    out_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    TOTAL: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_B: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    bh = i // D
    d = i % D
    b = bh // H
    h = bh % H
    xo = b * X_STRIDE_B + h * X_STRIDE_H + d * X_STRIDE_D
    a = tl.full((BLOCK,), 0.0, tl.float32)
    for n in tl.static_range(N):
        w = tl.load(w_ptr + n * NUM_ROWS + bh, valid, other=0)
        x = tl.load(recv_out_ptr + n * X_STRIDE_N + xo, valid, other=0).to(
            tl.float32
        )
        a += x * w
    z = tl.load(w_ptr + N * NUM_ROWS + bh, valid, other=1)
    tl.store(out_ptr + i, a * z, valid)


def _packet(recv_output, recv_lse, block):
    n, b, h, d = recv_output.shape
    if block > 256 and not (
        0 < n <= 8
        and b > 0
        and h > 0
        and d in (128, 512)
        and recv_output.is_contiguous()
        and recv_lse.is_contiguous()
    ):
        return 256
    return block


def dcp_lse_combine(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    n, b, h, d = recv_output.shape
    out = torch.empty(
        (b, h, d), device=recv_output.device, dtype=recv_output.dtype
    )
    total = b * h * d
    block = _packet(
        recv_output,
        recv_lse,
        16384 if total >= 524288 else 4096 if total >= 65536 else 1024,
    )
    xs0, xs1, xs2, xs3 = (int(s) for s in recv_output.stride())
    ls0, ls1, ls2 = (int(s) for s in recv_lse.stride())
    if (
        2 <= n <= 8
        and d in (128, 512)
        and total >= 131072
        and recv_output.is_contiguous()
        and recv_lse.is_contiguous()
    ):
        out_lse = (
            torch.empty((b, h), device=recv_lse.device, dtype=recv_lse.dtype)
            if return_lse
            else None
        )
        weights = torch.empty(
            (n + 1, b * h), device=recv_output.device, dtype=torch.float32
        )
        if d == 128 and total >= 524288:
            _dcp_lse_weights_reciprocal_kernel[(triton.cdiv(b * h, 512),)](
                recv_lse,
                weights,
                out_lse if return_lse else out,
                N=n,
                H=h,
                NUM_ROWS=b * h,
                L_STRIDE_N=ls0,
                L_STRIDE_B=ls1,
                L_STRIDE_H=ls2,
                BASE_E=is_lse_base_on_e,
                RETURN_LSE=return_lse,
                BLOCK=512,
                num_warps=1,
                enable_fp_fusion=False,
            )
            _dcp_lse_apply_reciprocal_kernel[(triton.cdiv(total, block),)](
                recv_output,
                weights,
                out,
                N=n,
                H=h,
                D=d,
                NUM_ROWS=b * h,
                TOTAL=total,
                X_STRIDE_N=xs0,
                X_STRIDE_B=xs1,
                X_STRIDE_H=xs2,
                X_STRIDE_D=xs3,
                BLOCK=block,
                num_warps=1,
                enable_fp_fusion=False,
            )
            return out, out_lse
        _dcp_lse_weights_kernel[(triton.cdiv(b * h, 512),)](
            recv_lse,
            weights,
            out_lse if return_lse else out,
            N=n,
            H=h,
            NUM_ROWS=b * h,
            L_STRIDE_N=ls0,
            L_STRIDE_B=ls1,
            L_STRIDE_H=ls2,
            BASE_E=is_lse_base_on_e,
            RETURN_LSE=return_lse,
            BLOCK=512,
            num_warps=1,
            enable_fp_fusion=False,
        )
        _dcp_lse_apply_kernel[(triton.cdiv(total, block),)](
            recv_output,
            weights,
            out,
            N=n,
            H=h,
            D=d,
            NUM_ROWS=b * h,
            TOTAL=total,
            X_STRIDE_N=xs0,
            X_STRIDE_B=xs1,
            X_STRIDE_H=xs2,
            X_STRIDE_D=xs3,
            BLOCK=block,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return out, out_lse
    if total > 0:
        _dcp_lse_combine_kernel[(triton.cdiv(total, block),)](
            recv_output,
            recv_lse,
            out,
            N=n,
            H=h,
            D=d,
            TOTAL=total,
            X_STRIDE_N=xs0,
            X_STRIDE_B=xs1,
            X_STRIDE_H=xs2,
            X_STRIDE_D=xs3,
            L_STRIDE_N=ls0,
            L_STRIDE_B=ls1,
            L_STRIDE_H=ls2,
            BASE_E=is_lse_base_on_e,
            BLOCK=block,
            num_warps=1,
            enable_fp_fusion=False,
        )
    out_lse = None
    if return_lse:
        out_lse = torch.empty(
            (b, h), device=recv_lse.device, dtype=recv_lse.dtype
        )
        if b * h > 0:
            _dcp_lse_kernel[(triton.cdiv(b * h, 256),)](
                recv_lse,
                out_lse,
                N=n,
                H=h,
                TOTAL=b * h,
                L_STRIDE_N=ls0,
                L_STRIDE_B=ls1,
                L_STRIDE_H=ls2,
                BASE_E=is_lse_base_on_e,
                BLOCK=256,
                num_warps=1,
                enable_fp_fusion=False,
            )
    return out, out_lse


__all__ = ["dcp_lse_combine"]
