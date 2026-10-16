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

_MAX_BLOCK_N = 8192
_TILE_ELEMS = 1024


def _prune_row_configs(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    block_n = min(triton.next_power_of_2(N), _MAX_BLOCK_N)
    max_block_m = max(
        1, min(32, triton.next_power_of_2(M), _TILE_ELEMS // block_n)
    )
    pruned = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        if block_m > max_block_m:
            continue
        kwargs_ = dict(config.kwargs)
        kwargs_["BLOCK_N"] = block_n
        kwargs_["MASK_N"] = N != block_n
        kwargs_["MASK_M"] = M % block_m != 0
        pruned.append(
            triton.Config(
                kwargs_,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        )
    return pruned


@triton.autotune(
    configs=tuple(
        triton.Config({"BLOCK_M": block_m}, num_warps=num_warps, num_stages=1)
        for block_m in (1, 2, 4, 8, 16, 32)
        for num_warps in (1, 2, 4, 8)
    ),
    key=["M", "N"],
    prune_configs_by={"early_config_prune": _prune_row_configs},
)
@triton.jit
def _rmsnorm_hf_row_kernel(
    X,
    W,
    Y,
    eps,
    M,
    N,
    stride_xm,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_M: tl.constexpr,
    MASK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M

    if MASK_N:
        cmask = cols < N
        w = tl.load(W + cols, mask=cmask, other=0.0)
    else:
        w = tl.load(W + cols)

    x_off = rows[:, None] * stride_xm + cols[None, :]
    y_off = rows[:, None] * stride_ym + cols[None, :]
    if MASK_M:
        if MASK_N:
            m = rmask[:, None] & cmask[None, :]
        else:
            m = rmask[:, None]
        x = tl.load(X + x_off, mask=m, other=0.0)
    else:
        if MASK_N:
            x = tl.load(X + x_off, mask=cmask[None, :], other=0.0)
        else:
            x = tl.load(X + x_off)

    xf = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(xf * xf, axis=1) / N + eps)

    out = (xf * (rstd[:, None] * w.to(tl.float32))).to(Y.dtype.element_ty)

    if MASK_M:
        if MASK_N:
            tl.store(Y + y_off, out, mask=m)
        else:
            tl.store(Y + y_off, out, mask=rmask[:, None])
    else:
        if MASK_N:
            tl.store(Y + y_off, out, mask=cmask[None, :])
        else:
            tl.store(Y + y_off, out)


@triton.jit
def _rmsnorm_hf_loop_kernel(
    X,
    W,
    Y,
    eps,
    M,
    N,
    stride_xm,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M

    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        c = off + cols
        if MASK_M:
            m = rmask[:, None] & (c < N)[None, :]
            x = tl.load(
                X + rows[:, None] * stride_xm + c[None, :], mask=m, other=0.0
            )
        else:
            x = tl.load(
                X + rows[:, None] * stride_xm + c[None, :],
                mask=(c < N)[None, :],
                other=0.0,
            )
        xf = x.to(tl.float32)
        ss += tl.sum(xf * xf, axis=1)

    rstd = tl.rsqrt(ss / N + eps)

    for off in range(0, N, BLOCK_N):
        c = off + cols
        cm = c < N
        w = tl.load(W + c, mask=cm, other=0.0)
        if MASK_M:
            m = rmask[:, None] & cm[None, :]
        else:
            m = tl.broadcast_to(cm[None, :], (BLOCK_M, BLOCK_N))
        x = tl.load(
            X + rows[:, None] * stride_xm + c[None, :], mask=m, other=0.0
        )
        out = (x.to(tl.float32) * (rstd[:, None] * w.to(tl.float32))).to(
            Y.dtype.element_ty
        )
        tl.store(Y + rows[:, None] * stride_ym + c[None, :], out, mask=m)


def rmsnorm_hf(input, weight, eps):
    x = input
    w = weight
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()

    num_tokens, hidden = x.shape
    out = torch.empty_like(x)

    if num_tokens == 0 or hidden == 0:
        return out

    if hidden <= _MAX_BLOCK_N:

        def grid(META):
            return (triton.cdiv(num_tokens, META["BLOCK_M"]),)

        _rmsnorm_hf_row_kernel[grid](
            x,
            w,
            out,
            eps,
            num_tokens,
            hidden,
            x.stride(0),
            out.stride(0),
        )
    else:
        block_n = _MAX_BLOCK_N
        grid = (num_tokens,)
        _rmsnorm_hf_loop_kernel[grid](
            x,
            w,
            out,
            eps,
            num_tokens,
            hidden,
            x.stride(0),
            out.stride(0),
            BLOCK_M=1,
            BLOCK_N=block_n,
            MASK_M=False,
            num_warps=4,
            num_stages=1,
        )
    return out


__all__ = ["rmsnorm_hf"]
