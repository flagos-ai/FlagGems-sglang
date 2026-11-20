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


@triton.jit(do_not_specialize=["X", "G", "B", "Out"])
def _tile(X, G, B, Out, P: tl.constexpr):
    START: tl.constexpr = P[0]
    N: tl.constexpr = P[1]
    C: tl.constexpr = P[2]
    GS: tl.constexpr = P[3]
    BS: tl.constexpr = P[4]
    SCALE: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    BIAS: tl.constexpr = P[7]
    BC: tl.constexpr = P[8]
    BR: tl.constexpr = P[9]
    row = START + tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), EPS)
    y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = y / (1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(Out + offset, y)
    else:
        tl.store(Out + offset, y, (row[:, None] < N) & (col[None, :] < C))


@triton.jit(do_not_specialize=["X", "G", "B"])
def _tile_inplace(X, G, B, P: tl.constexpr):
    START: tl.constexpr = P[0]
    N: tl.constexpr = P[1]
    C: tl.constexpr = P[2]
    GS: tl.constexpr = P[3]
    BS: tl.constexpr = P[4]
    SCALE: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    BIAS: tl.constexpr = P[7]
    BC: tl.constexpr = P[8]
    BR: tl.constexpr = P[9]
    row = START + tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), EPS)
    y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = y / (1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(X + offset, y)
    else:
        tl.store(X + offset, y, (row[:, None] < N) & (col[None, :] < C))


def _output(x, gamma, bias):
    dtype = torch.promote_types(x.dtype, gamma.dtype)
    reuse = dtype == x.dtype
    if reuse:
        low = x.data_ptr()
        high = low + x.numel() * x.element_size()
        for weight in (gamma, bias):
            if weight is not None:
                begin = weight.data_ptr()
                end = (
                    begin
                    + ((weight.numel() - 1) * weight.stride(0) + 1)
                    * weight.element_size()
                )
                if begin < high and end > low:
                    reuse = False
    if reuse:
        return x
    return torch.empty_strided(
        x.shape, x.stride(), dtype=dtype, device=x.device
    )


def wan_rmsnorm_silu(x, gamma, bias, rms_scale, eps):
    c = x.size(1)
    n = x.numel() // c
    out = _output(x, gamma, bias)
    b = gamma if bias is None else bias
    gs874 = gamma.stride(0)
    bs874 = gs874 if bias is None else b.stride(0)
    bc = triton.next_power_of_2(c)
    br = max(1, min(128, 16384 // bc))
    scale = c**0.5 if rms_scale is None else rms_scale
    step = 32768 * br
    if n <= step:
        if n:
            if bias is None:
                q970 = (n, c, gs874, scale, eps, bc, br)
                if out is x:
                    _short970_01.run(
                        x,
                        gamma,
                        q970,
                        grid=((n + br - 1) // br,),
                        warmup=False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
                else:
                    _short970_00.run(
                        x,
                        gamma,
                        out,
                        q970,
                        grid=((n + br - 1) // br,),
                        warmup=False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
            else:
                q970 = (n, c, gs874, bs874, scale, eps, bc, br)
                if out is x:
                    _short970_11.run(
                        x,
                        gamma,
                        b,
                        q970,
                        grid=((n + br - 1) // br,),
                        warmup=False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
                else:
                    _short970_10.run(
                        x,
                        gamma,
                        b,
                        out,
                        q970,
                        grid=((n + br - 1) // br,),
                        warmup=False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
        return out
    for start in range(0, n, step):
        count = min(n - start, step)
        params = (
            start,
            n,
            c,
            gs874,
            bs874,
            scale,
            eps,
            bias is not None,
            bc,
            br,
        )
        if out is x:
            _tile_inplace.run(
                x,
                gamma,
                b,
                params,
                grid=((count + br - 1) // br,),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
        else:
            _tile.run(
                x,
                gamma,
                b,
                out,
                params,
                grid=((count + br - 1) // br,),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
    return out


@triton.jit(do_not_specialize=["X", "G", "Out"])
def _short970_00(X, G, Out, Q: tl.constexpr):
    P: tl.constexpr = (
        0,
        Q[0],
        Q[1],
        Q[2],
        Q[3] if False else Q[2],
        Q[3],
        Q[4],
        False,
        Q[5],
        Q[6],
    )
    START: tl.constexpr = P[0]
    N: tl.constexpr = P[1]
    C: tl.constexpr = P[2]
    GS: tl.constexpr = P[3]
    BS: tl.constexpr = P[4]
    SCALE: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    BIAS: tl.constexpr = P[7]
    BC: tl.constexpr = P[8]
    BR: tl.constexpr = P[9]
    row = START + tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), EPS)
    y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(G + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = y / (1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(Out + offset, y)
    else:
        tl.store(Out + offset, y, (row[:, None] < N) & (col[None, :] < C))


@triton.jit(do_not_specialize=["X", "G"])
def _short970_01(X, G, Q: tl.constexpr):
    P: tl.constexpr = (
        0,
        Q[0],
        Q[1],
        Q[2],
        Q[3] if False else Q[2],
        Q[3],
        Q[4],
        False,
        Q[5],
        Q[6],
    )
    START: tl.constexpr = P[0]
    N: tl.constexpr = P[1]
    C: tl.constexpr = P[2]
    GS: tl.constexpr = P[3]
    BS: tl.constexpr = P[4]
    SCALE: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    BIAS: tl.constexpr = P[7]
    BC: tl.constexpr = P[8]
    BR: tl.constexpr = P[9]
    row = START + tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), EPS)
    y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(G + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = y / (1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(X + offset, y)
    else:
        tl.store(X + offset, y, (row[:, None] < N) & (col[None, :] < C))


@triton.jit(do_not_specialize=["X", "G", "B", "Out"])
def _short970_10(X, G, B, Out, Q: tl.constexpr):
    P: tl.constexpr = (
        0,
        Q[0],
        Q[1],
        Q[2],
        Q[3] if True else Q[2],
        Q[4],
        Q[5],
        True,
        Q[6],
        Q[7],
    )
    START: tl.constexpr = P[0]
    N: tl.constexpr = P[1]
    C: tl.constexpr = P[2]
    GS: tl.constexpr = P[3]
    BS: tl.constexpr = P[4]
    SCALE: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    BIAS: tl.constexpr = P[7]
    BC: tl.constexpr = P[8]
    BR: tl.constexpr = P[9]
    row = START + tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), EPS)
    y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = y / (1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(Out + offset, y)
    else:
        tl.store(Out + offset, y, (row[:, None] < N) & (col[None, :] < C))


@triton.jit(do_not_specialize=["X", "G", "B"])
def _short970_11(X, G, B, Q: tl.constexpr):
    P: tl.constexpr = (
        0,
        Q[0],
        Q[1],
        Q[2],
        Q[3] if True else Q[2],
        Q[4],
        Q[5],
        True,
        Q[6],
        Q[7],
    )
    START: tl.constexpr = P[0]
    N: tl.constexpr = P[1]
    C: tl.constexpr = P[2]
    GS: tl.constexpr = P[3]
    BS: tl.constexpr = P[4]
    SCALE: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    BIAS: tl.constexpr = P[7]
    BC: tl.constexpr = P[8]
    BR: tl.constexpr = P[9]
    row = START + tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), EPS)
    y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = y / (1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(X + offset, y)
    else:
        tl.store(X + offset, y, (row[:, None] < N) & (col[None, :] < C))


__all__ = ["wan_rmsnorm_silu"]
