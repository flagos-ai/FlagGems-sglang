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

"""Operator: fla/kda_gate_chunk_cumsum"""

import torch
import triton
import triton.language as tl


def _cfg(BLOCK_T, num_warps, num_stages, BLOCK_S, UNROLL=1):
    return triton.Config(
        {"BLOCK_T": BLOCK_T, "BLOCK_S": BLOCK_S, "UNROLL": UNROLL},
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _gate(x, a, lb_s, USE_LB: tl.constexpr):
    """KDA gate on fp32 x. ``a`` = exp(A_log), ``lb_s`` = lower_bound*scale.

    softplus is computed in the numerically stable form
    ``max(x, 0) + log(1 + exp(-|x|))`` so the exp argument is never positive.
    """
    if USE_LB:
        # Safe gate variant: lower_bound * sigmoid(exp(A_log) * x)
        return lb_s / (1.0 + tl.exp(-(a * x)))
    else:
        sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        return -a * sp


@triton.autotune(
    configs=[
        _cfg(1, 1, 1, 128, UNROLL=4),
        _cfg(1, 1, 1, 128, UNROLL=2),
        _cfg(8, 1, 1, 16, UNROLL=4),
        _cfg(8, 1, 1, 32, UNROLL=4),
        _cfg(4, 1, 1, 16, UNROLL=4),
        # Tiled small-tile regimes.
        _cfg(2, 1, 1, 64),
        _cfg(8, 1, 2, 16),
        _cfg(4, 1, 2, 16),
        _cfg(8, 1, 1, 32),
        _cfg(8, 4, 2, 32),
        _cfg(16, 4, 2, 32),
        # Full-chunk tiles: fewer programs doing more work — best for short T
        # where per-program setup dominates.
        _cfg(32, 4, 2, 64),
        _cfg(64, 4, 1, 64),
        _cfg(64, 8, 2, 64),
        _cfg(64, 4, 2, 32),
    ],
    key=["T", "BT", "S"],
)
@triton.jit
def _kda_gate_chunk_cumsum_kernel(
    g_ptr,
    dtb_ptr,
    al_ptr,
    o_ptr,
    T,
    H,
    S,
    scale,
    lower_bound,
    BT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
    UNROLL: tl.constexpr,
    EVEN_S: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LB: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_s = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H

    # g/out are contiguous [B, T, H, S]; fix (b, h) and stride rows by H*S.
    row_stride = H * S
    base = g_ptr + b * T * row_stride + h * S + pid_s * BLOCK_S
    obase = o_ptr + b * T * row_stride + h * S + pid_s * BLOCK_S

    s_off = tl.arange(0, BLOCK_S)
    if EVEN_S:
        s_off = tl.max_contiguous(tl.multiple_of(s_off, BLOCK_S), BLOCK_S)
    s_mask = s_off < S - pid_s * BLOCK_S

    a = tl.exp(tl.load(al_ptr + h).to(tl.float32))
    lb_s = lower_bound * scale
    if not USE_LB:
        a = a * scale
    if HAS_BIAS:
        bias = tl.load(
            dtb_ptr + h * S + pid_s * BLOCK_S + s_off, mask=s_mask, other=0.0
        ).to(tl.float32)

    carry = tl.zeros([BLOCK_S], dtype=tl.float32)
    if BLOCK_T == 1:
        t0 = pid_c * BT
        end = tl.minimum(t0 + BT, T)
        if UNROLL == 1:
            for i in tl.range(0, BT):
                t_glob = t0 + i
                m = s_mask & (t_glob < end)
                x = tl.load(
                    base + t_glob * row_stride + s_off,
                    mask=m,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                if HAS_BIAS:
                    x = x + bias
                c = carry + _gate(x, a, lb_s, USE_LB)
                tl.store(
                    obase + t_glob * row_stride + s_off,
                    c,
                    mask=m,
                    eviction_policy="evict_first",
                )
                carry = c
        else:
            for i in tl.range(0, BT, UNROLL):
                tb = t0 + i
                x0 = tl.load(
                    base + (tb + 0) * row_stride + s_off,
                    mask=s_mask & (tb + 0 < end),
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                x1 = tl.load(
                    base + (tb + 1) * row_stride + s_off,
                    mask=s_mask & (tb + 1 < end),
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                if UNROLL == 4:
                    x2 = tl.load(
                        base + (tb + 2) * row_stride + s_off,
                        mask=s_mask & (tb + 2 < end),
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    x3 = tl.load(
                        base + (tb + 3) * row_stride + s_off,
                        mask=s_mask & (tb + 3 < end),
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                if HAS_BIAS:
                    x0 = x0 + bias
                    x1 = x1 + bias
                    if UNROLL == 4:
                        x2 = x2 + bias
                        x3 = x3 + bias
                c = carry + _gate(x0, a, lb_s, USE_LB)
                tl.store(
                    obase + (tb + 0) * row_stride + s_off,
                    c,
                    mask=s_mask & (tb + 0 < end),
                    eviction_policy="evict_first",
                )
                carry = c
                c = carry + _gate(x1, a, lb_s, USE_LB)
                tl.store(
                    obase + (tb + 1) * row_stride + s_off,
                    c,
                    mask=s_mask & (tb + 1 < end),
                    eviction_policy="evict_first",
                )
                carry = c
                if UNROLL == 4:
                    c = carry + _gate(x2, a, lb_s, USE_LB)
                    tl.store(
                        obase + (tb + 2) * row_stride + s_off,
                        c,
                        mask=s_mask & (tb + 2 < end),
                        eviction_policy="evict_first",
                    )
                    carry = c
                    c = carry + _gate(x3, a, lb_s, USE_LB)
                    tl.store(
                        obase + (tb + 3) * row_stride + s_off,
                        c,
                        mask=s_mask & (tb + 3 < end),
                        eviction_policy="evict_first",
                    )
                    carry = c
    else:
        # Tiled regime: a [BLOCK_T, BLOCK_S] tile per step, cumsum over the
        # time axis, tail carried into the next tile.
        t_in = tl.arange(0, BLOCK_T)
        for i in tl.range(0, BT, BLOCK_T):
            t_glob = pid_c * BT + i + t_in
            m = (t_glob < T)[:, None] & s_mask[None, :]
            ptrs = base + t_glob[:, None] * row_stride + s_off[None, :]
            x = tl.load(
                ptrs, mask=m, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            if HAS_BIAS:
                x = x + bias[None, :]
            gate = _gate(x, a, lb_s, USE_LB)
            c = tl.cumsum(gate, axis=0) + carry[None, :]
            tl.store(
                obase + t_glob[:, None] * row_stride + s_off[None, :],
                c,
                mask=m,
                eviction_policy="evict_first",
            )
            # Masked-out rows only ever occur in the final (short) chunk, so
            # the carry they pollute is never consumed.
            carry = carry + tl.sum(gate, axis=0)


def kda_gate_chunk_cumsum(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    b, t, h, s = g.shape
    out = torch.empty((b, t, h, s), dtype=torch.float32, device=g.device)
    if t == 0 or b == 0:
        return out

    gc = g if g.is_contiguous() else g.contiguous()
    al = A_log if A_log.is_contiguous() else A_log.contiguous()
    dtb = dt_bias
    if dtb is not None and not dtb.is_contiguous():
        dtb = dtb.contiguous()

    nt = triton.cdiv(t, chunk_size)

    def grid(META):
        return (nt, b * h, triton.cdiv(s, META["BLOCK_S"]))

    _kda_gate_chunk_cumsum_kernel[grid](
        gc,
        dtb,
        al,
        out,
        t,
        h,
        s,
        float(scale) if scale is not None else 1.0,
        float(lower_bound) if lower_bound is not None else 0.0,
        BT=chunk_size,
        EVEN_S=(s & (s - 1)) == 0,
        HAS_BIAS=dt_bias is not None,
        USE_LB=lower_bound is not None,
    )
    return out


__all__ = ["kda_gate_chunk_cumsum"]
