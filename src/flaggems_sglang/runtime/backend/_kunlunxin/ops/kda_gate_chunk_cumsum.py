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
def _kda_act_ea_flat_nomask_kernel(
    g_ptr,
    a_log_ptr,
    sp_ptr,
    ea_ptr,
    scale,
    S: tl.constexpr,
    HS: tl.constexpr,
    BLOCK: tl.constexpr,
    EBLK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        eoffs = tl.arange(0, EBLK)
        ea = -tl.exp(tl.load(a_log_ptr + eoffs // S))
        if HAS_SCALE:
            ea = ea * scale
        tl.store(ea_ptr + eoffs, ea)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(g_ptr + offs).to(tl.float32)
    # sp(x) = max(x,0) + log1p(exp(-|x|)): branch-free, numerically stable,
    # matches torch's softplus to fp32 rounding.
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    tl.store(sp_ptr + offs, sp)


@triton.jit
def _kda_act_ea_flat_kernel(
    g_ptr,
    a_log_ptr,
    sp_ptr,
    ea_ptr,
    numel,
    scale,
    S: tl.constexpr,
    HS: tl.constexpr,
    BLOCK: tl.constexpr,
    EBLK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    # Flat 1D softplus with a tail mask (numel % BLOCK != 0). Both loads
    # and stores are masked: the last CTA reads past the end of g.
    pid = tl.program_id(0)
    if pid == 0:
        eoffs = tl.arange(0, EBLK)
        ea = -tl.exp(tl.load(a_log_ptr + eoffs // S))
        if HAS_SCALE:
            ea = ea * scale
        tl.store(ea_ptr + eoffs, ea)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    tl.store(sp_ptr + offs, sp, mask=mask)


@triton.jit
def _kda_sp_row_kernel(
    g_ptr,
    dt_bias_ptr,
    sp_ptr,
    hs,
    HAS_BIAS: tl.constexpr,
    CBLK: tl.constexpr,
):
    row = tl.program_id(0)
    pid_c = tl.program_id(1)
    c = pid_c * CBLK + tl.arange(0, CBLK)
    mask = c < hs
    base = row * hs + c
    x = tl.load(g_ptr + base, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        x = x + tl.load(dt_bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-ax))
    tl.store(sp_ptr + base, sp, mask=mask)


@triton.jit
def _kda_gate_row_kernel(
    g_ptr,
    ea_ptr,
    dt_bias_ptr,
    out_ptr,
    hs,
    lower_bound,
    HAS_BIAS: tl.constexpr,
    CBLK: tl.constexpr,
):
    row = tl.program_id(0)
    pid_c = tl.program_id(1)
    c = pid_c * CBLK + tl.arange(0, CBLK)
    mask = c < hs
    base = row * hs + c
    x = tl.load(g_ptr + base, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        x = x + tl.load(dt_bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    ea = tl.load(ea_ptr + c, mask=mask, other=0.0)
    # tl.sigmoid (rather than the manual 1/(1+exp(-y))) matches the
    # reference's safe-gate form; this backend lowers it correctly here.
    gate = lower_bound * tl.sigmoid(ea * x)
    tl.store(out_ptr + base, gate, mask=mask)


@triton.jit
def _kda_ea_kernel(
    a_log_ptr,
    ea_ptr,
    scale,
    S: tl.constexpr,
    HB: tl.constexpr,
    NEG: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * HB + tl.arange(0, HB)
    e = tl.exp(tl.load(a_log_ptr + (c // S)))
    if NEG:
        e = -e
    if HAS_SCALE:
        e = e * scale
    tl.store(ea_ptr + c, e)


@triton.jit
def _kda_dot_cumsum_nomask_kernel(
    sp_ptr,
    ea_ptr,
    out_ptr,
    scale,
    HS: tl.constexpr,
    TP: tl.constexpr,
    CB: tl.constexpr,
    HAS_E: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_r = tl.program_id(1)
    toks = tl.arange(0, TP)
    c = pid_r * CB + tl.arange(0, CB)
    base = (pid * TP + toks[:, None]) * HS + c[None, :]
    xv = tl.load(sp_ptr + base)
    Lv = tl.where(toks[:, None] >= toks[None, :], 1.0, 0.0)
    out = tl.dot(Lv, xv, input_precision="ieee")
    if HAS_E:
        ea = tl.load(ea_ptr + c)
        out = out * ea[None, :]
    if HAS_SCALE:
        out = out * scale
    tl.store(out_ptr + base, out)


@triton.jit
def _kda_dot_cumsum_kernel(
    sp_ptr,
    ea_ptr,
    out_ptr,
    t,
    scale,
    HS: tl.constexpr,
    TP: tl.constexpr,
    CB: tl.constexpr,
    HAS_E: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_r = tl.program_id(1)
    chunks_per_b = tl.cdiv(t, TP)
    pid_b = pid // chunks_per_b
    pid_c = pid % chunks_per_b

    toks = tl.arange(0, TP)
    c = pid_r * CB + tl.arange(0, CB)
    tok_global = pid_c * TP + toks
    mask_t = tok_global < t
    base = ((pid_b * t + tok_global[:, None]).to(tl.int64)) * HS + c[None, :]
    m2 = mask_t[:, None]

    xv = tl.load(sp_ptr + base, mask=m2, other=0.0)
    Lv = tl.where(toks[:, None] >= toks[None, :], 1.0, 0.0)
    out = tl.dot(Lv, xv, input_precision="ieee")
    if HAS_E:
        ea = tl.load(ea_ptr + c)
        out = out * ea[None, :]
    if HAS_SCALE:
        out = out * scale
    tl.store(out_ptr + base, out, mask=m2)


@triton.jit
def _kda_gate_cumsum_generic_kernel(
    g_ptr,
    a_log_ptr,
    dt_bias_ptr,
    out_ptr,
    t,
    hs,
    s,
    scale,
    lower_bound,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_LB: tl.constexpr,
    TP: tl.constexpr,
    SB: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    chunks_per_b = tl.cdiv(t, TP)
    pid_b = pid // chunks_per_b
    pid_c = pid % chunks_per_b

    toks = tl.arange(0, TP)
    cols = tl.arange(0, SB)
    mask_c = cols < s
    tok_g = pid_c * TP + toks
    mask_t = tok_g < t
    base = (
        ((pid_b * t + tok_g[:, None]).to(tl.int64)) * hs
        + pid_h * s
        + cols[None, :]
    )
    m2 = mask_t[:, None] & mask_c[None, :]

    x = tl.load(g_ptr + base, mask=m2, other=0.0).to(tl.float32)
    exp_a = tl.exp(tl.load(a_log_ptr + pid_h).to(tl.float32))
    if HAS_BIAS:
        x = (
            x
            + tl.load(
                dt_bias_ptr + pid_h * s + cols, mask=mask_c, other=0.0
            ).to(tl.float32)[None, :]
        )
    if HAS_LB:
        gate = lower_bound * tl.sigmoid(exp_a * x)
    else:
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        gate = -exp_a * sp
    if HAS_SCALE:
        gate = gate * scale
    gate = tl.where(m2, gate, 0.0)

    cum = tl.cumsum(gate, axis=0)
    tl.store(out_ptr + base, cum, mask=m2)


def kda_gate_chunk_cumsum(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    b, t, h, s = g.shape
    out = torch.empty((b, t, h, s), device=g.device, dtype=torch.float32)
    if b == 0 or t == 0 or h == 0 or s == 0:
        return out

    TP = max(16, triton.next_power_of_2(chunk_size))
    hs = h * s
    numel = b * t * hs

    if (
        s == triton.next_power_of_2(s)
        and TP <= 64
        and hs >= 16
        and hs <= 8192
        and numel < 2**31
        and (g.is_contiguous() or g.contiguous().data_ptr() == g.data_ptr())
    ):
        CB = min(1024, hs)
        while hs % CB != 0:
            CB //= 2
        if CB >= 16:
            gc = g if g.is_contiguous() else g.contiguous()
            a_log_flat = A_log.reshape(-1)
            sp32 = torch.empty(numel, device=g.device, dtype=torch.float32)
            ea = torch.empty(hs, device=g.device, dtype=torch.float32)
            eblk = triton.next_power_of_2(hs)
            scale_v = scale if scale is not None else 1.0

            has_lb = lower_bound is not None
            has_e = not has_lb
            has_scale = scale is not None
            if dt_bias is not None:
                bias_flat = dt_bias.reshape(-1).contiguous()
            if has_lb:
                _kda_ea_kernel[(hs // min(1024, eblk),)](
                    a_log_flat,
                    ea,
                    1.0,
                    S=s,
                    HB=min(1024, eblk),
                    NEG=False,
                    HAS_SCALE=False,
                    num_warps=4,
                )
                CBLK = min(512, triton.next_power_of_2(hs))
                _kda_gate_row_kernel[(b * t, triton.cdiv(hs, CBLK))](
                    gc,
                    ea,
                    (
                        bias_flat if dt_bias is not None else gc
                    ),  # dummy when unused
                    sp32,
                    hs,
                    lower_bound,
                    HAS_BIAS=dt_bias is not None,
                    CBLK=CBLK,
                    num_warps=4,
                )
            elif dt_bias is None:
                if numel % 8192 == 0:
                    _kda_act_ea_flat_nomask_kernel[(numel // 8192,)](
                        gc,
                        a_log_flat,
                        sp32,
                        ea,
                        scale_v,
                        S=s,
                        HS=hs,
                        BLOCK=8192,
                        EBLK=eblk,
                        HAS_SCALE=has_scale,
                        num_warps=4,
                    )
                else:
                    _kda_act_ea_flat_kernel[(triton.cdiv(numel, 8192),)](
                        gc,
                        a_log_flat,
                        sp32,
                        ea,
                        numel,
                        scale_v,
                        S=s,
                        HS=hs,
                        BLOCK=8192,
                        EBLK=eblk,
                        HAS_SCALE=has_scale,
                        num_warps=4,
                    )
            else:
                # dt_bias path: ea (with sign/scale folded) via the
                # dedicated kernel; softplus by the row kernel.
                _kda_ea_kernel[(hs // min(1024, eblk),)](
                    a_log_flat,
                    ea,
                    scale_v,
                    S=s,
                    HB=min(1024, eblk),
                    NEG=True,
                    HAS_SCALE=has_scale,
                    num_warps=4,
                )
                CBLK = min(512, triton.next_power_of_2(hs))
                _kda_sp_row_kernel[(b * t, triton.cdiv(hs, CBLK))](
                    gc,
                    bias_flat,
                    sp32,
                    hs,
                    HAS_BIAS=True,
                    CBLK=CBLK,
                    num_warps=4,
                )

            if t % TP == 0:
                _kda_dot_cumsum_nomask_kernel[(b * (t // TP), hs // CB)](
                    sp32,
                    ea,
                    out,
                    scale_v,
                    HS=hs,
                    TP=TP,
                    CB=CB,
                    HAS_E=has_e,
                    HAS_SCALE=has_scale and not has_e,
                    num_warps=4,
                    num_stages=2,
                )
            else:
                _kda_dot_cumsum_kernel[(b * triton.cdiv(t, TP), hs // CB)](
                    sp32,
                    ea,
                    out,
                    t,
                    scale_v,
                    HS=hs,
                    TP=TP,
                    CB=CB,
                    HAS_E=has_e,
                    HAS_SCALE=has_scale and not has_e,
                    num_warps=4,
                    num_stages=2,
                )
            return out

    # Generic path (s not a power of two, or chunk_size above the dot
    # tile limit): correct but slower.
    grid = (b * triton.cdiv(t, TP), h)
    _kda_gate_cumsum_generic_kernel[grid](
        g.contiguous(),
        A_log.contiguous(),
        (
            dt_bias.reshape(-1).contiguous() if dt_bias is not None else g
        ),  # dummy ptr when unused
        out.view(b * t, hs),
        t,
        hs,
        s,
        scale if scale is not None else 1.0,
        lower_bound if lower_bound is not None else 0.0,
        HAS_BIAS=dt_bias is not None,
        HAS_SCALE=scale is not None,
        HAS_LB=lower_bound is not None,
        TP=TP,
        SB=triton.next_power_of_2(s),
        num_warps=4,
    )
    return out


__all__ = ["kda_gate_chunk_cumsum"]
