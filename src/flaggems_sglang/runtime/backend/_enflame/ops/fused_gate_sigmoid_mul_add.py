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

# DSM budget: tile elements (R * BC) capped at 65536 keeps the compiled
# dsm_memory_size under the 917504-byte hardware limit on Libra (gcu400).
# Phase 2 (scale * s + f, store) runs in the tensors' native dtype with the
# per-row gate cast once (sc): bf16 fma avoids two per-element f32 promotes
# (measured -2..-5% on the n>=64 band; tolerance 1.5e-2 absorbs the 2-ulp
# rounding, diag bad=0 vs fp32 reference).
_ELEM_BUDGET = 65536
# Row-batch target: ~32 programs for mid n; capped at 32 rows per program
# (measured optimum at n=4096 keeps grid>=48 with the widest column block).
_GRID_TARGET = 32
_ROW_CAP = 32


@triton.jit
def _row(
    X,
    W,
    S,
    F,
    Out,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    WS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    BC: tl.constexpr,
    NFULL: tl.constexpr,
    TAIL: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(NFULL):
        col = k * BC + tl.arange(0, BC)
        x = tl.load(X + row * X0 + col * X1).to(tl.float32)
        w = tl.load(W + col * WS).to(tl.float32)
        acc += tl.sum(x * w, 0)
    if TAIL:
        col = NFULL * BC + tl.arange(0, BC)
        cm = col < D
        x = tl.load(X + row * X0 + col * X1, cm, other=0.0).to(tl.float32)
        w = tl.load(W + col * WS, cm, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, 0)
    scale = 1.0 / (1.0 + tl.exp(-acc))
    for k in range(NFULL):
        col = k * BC + tl.arange(0, BC)
        s = tl.load(S + row * S0 + col * S1).to(tl.float32)
        f = tl.load(F + row * F0 + col * F1).to(tl.float32)
        tl.store(Out + row * D + col, f + scale * s)
    if TAIL:
        col = NFULL * BC + tl.arange(0, BC)
        cm = col < D
        s = tl.load(S + row * S0 + col * S1, cm, other=0.0).to(tl.float32)
        f = tl.load(F + row * F0 + col * F1, cm, other=0.0).to(tl.float32)
        tl.store(Out + row * D + col, f + scale * s, cm)


@triton.jit
def _tile(
    X,
    W,
    S,
    F,
    Out,
    N: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    WS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
    NFULL: tl.constexpr,
    TAIL: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * R + tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)
    if EVEN:
        for k in range(NFULL):
            col = k * BC + tl.arange(0, BC)
            x = tl.load(X + row[:, None] * X0 + col[None, :] * X1).to(tl.float32)
            w = tl.load(W + col * WS).to(tl.float32)
            acc += tl.sum(x * w[None, :], 1)
        if TAIL:
            col = NFULL * BC + tl.arange(0, BC)
            cm = col < D
            x = tl.load(
                X + row[:, None] * X0 + col[None, :] * X1, cm[None, :], other=0.0
            ).to(tl.float32)
            w = tl.load(W + col * WS, cm, other=0.0).to(tl.float32)
            acc += tl.sum(x * w[None, :], 1)
        scale = 1.0 / (1.0 + tl.exp(-acc))
        sc = scale.to(Out.dtype.element_ty)
        for k in range(NFULL):
            col = k * BC + tl.arange(0, BC)
            s = tl.load(S + row[:, None] * S0 + col[None, :] * S1)
            f = tl.load(F + row[:, None] * F0 + col[None, :] * F1)
            tl.store(
                Out + row[:, None] * D + col[None, :], f + sc[:, None] * s
            )
        if TAIL:
            col = NFULL * BC + tl.arange(0, BC)
            cm = col < D
            s = tl.load(
                S + row[:, None] * S0 + col[None, :] * S1, cm[None, :], other=0.0
            )
            f = tl.load(
                F + row[:, None] * F0 + col[None, :] * F1, cm[None, :], other=0.0
            )
            tl.store(
                Out + row[:, None] * D + col[None, :],
                f + sc[:, None] * s,
                cm[None, :],
            )
    else:
        rm = row < N
        for k in range(NFULL):
            col = k * BC + tl.arange(0, BC)
            x = tl.load(
                X + row[:, None] * X0 + col[None, :] * X1, rm[:, None], other=0.0
            ).to(tl.float32)
            w = tl.load(W + col * WS).to(tl.float32)
            acc += tl.sum(x * w[None, :], 1)
        if TAIL:
            col = NFULL * BC + tl.arange(0, BC)
            cm = col < D
            m = rm[:, None] & cm[None, :]
            x = tl.load(X + row[:, None] * X0 + col[None, :] * X1, m, other=0.0).to(
                tl.float32
            )
            w = tl.load(W + col * WS, cm, other=0.0).to(tl.float32)
            acc += tl.sum(x * w[None, :], 1)
        scale = 1.0 / (1.0 + tl.exp(-acc))
        sc = scale.to(Out.dtype.element_ty)
        for k in range(NFULL):
            col = k * BC + tl.arange(0, BC)
            s = tl.load(
                S + row[:, None] * S0 + col[None, :] * S1, rm[:, None], other=0.0
            )
            f = tl.load(
                F + row[:, None] * F0 + col[None, :] * F1, rm[:, None], other=0.0
            )
            tl.store(
                Out + row[:, None] * D + col[None, :],
                f + sc[:, None] * s,
                rm[:, None],
            )
        if TAIL:
            col = NFULL * BC + tl.arange(0, BC)
            cm = col < D
            m = rm[:, None] & cm[None, :]
            s = tl.load(S + row[:, None] * S0 + col[None, :] * S1, m, other=0.0)
            f = tl.load(F + row[:, None] * F0 + col[None, :] * F1, m, other=0.0)
            tl.store(
                Out + row[:, None] * D + col[None, :], f + sc[:, None] * s, m
            )


def _pow2(v):
    return 1 << max(0, (v - 1).bit_length()) if v > 1 else 1


def fused_gate_sigmoid_mul_add(
    hidden_states, gate_weight, shared_output, final_hidden_states
):
    n, d = hidden_states.shape
    output = torch.empty(
        (n, d),
        dtype=final_hidden_states.dtype,
        device=final_hidden_states.device,
    )
    if not n or not d:
        return output
    b = _pow2(d)
    if n <= _GRID_TARGET:
        bc = b if b <= _ELEM_BUDGET else _ELEM_BUDGET
        nfull = d // bc
        _row.run(
            hidden_states,
            gate_weight,
            shared_output,
            final_hidden_states,
            output,
            d,
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_weight.stride(0),
            shared_output.stride(0),
            shared_output.stride(1),
            final_hidden_states.stride(0),
            final_hidden_states.stride(1),
            bc,
            nfull,
            d - nfull * bc,
            grid=(n,),
            warmup=False,
            num_warps=16,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return output
    r = _pow2((n + _GRID_TARGET - 1) // _GRID_TARGET)
    if r > _ROW_CAP:
        r = _ROW_CAP
    bc = b if b <= _ELEM_BUDGET // r else _ELEM_BUDGET // r
    if bc < 1:
        bc = 1
    r = _ELEM_BUDGET // bc if _ELEM_BUDGET // bc < r else r
    nfull = d // bc
    _tile.run(
        hidden_states,
        gate_weight,
        shared_output,
        final_hidden_states,
        output,
        n,
        d,
        hidden_states.stride(0),
        hidden_states.stride(1),
        gate_weight.stride(0),
        shared_output.stride(0),
        shared_output.stride(1),
        final_hidden_states.stride(0),
        final_hidden_states.stride(1),
        r,
        bc,
        nfull,
        d - nfull * bc,
        n % r == 0,
        grid=((n + r - 1) // r,),
        warmup=False,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["fused_gate_sigmoid_mul_add"]
