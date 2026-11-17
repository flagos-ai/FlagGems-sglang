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
    magnitude = tl.minimum(v, 448.0)
    bits = magnitude.to(tl.int32, bitcast=True)
    bias = tl.maximum(bits & 2139095040, 1015021568) + 167772160
    biased = magnitude + bias.to(tl.float32, bitcast=True)
    code = biased.to(tl.int32, bitcast=True) - bias + (bias >> 20) - 1128
    code = tl.where(v != v, 127, code)
    return code.to(tl.uint8)


@triton.jit
def _activate(
    gate, up, ALPHA: tl.constexpr, BETA: tl.constexpr, LIMIT: tl.constexpr
):
    if LIMIT is not None:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    biased = up + BETA
    sign = (
        gate.to(tl.int32, bitcast=True) ^ biased.to(tl.int32, bitcast=True)
    ) >> 24 & 128
    activated = gate * (1.0 / (1.0 + tl.exp(-(ALPHA * gate)))) * biased
    return (activated, sign)


@triton.jit
def _rounded_natural_log(x):
    bits = x.to(tl.int32, bitcast=True)
    e = (bits >> 23) - 127
    m = (bits & 8388607 | 1065353216).to(tl.float32, bitcast=True)
    upper = m > 1.4142135623730951
    m = tl.where(upper, m * 0.5, m)
    e = e + upper.to(tl.int32)
    y = tl.div_rn(m - 1.0, m + 1.0)
    z = y * y
    p = 1.0 / 3.0
    logm = 2.0 * y * (1.0 + z * p)
    ef = e.to(tl.float32)
    return tl.fma(ef, 0.6931471824645996, logm)


@triton.jit
def _scale(activated):
    maximum = tl.maximum(tl.max(tl.abs(activated), 1), 1e-30)
    ratio = maximum / 448.0
    exponent = tl.minimum(
        tl.maximum(
            tl.ceil(_rounded_natural_log(ratio) * 1.4426950408889634) + 127.0,
            0.0,
        ),
        254.0,
    ).to(tl.int32)
    inverse_bits = tl.where(
        exponent == 254, 4194304, (254 - exponent).to(tl.int32) << 23
    )
    inverse = inverse_bits.to(tl.int32).to(tl.float32, bitcast=True)
    return (inverse, exponent)


@triton.jit(do_not_specialize=["Q", "S"])
def _kernel(X, Q, S, META: tl.constexpr):
    N: tl.constexpr = META[0]
    G: tl.constexpr = META[1]
    SX: tl.constexpr = META[2]
    CX: tl.constexpr = META[3]
    ALPHA: tl.constexpr = META[4]
    BETA: tl.constexpr = META[5]
    LIMIT: tl.constexpr = META[6]
    R: tl.constexpr = META[7]
    col = tl.arange(0, 32)
    for tile in range(tl.program_id(0), tl.cdiv(G, R), tl.num_programs(0)):
        group = tile * R + tl.arange(0, R)
        row = group // (N // 32)
        c = (group % (N // 32))[:, None] * 32 + col[None, :]
        address = row[:, None] * SX + c * CX
        if G % R == 0:
            gate = tl.load(X + address).to(tl.float32)
            up = tl.load(X + address + N * CX).to(tl.float32)
        else:
            gate = tl.load(X + address, group[:, None] < G, 0).to(tl.float32)
            up = tl.load(X + address + N * CX, group[:, None] < G, 0).to(
                tl.float32
            )
        activated, sign = _activate(gate, up, ALPHA, BETA, LIMIT)
        inverse, exponent = _scale(activated)
        v = (activated * inverse[:, None]).to(tl.float32)
        out = group[:, None] * 32 + col[None, :]
        encoded = _encode(tl.abs(v)) | sign.to(tl.uint8)
        tl.store(Q + out, encoded, group[:, None] < G)
        tl.store(S + group, exponent.to(tl.uint8), group < G)


@triton.jit(do_not_specialize=["Q", "S"])
def _contiguous(X, Q, S, META: tl.constexpr):
    N: tl.constexpr = META[0]
    ALPHA: tl.constexpr = META[1]
    BETA: tl.constexpr = META[2]
    LIMIT: tl.constexpr = META[3]
    R: tl.constexpr = META[4]
    col = tl.arange(0, 32)
    row = tl.program_id(1)
    local_group = tl.program_id(0) * R + tl.arange(0, R)
    group = row * (N // 32) + local_group
    address = row * (2 * N) + local_group[:, None] * 32 + col[None, :]
    gate = tl.load(X + address, offset_state_policy="continuous").to(
        tl.float32
    )
    up = tl.load(X + address + N, offset_state_policy="continuous").to(
        tl.float32
    )
    activated, sign = _activate(gate, up, ALPHA, BETA, LIMIT)
    inverse, exponent = _scale(activated)
    v = (activated * inverse[:, None]).to(tl.float32)
    out = group[:, None] * 32 + col[None, :]
    encoded = _encode(tl.abs(v)) | sign.to(tl.uint8)
    tl.store(Q + out, encoded, offset_state_policy="continuous")
    tl.store(
        S + group, exponent.to(tl.uint8), offset_state_policy="continuous"
    )


def swiglu_oai_mxfp8_quant(gate_up, alpha, beta, limit):
    shape = gate_up.shape
    width = shape[-1]
    n = width // 2
    if n % 32:
        raise ValueError("half of the last dimension must be divisible by 32")
    x = gate_up if gate_up.ndim == 2 else gate_up.reshape(-1, width)
    groups = x.shape[0] * (n // 32)
    q = x.new_empty((*shape[:-1], n), dtype=torch.float8_e4m3fn)
    scales = x.new_empty((*shape[:-1], n // 32), dtype=torch.uint8)
    if groups:
        q_bytes = q.view(torch.uint8)
        per_row = n // 32
        rows = 64 if per_row % 64 == 0 else 16
        if x.stride(0) == width and x.stride(1) == 1 and (per_row % rows == 0):
            _contiguous.run(
                x,
                q_bytes,
                scales,
                (n, alpha, beta, limit, rows),
                num_warps=4,
                enable_fp_fusion=False,
                isCloseUnrollControl=True,
                isCloseMemoryAsync=False,
                grid=(per_row // rows, x.shape[0]),
                warmup=False,
            )
        else:
            rows = 4
            _kernel.run(
                x,
                q_bytes,
                scales,
                (
                    n,
                    groups,
                    x.stride(0),
                    x.stride(1),
                    alpha,
                    beta,
                    limit,
                    rows,
                ),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(triton.cdiv(groups, rows),),
                warmup=False,
            )
    return (q, scales)


__all__ = ["swiglu_oai_mxfp8_quant"]
