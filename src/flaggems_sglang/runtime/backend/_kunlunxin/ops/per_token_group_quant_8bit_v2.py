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

import triton
import triton.language as tl


@triton.jit
def _max_nan(a, b):
    ai = tl.cast(a, tl.float32).to(tl.int32, bitcast=True) & 2147483647
    bi = tl.cast(b, tl.float32).to(tl.int32, bitcast=True) & 2147483647
    bi = tl.where((b < 0) & (bi <= 2139095040), 0, bi)
    return tl.maximum(ai, bi).to(tl.float32, bitcast=True)


@triton.jit
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits & 2147483647
    sign = bits >> 24 & 128
    exponent = tl.maximum(mag >> 23, 121)
    magic = exponent + 20 << 23
    rounded = (tl.abs(v) + magic.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    code = (rounded & 31) + (exponent - 121 << 3)
    return tl.where(
        mag > 2139095040, 127 | sign, tl.minimum(code, 127) | sign
    ).to(tl.uint8)


@triton.jit
def _quant(X, Q: tl.pointer_type(tl.uint8), S, P: tl.constexpr):
    N: tl.constexpr = P[0]
    H: tl.constexpr = P[1]
    G: tl.constexpr = P[2]
    X0: tl.constexpr = P[3]
    X1: tl.constexpr = P[4]
    Q0: tl.constexpr = P[5]
    Q1: tl.constexpr = P[6]
    S0: tl.constexpr = P[7]
    S1: tl.constexpr = P[8]
    EPS: tl.constexpr = P[9]
    LO: tl.constexpr = P[10]
    HI: tl.constexpr = P[11]
    BG: tl.constexpr = P[12]
    BR: tl.constexpr = P[13]
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BG)
    ng = H // G
    row = r // ng
    col = (r % ng)[:, None] * G + c[None, :]
    if X0 == H and X1 == 1 and (BG == G) and (N * ng % BR == 0):
        x = tl.load(
            X + r[:, None] * G + c[None, :], offset_state_policy="continuous"
        ).to(tl.float32)
    else:
        x = tl.load(
            X + row[:, None] * X0 + col * X1,
            (r[:, None] < N * ng) & (c[None, :] < G),
            other=0,
        ).to(tl.float32)
    amax = _max_nan(
        tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1).to(
            tl.float32, bitcast=True
        ),
        EPS,
    )
    scale = tl.div_rn(amax, HI)
    scale = tl.where(
        amax.to(tl.int32, bitcast=True) & 2147483647 > 2139095040,
        float("nan"),
        scale,
    )
    if LO == -448.0 and HI == 448.0 and (EPS >= 5.266214694141234e-36):
        q = tl.div_rn(x, scale[:, None])
    else:
        q = tl.minimum(
            tl.maximum(
                tl.div_rn(x, scale[:, None]),
                LO,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            HI,
            propagate_nan=tl.PropagateNan.ALL,
        )
    bits = _encode(q)
    bits = tl.where(
        (scale.to(tl.int32, bitcast=True) & 2147483647 > 2139095040)[:, None],
        127,
        bits,
    )
    if Q0 == H and Q1 == 1 and (BG == G) and (N * ng % BR == 0):
        tl.store(
            Q + r[:, None] * G + c[None, :],
            bits,
            offset_state_policy="continuous",
        )
    else:
        tl.store(
            Q + row[:, None] * Q0 + col * Q1,
            bits,
            (r[:, None] < N * ng) & (c[None, :] < G),
        )
    if S0 == ng and S1 == 1 and (N * ng % BR == 0):
        tl.store(S + r, scale, offset_state_policy="continuous")
    else:
        tl.store(S + row * S0 + r % ng * S1, scale, r < N * ng)


@triton.jit
def _packed(
    X: tl.pointer_type(tl.uint64),
    Q: tl.pointer_type(tl.uint32),
    S,
    P: tl.constexpr,
):
    G: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    BR: tl.constexpr = P[5]
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, G // 4)
    offsets = r[:, None] * (G // 4) + c[None, :]
    packed = tl.load(X + offsets, offset_state_policy="continuous")
    low = packed.to(tl.uint32)
    high = (packed >> 32).to(tl.uint32)
    a = (low << 16).to(tl.float32, bitcast=True)
    b = (low & 4294901760).to(tl.float32, bitcast=True)
    c = (high << 16).to(tl.float32, bitcast=True)
    d = (high & 4294901760).to(tl.float32, bitcast=True)
    magnitude = _max_nan(
        _max_nan(tl.abs(a), tl.abs(b)), _max_nan(tl.abs(c), tl.abs(d))
    )
    amax = _max_nan(
        tl.max(magnitude.to(tl.int32, bitcast=True), 1).to(
            tl.float32, bitcast=True
        ),
        EPS,
    )
    scale = tl.div_rn(amax, HI)
    scale = tl.where(
        amax.to(tl.int32, bitcast=True) & 2147483647 > 2139095040,
        float("nan"),
        scale,
    )
    if LO == -448.0 and HI == 448.0 and (EPS >= 5.266214694141234e-36):
        qa = tl.div_rn(a, scale[:, None])
    else:
        qa = tl.minimum(
            tl.maximum(
                tl.div_rn(a, scale[:, None]),
                LO,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            HI,
            propagate_nan=tl.PropagateNan.ALL,
        )
    if LO == -448.0 and HI == 448.0 and (EPS >= 5.266214694141234e-36):
        qb = tl.div_rn(b, scale[:, None])
    else:
        qb = tl.minimum(
            tl.maximum(
                tl.div_rn(b, scale[:, None]),
                LO,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            HI,
            propagate_nan=tl.PropagateNan.ALL,
        )
    if LO == -448.0 and HI == 448.0 and (EPS >= 5.266214694141234e-36):
        qc = tl.div_rn(c, scale[:, None])
    else:
        qc = tl.minimum(
            tl.maximum(
                tl.div_rn(c, scale[:, None]),
                LO,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            HI,
            propagate_nan=tl.PropagateNan.ALL,
        )
    if LO == -448.0 and HI == 448.0 and (EPS >= 5.266214694141234e-36):
        qd = tl.div_rn(d, scale[:, None])
    else:
        qd = tl.minimum(
            tl.maximum(
                tl.div_rn(d, scale[:, None]),
                LO,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            HI,
            propagate_nan=tl.PropagateNan.ALL,
        )
    result = (
        _encode(qa).to(tl.uint32)
        | _encode(qb).to(tl.uint32) << 8
        | _encode(qc).to(tl.uint32) << 16
        | _encode(qd).to(tl.uint32) << 24
    )
    result = tl.where(
        (scale.to(tl.int32, bitcast=True) & 2147483647 > 2139095040)[:, None],
        2139062143,
        result,
    )
    tl.store(Q + offsets, result, offset_state_policy="continuous")
    tl.store(S + r, scale, offset_state_policy="continuous")


def per_token_group_quant_8bit_v2(
    input, output_q, output_s, group_size, eps, min_8bit, max_8bit
):
    n, h = input.shape
    rows = n * (h // group_size)
    br = 64 if rows >= 256 else 16
    if (
        rows
        and rows % br == 0
        and (group_size >= 4)
        and (group_size & group_size - 1 == 0)
        and (h % group_size == 0)
        and input.is_contiguous()
        and output_q.is_contiguous()
        and output_s.is_contiguous()
        and (input.data_ptr() % 8 == 0)
        and (output_q.data_ptr() % 4 == 0)
    ):
        _packed.run(
            input,
            output_q,
            output_s,
            (rows, group_size, eps, min_8bit, max_8bit, br),
            grid=(rows // br,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    elif rows:
        _quant.run(
            input,
            output_q,
            output_s,
            (
                n,
                h,
                group_size,
                *input.stride(),
                *output_q.stride(),
                *output_s.stride(),
                eps,
                min_8bit,
                max_8bit,
                triton.next_power_of_2(group_size),
                br,
            ),
            grid=(triton.cdiv(rows, br),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (output_q, output_s)


__all__ = ["per_token_group_quant_8bit_v2"]
