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

try:
    from triton.language.extra import libdevice
except ImportError:
    from triton.language.extra.hip import libdevice

import triton
import triton.language as tl


@triton.jit
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    mag = raw & 2147483647
    a = mag.to(tl.float32, bitcast=True)
    exponent = tl.maximum(mag & 2139095040, 1015021568)
    bias = exponent + 167772160
    biased = (a + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    code = biased + (bias >> 20) - 1128
    code = tl.where(mag > 1139277824, 127, code)
    return (code | raw >> 24 & 128).to(tl.uint8)


@triton.jit
def _silu(g):
    den = 1.0 + libdevice.exp(-g)
    q, inv, err, ds, ns = tl.inline_asm_elementwise(
        (
            "v_rcp_f32 $1, $6;\n"
            "v_mul_f32 $0, $5, $1;\n"
            "v_fma_f32 $2, -$6, $0, $5;\n"
            "v_fma_f32 $0, $2, $1, $0;\n"
            "v_cmp_gt_f32 vcc, -87.0, $5;\n"
            "s_cbranch_vccz 2f;\n"
            "v_div_scale_f32 $3, vcc, $6, $6, $5;\n"
            "v_div_scale_f32 $4, vcc, $5, $6, $5;\n"
            "v_rcp_f32 $1, $3;\n"
            "v_mul_f32 $0, $4, $1;\n"
            "v_fma_f32 $2, -$3, $0, $4;\n"
            "v_div_fmas_f32 $0, $2, $1, $0;\n"
            "2:\n"
            "v_div_fixup_f32 $0, $0, $6, $5;"
        ),
        "=&v,=&v,=&v,=&v,=&v,v,v,~{vcc}",
        [g, den],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )
    return q


@triton.jit
def _guarded_den(g):
    den, y, err, exponent = tl.inline_asm_elementwise(
        (
            "\n"
            "v_mul_f32 $1, 0xbfb8aa3b, $4;\n"
            "v_exp_f32 $0, $1;\n"
            "v_cmp_nlt_f32 vcc, -16.0, $4;\n"
            "s_cbranch_vccz 7f;\n"
            "v_mov_b32 $2, 0x3fb8aa3b;\n"
            "v_fma_f32 $2, -$4, $2, -$1;\n"
            "v_mov_b32 $3, 0x32a5705f;\n"
            "v_fma_f32 $2, -$4, $3, $2;\n"
            "v_rndne_f32 $3, $1;\n"
            "v_sub_f32 $1, $1, $3;\n"
            "v_add_f32 $1, $2, $1;\n"
            "v_exp_f32 $0, $1;\n"
            "v_cvt_i32_f32 $3, $3;\n"
            "v_ldexp_f32 $0, $0, $3;\n"
            "v_sub_f32 $3, 0, $4;\n"
            "v_cmp_gt_f32 vcc, 0xc2ce8ed0, $3;\n"
            "v_cndmask_b32 $0, $0, 0, vcc;\n"
            "v_cmp_lt_f32 vcc, 0x42b17218, $3;\n"
            "v_mov_b32 $2, 0x7f800000;\n"
            "v_cndmask_b32 $0, $0, $2, vcc;\n"
            "7:\n"
            "v_add_f32 $0, 1.0, $0;\n"
        ),
        "=&v,=&v,=&v,=&v,v,~{vcc}",
        [g],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )
    return den


@triton.jit
def _fast_silu(g):
    den = _guarded_den(g)
    q, inv, err, ds, ns = tl.inline_asm_elementwise(
        (
            "v_rcp_f32 $1, $6;\n"
            "v_mul_f32 $0, $5, $1;\n"
            "v_cmp_nlt_f32 vcc, -64.0, $5;\n"
            "s_cbranch_vccz 9f;\n"
            "v_rcp_f32 $1, $6;\n"
            "v_mul_f32 $0, $5, $1;\n"
            "v_fma_f32 $2, -$6, $0, $5;\n"
            "v_fma_f32 $0, $2, $1, $0;\n"
            "v_cmp_gt_f32 vcc, -87.0, $5;\n"
            "s_cbranch_vccz 2f;\n"
            "v_div_scale_f32 $3, vcc, $6, $6, $5;\n"
            "v_div_scale_f32 $4, vcc, $5, $6, $5;\n"
            "v_rcp_f32 $1, $3;\n"
            "v_mul_f32 $0, $4, $1;\n"
            "v_fma_f32 $2, -$3, $0, $4;\n"
            "v_div_fmas_f32 $0, $2, $1, $0;\n"
            "2:\n"
            "v_div_fixup_f32 $0, $0, $6, $5;\n"
            "9:"
        ),
        "=&v,=&v,=&v,=&v,=&v,v,v,~{vcc}",
        [g, den],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )
    return q


@triton.jit
def _value(g, u, s):
    sr = s.to(tl.int32, bitcast=True)
    sm = sr & 2147483647
    if (sm & 8388607 == 0) & (sm >= 528482304) & (sm <= 1602224128):
        v = _fast_silu(g) * u
        v = v * (2130706432 - sm | sr & -2147483648).to(
            tl.float32, bitcast=True
        )
    else:
        v = _silu(g) * u
        if (sm & 8388607 == 0) & (sm >= 8388608) & (sm <= 2122317824):
            v = v * (2130706432 - sm | sr & -2147483648).to(
                tl.float32, bitcast=True
            )
        else:
            v = v / s
    return v


@triton.jit
def _work(
    X,
    Q,
    S,
    n,
    tile,
    D: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    XS: tl.constexpr,
    XI: tl.constexpr,
    QS: tl.constexpr,
    QI: tl.constexpr,
):
    row = tile // C
    col = tile % C * B + tl.arange(0, B)
    src = X + row * XS + col * XI
    edge = col < D
    g = tl.load(src, edge, 0, cache_modifier="").to(tl.float32)
    u = tl.load(src + D * XI, edge, 0, cache_modifier="").to(tl.float32)
    v = _value(g, u, tl.load(S))
    tl.store(
        Q + row * QS + col * QI,
        _encode(v),
        (col < D) & (row < n),
        cache_modifier=".cs",
    )


@triton.jit
def _kernel(
    X,
    Q,
    S,
    NT,
    D: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    XS: tl.constexpr,
    XI: tl.constexpr,
    QS: tl.constexpr,
    QI: tl.constexpr,
    P: tl.constexpr,
    FULL: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    n = tl.load(NT)
    pid = tl.program_id(0).to(tl.uint32)
    if FULL:
        _work(X, Q, S, n, pid, D, B, C, XS, XI, QS, QI)
    else:
        for tile in range(pid, n * C, P):
            _work(X, Q, S, n, tile, D, B, C, XS, XI, QS, QI)


def silu_mul_static_tensorwise_quant(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    m = input.shape[0]
    if m == 0:
        return output
    b = min(
        (
            1024
            if min(max(1, expected_num_tokens), m) * intermediate_size
            <= 2097152
            else 2048
        ),
        triton.next_power_of_2(intermediate_size),
    )
    c = triton.cdiv(intermediate_size, b)
    rows = min(max(1, expected_num_tokens), m)
    p = rows * c
    _kernel[p,](
        input,
        output,
        scale,
        num_tokens_tensor,
        intermediate_size,
        b,
        c,
        *input.stride(),
        *output.stride(),
        p,
        rows == m,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["silu_mul_static_tensorwise_quant"]
