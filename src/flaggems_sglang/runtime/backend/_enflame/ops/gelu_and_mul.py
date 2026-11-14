# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Enflame GCU specialization for sglang ``gelu_and_mul``.

The original 2D block-pointer kernel (tile up to 65536 elements, 12-CTA cap)
cannot freshly compile on the post-2026-11-07 toolchain (rank-2
``memref_ext.slice_pad_start`` operandSegmentSizes 8 vs 7 regression; see
compiler_issues/vllm_silu_flat_slice_start.md) and only used 12 of the 48
SPs on GCU400. The 2026-11-08 rewrite streamed the output space with
row-aligned 1-D chunks (one row-chunk per loop iteration). The 2026-11-12
E-1 round keeps that skeleton and attacks the two measured bottlenecks:

* **Per-iteration latency.** The ``range`` grid-stride loop is not software
  pipelined by this backend, so each iteration pays the full memory-op
  latency before the next iteration's loads can issue. With one 1024-element
  row-chunk per iteration (``d = 1024``) this capped the kernel at
  ~435 GB/s (measured), far below the ~1.3-2.0 TB/s streaming floor. Fix:
  small-block shapes now process 4 (block <= 1024) or 2 (block == 2048)
  rows per iteration with **all loads textually issued first** and stores
  last ("loads 全前置" pattern; the loads of one iteration overlap each
  other and the math of the previous row). Row tails are handled by the
  idempotent anchored-group trick (``r0 = min(g*R, m-R)``; rows are
  independent so the overlapped recompute writes identical values).
  Measured on GCU400 (bf16, official bench kernel mode, do_bench median):
  (4096, 1024) 59.7 -> 25.4 us, (512, 1024) 10.5 -> 6.9 us.
* **Exposed ALU chain.** The deg-8 Horner fit costs ~1.4-1.7 ns/elem on
  top of the memory floor for 16-bit inputs (the fp32 upcast doubles the
  internal vector slots). 16-bit outputs only need the deg-6 fit (fit
  error 2.4e-5 in gelu(x), two orders below the bf16/fp16 output ulp; the
  deg-6 leading coefficient is positive so the s = min(x*x, 32) clamp is
  dropped -- the polynomial extrapolates monotonically and sigmoid
  saturates to Phi's exact 0/1 tails; verified at +-inf / +-65504 /
  |x| = 8 on the vllm twin op). fp32 keeps a deg-7 fit with the clamp
  (deg-6's error amplified by |y| exceeds the 1e-4 atol; deg-7 stays at
  2.8e-5). num_warps=4 hides the remaining chain behind the loads for
  the wide-block shapes: (4096, 4096) 110.2 -> 88.6 us,
  (4096, 8192) 216.7 -> 163.5 us.

Rejected this round (measured, do_bench median): flat output-space
streaming with shift/and row recovery (``col = offs & (D-1)``) degrades
the load lowering to ~500-670 GB/s (vs 930+ for the affine row base);
2D [R, BLOCK] tiles and R-grouped loads at BLOCK >= 4096 lose to the 1-D
form (2D tile element chains spill to DSM; 4x4096 live set blows the
register budget); BLOCK = 8192 compiles but is ~1.7x slower than 4096.

* main path (contiguous input): block = largest power-of-two divisor of
  ``d`` capped to [256, 4096] (exact chunking, no column masks; falls back
  to a masked next-pow2 block when ``d`` has no such divisor, e.g. d=16).
  16-bit dtypes with small blocks take the grouped-row kernels; everything
  else takes the 1-D row-chunk kernel (EVEN chunks drop all masking).
* the erf term is evaluated through the sigmoid-Chebyshev composition
  verified on vllm gelu_and_mul (whole-range |err| <= 7.2e-7 vs torch.erf
  at deg-8), avoiding the extern ``__ef_erff`` call that dominates
  streaming elementwise kernels on this backend.
* strided fallback: same row-chunk shape with runtime strides and i64 row
  bases (also serves inputs beyond the int32 element-offset domain); math
  unchanged from the 2026-11-08 rewrite (deg-8 + clamp).
"""

import torch
import triton
import triton.language as tl

__all__ = ["gelu_and_mul"]

_PERSISTENT_CTAS = 48  # GCU400 SP count (device query reports 4; see AGENTS)
_FLAT_BLOCK = 4096
_FLAT_MIN_BLOCK = 256
_STRIDED_BLOCK = 2048
_INT32_ELEM_MAX = 2**31 - 1

# deg-8 Chebyshev fit of erf(z) = tanh(z * g(z^2)) on z in [0, 4] with the
# 2/sqrt(2) folding of gelu(x) = x * sigmoid(x * P(min(x*x, 32))).
# Strided fallback + non-{16,32}-bit dtypes.
_G0 = tl.constexpr(1.595768893)
_G1 = tl.constexpr(0.0726593119)
_G2 = tl.constexpr(-5.225587988e-05)
_G3 = tl.constexpr(-1.171002928e-04)
_G4 = tl.constexpr(9.309218932e-06)
_G5 = tl.constexpr(-4.064449204e-07)
_G6 = tl.constexpr(1.062684386e-08)
_G7 = tl.constexpr(-1.551424178e-10)
_G8 = tl.constexpr(9.721780608e-13)

# deg-6 Chebyshev least-squares fit of P(s) = 2*atanh(erf(sqrt(s/2)))/sqrt(s)
# over s in [0, 32]; 16-bit path, no clamp (positive leading coefficient
# keeps the s > 32 extrapolation monotone, so sigmoid saturates to Phi's
# exact tails). Coefficients verified on the vllm twin op (2026-11-11).
_E0 = tl.constexpr(1.5953784901812362)
_E1 = tl.constexpr(0.07338409181814284)
_E2 = tl.constexpr(-0.0003813880148825668)
_E3 = tl.constexpr(-5.428118007938764e-05)
_E4 = tl.constexpr(3.1787987321044684e-06)
_E5 = tl.constexpr(-7.568447796921953e-08)
_E6 = tl.constexpr(6.8404153739139e-10)

# deg-7 fit of the same target; 32-bit path, clamped at s = 32 (the
# negative leading coefficient makes unclamped extrapolation unsafe).
_D0 = tl.constexpr(1.5956858496775363)
_D1 = tl.constexpr(0.07284617639058376)
_D2 = tl.constexpr(-0.00015445021611273802)
_D3 = tl.constexpr(-9.368046462698117e-05)
_D4 = tl.constexpr(6.564689983953184e-06)
_D5 = tl.constexpr(-2.280499144167936e-07)
_D6 = tl.constexpr(4.1228476561246845e-09)
_D7 = tl.constexpr(-3.0703626060118777e-11)


@triton.jit
def _gelu_erf_cheap(xf):
    sq = tl.minimum(xf * xf, 32.0)
    g = _G8
    g = _G7 + sq * g
    g = _G6 + sq * g
    g = _G5 + sq * g
    g = _G4 + sq * g
    g = _G3 + sq * g
    g = _G2 + sq * g
    g = _G1 + sq * g
    g = _G0 + sq * g
    return xf * tl.sigmoid(xf * g)


@triton.jit
def _gelu6(xf):
    sq = xf * xf
    g = _E6
    g = _E5 + sq * g
    g = _E4 + sq * g
    g = _E3 + sq * g
    g = _E2 + sq * g
    g = _E1 + sq * g
    g = _E0 + sq * g
    return xf * tl.sigmoid(xf * g)


@triton.jit
def _gelu7(xf):
    sq = tl.minimum(xf * xf, 32.0)
    g = _D7
    g = _D6 + sq * g
    g = _D5 + sq * g
    g = _D4 + sq * g
    g = _D3 + sq * g
    g = _D2 + sq * g
    g = _D1 + sq * g
    g = _D0 + sq * g
    return xf * tl.sigmoid(xf * g)


@triton.jit
def _gelu_poly(xf, DEG: tl.constexpr):
    if DEG == 6:
        out = _gelu6(xf)
    elif DEG == 7:
        out = _gelu7(xf)
    else:
        out = _gelu_erf_cheap(xf)
    return out


@triton.jit
def _gelu_and_mul_row_kernel(
    x_ptr,
    o_ptr,
    m,
    nct,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    DEG: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for tile in range(pid, m * nct, nprog):
        row = tile // nct
        c0 = (tile - row * nct) * BLOCK
        rb = row * (2 * D)
        offs = c0 + tl.arange(0, BLOCK)
        if EVEN:
            gate = tl.load(x_ptr + rb + offs)
            up = tl.load(x_ptr + rb + D + offs)
            gelu = _gelu_poly(gate.to(tl.float32), DEG)
            tl.store(
                o_ptr + row * D + offs,
                (gelu * up.to(tl.float32)).to(o_ptr.dtype.element_ty),
            )
        else:
            cm = offs < D
            gate = tl.load(x_ptr + rb + offs, mask=cm)
            up = tl.load(x_ptr + rb + D + offs, mask=cm)
            gelu = _gelu_poly(gate.to(tl.float32), DEG)
            tl.store(
                o_ptr + row * D + offs,
                (gelu * up.to(tl.float32)).to(o_ptr.dtype.element_ty),
                mask=cm,
            )


@triton.jit
def _gelu_and_mul_rg4_kernel(
    x_ptr,
    o_ptr,
    m,
    nct,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 4 rows per iteration; all 8 loads issue before any math/store so the
    # load latencies overlap. Anchored last group (rows are independent,
    # overlap recompute is idempotent). Requires BLOCK | D (host-guaranteed).
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ngroups = tl.cdiv(m, 4)
    for g in range(pid, ngroups * nct, nprog):
        grp = g // nct
        c0 = (g - grp * nct) * BLOCK
        r0 = tl.minimum(grp * 4, m - 4)
        rb0 = r0 * (2 * D)
        offs = c0 + tl.arange(0, BLOCK)
        ga0 = tl.load(x_ptr + rb0 + offs)
        up0 = tl.load(x_ptr + rb0 + D + offs)
        ga1 = tl.load(x_ptr + rb0 + 2 * D + offs)
        up1 = tl.load(x_ptr + rb0 + 3 * D + offs)
        ga2 = tl.load(x_ptr + rb0 + 4 * D + offs)
        up2 = tl.load(x_ptr + rb0 + 5 * D + offs)
        ga3 = tl.load(x_ptr + rb0 + 6 * D + offs)
        up3 = tl.load(x_ptr + rb0 + 7 * D + offs)
        ob0 = r0 * D
        tl.store(
            o_ptr + ob0 + offs,
            (_gelu6(ga0.to(tl.float32)) * up0.to(tl.float32)).to(o_ptr.dtype.element_ty),
        )
        tl.store(
            o_ptr + ob0 + D + offs,
            (_gelu6(ga1.to(tl.float32)) * up1.to(tl.float32)).to(o_ptr.dtype.element_ty),
        )
        tl.store(
            o_ptr + ob0 + 2 * D + offs,
            (_gelu6(ga2.to(tl.float32)) * up2.to(tl.float32)).to(o_ptr.dtype.element_ty),
        )
        tl.store(
            o_ptr + ob0 + 3 * D + offs,
            (_gelu6(ga3.to(tl.float32)) * up3.to(tl.float32)).to(o_ptr.dtype.element_ty),
        )


@triton.jit
def _gelu_and_mul_rg2_kernel(
    x_ptr,
    o_ptr,
    m,
    nct,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 2 rows per iteration (block == 2048 register-budget sweet spot).
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ngroups = tl.cdiv(m, 2)
    for g in range(pid, ngroups * nct, nprog):
        grp = g // nct
        c0 = (g - grp * nct) * BLOCK
        r0 = tl.minimum(grp * 2, m - 2)
        rb0 = r0 * (2 * D)
        offs = c0 + tl.arange(0, BLOCK)
        ga0 = tl.load(x_ptr + rb0 + offs)
        up0 = tl.load(x_ptr + rb0 + D + offs)
        ga1 = tl.load(x_ptr + rb0 + 2 * D + offs)
        up1 = tl.load(x_ptr + rb0 + 3 * D + offs)
        ob0 = r0 * D
        tl.store(
            o_ptr + ob0 + offs,
            (_gelu6(ga0.to(tl.float32)) * up0.to(tl.float32)).to(o_ptr.dtype.element_ty),
        )
        tl.store(
            o_ptr + ob0 + D + offs,
            (_gelu6(ga1.to(tl.float32)) * up1.to(tl.float32)).to(o_ptr.dtype.element_ty),
        )


@triton.jit
def _gelu_and_mul_strided_kernel(
    x_ptr,
    o_ptr,
    m,
    nct,
    d,
    isb,
    ish,
    osb,
    osh,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for tile in range(pid, m * nct, nprog):
        row = tile // nct
        c0 = (tile - row * nct) * BLOCK
        rb = (row * isb).to(tl.int64)
        ob = (row * osb).to(tl.int64)
        offs = c0 + tl.arange(0, BLOCK)
        cm = offs < d
        gate = tl.load(x_ptr + rb + offs * ish, mask=cm).to(tl.float32)
        gelu = _gelu_erf_cheap(gate)
        up = tl.load(x_ptr + rb + (d + offs) * ish, mask=cm)
        tl.store(
            o_ptr + ob + offs * osh,
            (gelu * up.to(tl.float32)).to(o_ptr.dtype.element_ty),
            mask=cm,
        )


def gelu_and_mul(hidden_states):
    batch_size, doubled_hidden_size = hidden_states.shape
    hidden_size = doubled_hidden_size // 2
    output = torch.empty(
        (batch_size, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if batch_size == 0 or hidden_size == 0:
        return output
    is_contiguous = (
        hidden_states.stride(1) == 1
        and hidden_states.stride(0) == doubled_hidden_size
        and output.is_contiguous()
    )
    if is_contiguous and hidden_states.numel() <= _INT32_ELEM_MAX:
        # Exact power-of-two chunking when possible (no column masks).
        pow2_div = hidden_size & -hidden_size
        if pow2_div >= _FLAT_MIN_BLOCK:
            block = min(_FLAT_BLOCK, pow2_div)
            nct = hidden_size // block
        else:
            block = min(
                _FLAT_BLOCK,
                max(_FLAT_MIN_BLOCK, triton.next_power_of_2(hidden_size)),
            )
            nct = triton.cdiv(hidden_size, block)
        even = hidden_size % block == 0
        dtype = hidden_states.dtype
        tiles = batch_size * nct
        if dtype in (torch.float16, torch.bfloat16):
            # Grouped-row arms only once the 1-D form is iteration-bound
            # (>= 2 full waves); below that the 1-D form fills the SPs better
            # (measured crossover at d=1024: bs=64 -> 1-D wins, bs>=96 ->
            # grouped wins).
            if even and block <= 1024 and batch_size >= 4 and tiles >= 96:
                grid = (min(_PERSISTENT_CTAS, triton.cdiv(batch_size, 4) * nct),)
                _gelu_and_mul_rg4_kernel[grid](
                    hidden_states,
                    output,
                    batch_size,
                    nct,
                    D=hidden_size,
                    BLOCK=block,
                    num_warps=4,
                )
                return output
            if even and block == 2048 and batch_size >= 2 and tiles >= 96:
                grid = (min(_PERSISTENT_CTAS, triton.cdiv(batch_size, 2) * nct),)
                _gelu_and_mul_rg2_kernel[grid](
                    hidden_states,
                    output,
                    batch_size,
                    nct,
                    D=hidden_size,
                    BLOCK=block,
                    num_warps=4,
                )
                return output
            deg = 6
        elif dtype == torch.float32:
            deg = 7
        else:
            deg = 8
        grid = (min(_PERSISTENT_CTAS, batch_size * nct),)
        _gelu_and_mul_row_kernel[grid](
            hidden_states,
            output,
            batch_size,
            nct,
            D=hidden_size,
            BLOCK=block,
            EVEN=even,
            DEG=deg,
            num_warps=4 if block >= _FLAT_BLOCK else 2,
        )
    else:
        block = min(
            _STRIDED_BLOCK,
            max(_FLAT_MIN_BLOCK, triton.next_power_of_2(hidden_size)),
        )
        nct = triton.cdiv(hidden_size, block)
        grid = (min(_PERSISTENT_CTAS, batch_size * nct),)
        _gelu_and_mul_strided_kernel[grid](
            hidden_states,
            output,
            batch_size,
            nct,
            hidden_size,
            hidden_states.stride(0),
            hidden_states.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK=block,
            num_warps=4,
        )
    return output
