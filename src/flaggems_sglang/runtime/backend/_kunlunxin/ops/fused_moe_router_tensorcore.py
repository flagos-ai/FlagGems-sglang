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

"""Fused MoE router: logits (GEMM) -> softcap -> bias -> softmax -> top-k.

Target notes (Kunlunxin/flagtree XPU TritonSDNN backend, established with
isolated compile/run experiments):

* supported: tl.dot, single-output tl.max/tl.sum reductions, elementwise
  add/sub/mul/minimum/maximum, offset<->float casts, the exp activation LUT and
  affine masked loads/stores;
* rejected: tl.where/selects, float<->int casts, division, tl.argmax/argmin and
  non-affine (scatter) stores;
* this build exposes no tanh at all, and `tl.log2` returns the NATURAL
  logarithm, so 1/x is exp(-tl.log2(x));
* every activation argument must stay inside the LUT input window
  [-15.98, 0.0], otherwise the device faults;
* the tile DMA moves whole BLOCK_M x BLOCK_K tiles and ignores row masks, so
  every buffer it touches is given a whole tile's worth of rows;
* an epilogue placed inside the pipelined GEMM kernel corrupts results and a
  trailing multiply by a broadcast constant can be dropped, so the GEMM and the
  epilogue are separate kernels and constants are folded early.

The epilogue uses:
  * a 0/1 membership indicator exp(clamp((l - rowmax)*1e6, -15, 0)),
  * rank recovery as max(indicator * expert_index), where the index vector is
    just the lane offset - which expert wins is decided by the data,
  * softmax normalisation through exp(-ln(S)) instead of a division,
  * tanh(y) = sign(y)*(1-w)/(1+w), w = exp(-2|y|), for softcapping, with the
    softcap factor folded into the sign term (a trailing scaling is dropped).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    lg_ptr,
    M,
    K,
    E,
    s_xm,
    s_xk,
    s_we,
    s_wk,
    s_lg,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, BLOCK_E)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    e_mask = offs_e < E

    acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        xt = tl.load(
            x_ptr + offs_m[:, None] * s_xm + kk[None, :] * s_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        wt = tl.load(
            w_ptr + offs_e[:, None] * s_we + kk[None, :] * s_wk,
            mask=e_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(xt, tl.trans(wt), acc)

    logits = acc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=e_mask, other=0.0)
        logits = logits + bias[None, :]
    tl.store(
        lg_ptr + offs_m[:, None] * s_lg + offs_e[None, :],
        logits,
        mask=m_mask[:, None],
    )


@triton.jit
def _router_top2_kernel(
    lg_ptr,
    cs_ptr,
    bias_ptr,
    ow_ptr,
    oi_ptr,
    M,
    E,
    s_lg,
    s_ow,
    s_oi,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    FULL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, BLOCK_E)
    ptrs = lg_ptr + offs_m[:, None] * s_lg + offs_e[None, :]
    if FULL:
        # every lane is a real expert of a real row: no load mask needed
        logits = tl.load(ptrs)
    else:
        live = (offs_m < M)[:, None] & (offs_e < E)[None, :]
        # out-of-range experts are pushed out of the running by the load itself
        logits = tl.load(ptrs, mask=live, other=-1.0e30)
    fe = offs_e.to(tl.float32)
    if HAS_SOFTCAP:
        # constants arrive as broadcast vectors: scalar operands of elementwise
        # ops are not reliable on this backend
        c0 = tl.load(cs_ptr + 0 * BLOCK_E + offs_e)[None, :]  # 1/softcap
        c2 = tl.load(cs_ptr + 2 * BLOCK_E + offs_e)[None, :]  # 1.0
        c3 = tl.load(cs_ptr + 3 * BLOCK_E + offs_e)[None, :]  # 0.0
        c4 = tl.load(cs_ptr + 4 * BLOCK_E + offs_e)[None, :]  # 10000.0
        c5 = tl.load(cs_ptr + 5 * BLOCK_E + offs_e)[None, :]  # softcap
        y = logits * c0
        ay = tl.maximum(y, c3 - y)
        w = tl.exp(c3 - (c2 + c2) * ay)
        t = (c2 - w) * tl.exp(c3 - tl.log2(c2 + w))
        ayk = tl.maximum(ay * c4, c2)
        sgn = (y * c4) * tl.exp(c3 - tl.log2(ayk))
        logits = (sgn * c5) * t

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=offs_e < E, other=0.0)
        logits = logits + bias[None, :]

    lo = fe[None, :] * 0.0 - 15.0
    zero = fe[None, :] * 0.0
    live_r = offs_m < M
    rowmax = tl.max(logits, axis=1)
    shifted = logits - rowmax[:, None]
    denom = tl.sum(tl.exp(tl.maximum(shifted, lo)), axis=1)
    log_denom = tl.log2(denom)
    inv_denom = tl.exp(0.0 - log_denom)

    # 0/1 membership: 1 exactly on the lanes attaining the row max, 0 elsewhere.
    # A linear clamp replaces a scaled exp so that the softmax denominator is
    # the only full-tile activation evaluation.
    ind0 = tl.maximum(shifted * 1000000.0 + 1.0, zero)
    tl.store(ow_ptr + offs_m * s_ow, inv_denom, mask=live_r)
    tl.store(
        oi_ptr + offs_m * s_oi,
        tl.max(ind0 * fe[None, :], axis=1).to(tl.int32),
        mask=live_r,
    )

    rest = logits - 1000.0 * ind0
    mx2 = tl.max(rest, axis=1)
    tl.store(
        ow_ptr + offs_m * s_ow + 1,
        tl.exp(mx2 - rowmax) * inv_denom,
        mask=live_r,
    )
    ind1 = tl.maximum((rest - mx2[:, None]) * 1000000.0 + 1.0, zero)
    tl.store(
        oi_ptr + offs_m * s_oi + 1,
        tl.max(ind1 * fe[None, :], axis=1).to(tl.int32),
        mask=live_r,
    )


@triton.jit
def _router_topk_kernel(
    lg_ptr,
    cs_ptr,
    bias_ptr,
    ow_ptr,
    oi_ptr,
    M,
    E,
    topk,
    s_lg,
    s_ow,
    s_oi,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, BLOCK_E)
    live = (offs_m < M)[:, None] & (offs_e < E)[None, :]
    logits = tl.load(
        lg_ptr + offs_m[:, None] * s_lg + offs_e[None, :],
        mask=live,
        other=-1.0e30,
    )
    fe = offs_e.to(tl.float32)
    if HAS_SOFTCAP:
        c0 = tl.load(cs_ptr + 0 * BLOCK_E + offs_e)[None, :]
        c2 = tl.load(cs_ptr + 2 * BLOCK_E + offs_e)[None, :]
        c3 = tl.load(cs_ptr + 3 * BLOCK_E + offs_e)[None, :]
        c4 = tl.load(cs_ptr + 4 * BLOCK_E + offs_e)[None, :]
        c5 = tl.load(cs_ptr + 5 * BLOCK_E + offs_e)[None, :]
        y = logits * c0
        ay = tl.maximum(y, c3 - y)
        w = tl.exp(c3 - (c2 + c2) * ay)
        t = (c2 - w) * tl.exp(c3 - tl.log2(c2 + w))
        ayk = tl.maximum(ay * c4, c2)
        sgn = (y * c4) * tl.exp(c3 - tl.log2(ayk))
        logits = (sgn * c5) * t

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=offs_e < E, other=0.0)
        logits = logits + bias[None, :]

    lo = fe[None, :] * 0.0 - 15.0
    zero = fe[None, :] * 0.0
    live_r = offs_m < M
    rowmax = tl.max(logits, axis=1)
    shifted = logits - rowmax[:, None]
    denom = tl.sum(tl.exp(tl.maximum(shifted, lo)), axis=1)
    log_denom = tl.log2(denom)

    rest = logits
    for j in range(0, topk):
        mx = tl.max(rest, axis=1)
        ind = tl.maximum((rest - mx[:, None]) * 1000000.0 + 1.0, zero)
        tl.store(
            ow_ptr + offs_m * s_ow + j,
            tl.exp(mx - rowmax) * tl.exp(0.0 - log_denom),
            mask=live_r,
        )
        tl.store(
            oi_ptr + offs_m * s_oi + j,
            tl.max(ind * fe[None, :], axis=1).to(tl.int32),
            mask=live_r,
        )
        rest = rest - 1000.0 * ind


def _launch(x, router_weight, topk, moe_softcapping, correction_bias):
    if correction_bias is not None:
        correction_bias = correction_bias.contiguous()
    M, K = x.shape
    E = router_weight.shape[0]
    k = int(topk)
    if k < 1:
        k = 1
    if k > E:
        k = E

    device = x.device
    # measured on the target: the GEMM wants the largest tile it can hold once
    # the grid is wide, and tiny tiles when it is not, while the epilogue is
    # always fastest with a narrow row tile because that widens its grid
    if M >= 1024:
        BLOCK_M, num_warps_gemm = 256, 8
    elif M >= 128:
        BLOCK_M, num_warps_gemm = 64, 4
    else:
        BLOCK_M, num_warps_gemm = 16, 4
    BLOCK_ME = 16
    # the epilogue tile must not be narrower than the backend's vector unit
    BLOCK_E = max(32, triton.next_power_of_2(E))
    BLOCK_K = 256
    padM = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M

    lg = torch.empty((padM, BLOCK_E), dtype=torch.float32, device=device)
    ow_buf = torch.empty(padM * k, dtype=torch.float32, device=device)
    oi_buf = torch.empty(padM * k, dtype=torch.int32, device=device)

    softcap = 0.0 if moe_softcapping is None else float(moe_softcapping)
    if softcap != 0.0:
        consts = [1.0 / softcap, 2.0, 1.0, 0.0, 10000.0, softcap]
        cs = (
            torch.tensor(consts, dtype=torch.float32)
            .repeat_interleave(BLOCK_E)
            .view(6, BLOCK_E)
            .to(device)
            .contiguous()
        )
    else:
        cs = ow_buf  # unused

    grid = (triton.cdiv(M, BLOCK_M),)
    _router_gemm_kernel[grid](
        x,
        router_weight,
        correction_bias if correction_bias is not None else lg,
        lg,
        M,
        K,
        E,
        x.stride(0),
        x.stride(1),
        router_weight.stride(0),
        router_weight.stride(1),
        lg.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_E=BLOCK_E,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=False,
        num_warps=num_warps_gemm,
        num_stages=2,
    )
    grid_e = (triton.cdiv(M, BLOCK_ME),)
    if k == 2:
        _router_top2_kernel[grid_e](
            lg,
            cs,
            correction_bias if correction_bias is not None else lg,
            ow_buf,
            oi_buf,
            M,
            E,
            lg.stride(0),
            k,
            k,
            BLOCK_M=BLOCK_ME,
            BLOCK_E=BLOCK_E,
            HAS_SOFTCAP=softcap != 0.0,
            HAS_BIAS=correction_bias is not None,
            FULL=(E == BLOCK_E and M % BLOCK_ME == 0),
            num_warps=4,
        )
    else:
        _router_topk_kernel[grid_e](
            lg,
            cs,
            correction_bias if correction_bias is not None else lg,
            ow_buf,
            oi_buf,
            M,
            E,
            k,
            lg.stride(0),
            k,
            k,
            BLOCK_M=BLOCK_ME,
            BLOCK_E=BLOCK_E,
            HAS_SOFTCAP=softcap != 0.0,
            HAS_BIAS=correction_bias is not None,
            num_warps=4,
        )

    out_w = ow_buf[: M * k].view(M, k)
    out_i = oi_buf[: M * k].view(M, k)
    return out_w, out_i


def fused_moe_router_tensorcore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    return _launch(x, router_weight, topk, moe_softcapping, correction_bias)


__all__ = ["fused_moe_router_tensorcore"]
