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
def _mono(bits):
    """Signed-int image preserving IEEE-754 order; its own inverse."""
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _topk_sigmoid_kernel(
    x_ptr,
    w_ptr,
    i_ptr,
    n_experts,
    scale,
    stride_x,
    stride_w,
    stride_i,
    k: tl.constexpr,
    RENORM: tl.constexpr,
    BLOCK: tl.constexpr,
    KSLOT: tl.constexpr,
    NOMASK: tl.constexpr,
    FAST32: tl.constexpr,
    I32LOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid.to(tl.int64)
    cols = tl.arange(0, BLOCK)

    # Load logits. Padded lanes get -inf, whose monotonic image is the
    # global minimum, so they are never selected (real logits are finite).
    if I32LOAD:
        # fp32 input with the FAST32 selection path: the raw bit pattern is
        # what the monotonic map consumes, so skip the float convert entirely.
        b32 = x_ptr.to(tl.pointer_type(tl.int32))
        if NOMASK:
            key = _mono(tl.load(b32 + row * stride_x + cols))
        else:
            key = _mono(
                tl.load(
                    b32 + row * stride_x + cols,
                    mask=cols < n_experts,
                    other=-8388608,
                )
            )
    else:
        if NOMASK:
            s = tl.load(x_ptr + row * stride_x + cols).to(tl.float32)
        else:
            s = tl.load(
                x_ptr + row * stride_x + cols,
                mask=cols < n_experts,
                other=float("-inf"),
            ).to(tl.float32)
        key = _mono(s.to(tl.int32, bitcast=True))

    kslots = tl.arange(0, KSLOT)
    vals = tl.zeros([KSLOT], dtype=tl.float32)
    if FAST32:
        cols64 = cols.to(tl.int64)
        acc = tl.zeros([BLOCK], dtype=tl.int64)
        for j in tl.static_range(0, k):
            pm = tl.max(key, axis=0)
            match = key == pm
            acc = acc | tl.where(match, cols64 << (8 * j), 0)
            vals = tl.where(
                kslots == j, _mono(pm).to(tl.float32, bitcast=True), vals
            )
            key = tl.where(match, -2147483648, key)
        idx = ((tl.sum(acc, axis=0) >> (kslots * 8)) & 0xFF).to(tl.int32)
    else:
        # General fallback: packed (value, reversed column) int64 key, one
        # reduction per round, column rides in the low word.
        rev = (BLOCK - 1 - cols).to(tl.int64)
        packed = (key.to(tl.int64) << 32) | rev
        acc = tl.zeros([KSLOT], dtype=tl.int64)
        for j in tl.static_range(0, k):
            pm = tl.max(packed, axis=0)
            acc = tl.where(kslots == j, pm, acc)
            packed = tl.where(packed == pm, (-9223372036854775808), packed)
        vals = _mono((acc >> 32).to(tl.int32)).to(tl.float32, bitcast=True)
        idx = (BLOCK - 1 - (acc & 0xFFFFFFFF)).to(tl.int32)

    km = kslots < k
    wsum = tl.sigmoid(vals)
    if RENORM:
        total = tl.sum(tl.where(km, wsum, 0.0), axis=0) + 1e-20
        wsum = tl.where(km, wsum * (scale / total), 0.0)

    if KSLOT == k:
        tl.store(w_ptr + row * stride_w + kslots, wsum)
        tl.store(i_ptr + row * stride_i + kslots, idx)
    else:
        tl.store(w_ptr + row * stride_w + kslots, wsum, mask=km)
        tl.store(i_ptr + row * stride_i + kslots, idx, mask=km)


@triton.jit
def _merge_top8(
    a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7
):
    """Bitonic merge of two sorted-desc top-8 lists -> sorted-desc."""
    c0 = tl.maximum(a0, b7)
    c4 = tl.maximum(a4, b3)
    c1 = tl.maximum(a1, b6)
    c5 = tl.maximum(a5, b2)
    c2 = tl.maximum(a2, b5)
    c6 = tl.maximum(a6, b1)
    c3 = tl.maximum(a3, b4)
    c7 = tl.maximum(a7, b0)
    d0 = tl.maximum(c0, c4)
    d4 = tl.minimum(c0, c4)
    d1 = tl.maximum(c1, c5)
    d5 = tl.minimum(c1, c5)
    d2 = tl.maximum(c2, c6)
    d6 = tl.minimum(c2, c6)
    d3 = tl.maximum(c3, c7)
    d7 = tl.minimum(c3, c7)
    e0 = tl.maximum(d0, d2)
    e2 = tl.minimum(d0, d2)
    e1 = tl.maximum(d1, d3)
    e3 = tl.minimum(d1, d3)
    e4 = tl.maximum(d4, d6)
    e6 = tl.minimum(d4, d6)
    e5 = tl.maximum(d5, d7)
    e7 = tl.minimum(d5, d7)
    f0 = tl.maximum(e0, e1)
    f1 = tl.minimum(e0, e1)
    f2 = tl.maximum(e2, e3)
    f3 = tl.minimum(e2, e3)
    f4 = tl.maximum(e4, e5)
    f5 = tl.minimum(e4, e5)
    f6 = tl.maximum(e6, e7)
    f7 = tl.minimum(e6, e7)
    return (f0, f1, f2, f3, f4, f5, f6, f7)


@triton.jit
def _topk_sigmoid_tuple_kernel(
    x_ptr,
    w_ptr,
    i_ptr,
    n_experts,
    scale,
    stride_x,
    stride_w,
    stride_i,
    k: tl.constexpr,
    RENORM: tl.constexpr,
    BLOCK: tl.constexpr,
    KSLOT: tl.constexpr,
    NOMASK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid.to(tl.int64)
    NL: tl.constexpr = BLOCK // 4
    offs = tl.arange(0, BLOCK)

    if NOMASK:
        s = tl.load(x_ptr + row * stride_x + offs).to(tl.float32)
    else:
        s = tl.load(
            x_ptr + row * stride_x + offs,
            mask=offs < n_experts,
            other=float("-inf"),
        ).to(tl.float32)
    key = _mono(s.to(tl.int32, bitcast=True))

    packed = (key.to(tl.int64) << 32) | offs.to(tl.int64)
    r = tl.reshape(packed, [NL, 2, 2])
    (e01, e23) = tl.split(r)  # e01 = slots 4l..4l+1, e23 = slots 4l+2..4l+3
    (e0, e1) = tl.split(e01)
    (e2, e3) = tl.split(e23)
    # Sort-4 network: 5 compare-exchanges, serial ALU.
    lo01 = tl.minimum(e0, e1)
    hi01 = tl.maximum(e0, e1)
    lo23 = tl.minimum(e2, e3)
    hi23 = tl.maximum(e2, e3)
    glo = tl.minimum(lo01, lo23)
    ghi = tl.maximum(hi01, hi23)
    m1 = tl.maximum(lo01, lo23)
    m2 = tl.minimum(hi01, hi23)
    p0 = ghi
    p1 = tl.maximum(m1, m2)
    p2 = tl.minimum(m1, m2)
    p3 = glo
    v0 = (p0 >> 32).to(tl.int32)
    v1 = (p1 >> 32).to(tl.int32)
    v2 = (p2 >> 32).to(tl.int32)
    v3 = (p3 >> 32).to(tl.int32)
    c0 = p0 & 0xFFFFFFFF
    c1 = p1 & 0xFFFFFFFF
    c2 = p2 & 0xFFFFFFFF
    c3 = p3 & 0xFFFFFFFF

    # Single cross-lane reduction: merge all lanes' sorted top-4 (padded with
    # the int minimum, which no finite logit can beat) into the global top-8.
    NEG: tl.constexpr = -2147483648
    z = tl.full([NL], NEG, tl.int32)
    w0, w1, w2, w3, w4, w5, w6, w7 = tl.reduce(
        (v0, v1, v2, v3, z, z, z, z), 0, _merge_top8
    )

    kslots = tl.arange(0, KSLOT)
    vals = tl.where(
        kslots == 0,
        w0,
        tl.where(
            kslots == 1,
            w1,
            tl.where(
                kslots == 2,
                w2,
                tl.where(
                    kslots == 3,
                    w3,
                    tl.where(
                        kslots == 4,
                        w4,
                        tl.where(
                            kslots == 5, w5, tl.where(kslots == 6, w6, w7)
                        ),
                    ),
                ),
            ),
        ),
    )

    acc = tl.zeros([NL], dtype=tl.int64)
    r0 = (
        (v0 < w0).to(tl.int32)
        + (v0 < w1).to(tl.int32)
        + (v0 < w2).to(tl.int32)
        + (v0 < w3).to(tl.int32)
        + (v0 < w4).to(tl.int32)
        + (v0 < w5).to(tl.int32)
        + (v0 < w6).to(tl.int32)
        + (v0 < w7).to(tl.int32)
    )
    acc += tl.where(r0 < 8, c0.to(tl.int64) << (8 * r0), 0)
    r1 = (
        (v1 < w0).to(tl.int32)
        + (v1 < w1).to(tl.int32)
        + (v1 < w2).to(tl.int32)
        + (v1 < w3).to(tl.int32)
        + (v1 < w4).to(tl.int32)
        + (v1 < w5).to(tl.int32)
        + (v1 < w6).to(tl.int32)
        + (v1 < w7).to(tl.int32)
    )
    acc += tl.where(r1 < 8, c1.to(tl.int64) << (8 * r1), 0)
    r2 = (
        (v2 < w0).to(tl.int32)
        + (v2 < w1).to(tl.int32)
        + (v2 < w2).to(tl.int32)
        + (v2 < w3).to(tl.int32)
        + (v2 < w4).to(tl.int32)
        + (v2 < w5).to(tl.int32)
        + (v2 < w6).to(tl.int32)
        + (v2 < w7).to(tl.int32)
    )
    acc += tl.where(r2 < 8, c2.to(tl.int64) << (8 * r2), 0)
    r3 = (
        (v3 < w0).to(tl.int32)
        + (v3 < w1).to(tl.int32)
        + (v3 < w2).to(tl.int32)
        + (v3 < w3).to(tl.int32)
        + (v3 < w4).to(tl.int32)
        + (v3 < w5).to(tl.int32)
        + (v3 < w6).to(tl.int32)
        + (v3 < w7).to(tl.int32)
    )
    acc += tl.where(r3 < 8, c3.to(tl.int64) << (8 * r3), 0)
    sh = kslots * 8
    idx = ((tl.sum(acc, axis=0) >> sh) & 0xFF).to(tl.int32)

    km = kslots < k
    # vals hold monotonic keys; _mono is its own inverse, so applying it
    # recovers the exact fp32 bits of the winning logits.
    wsum = tl.sigmoid(_mono(vals).to(tl.float32, bitcast=True))
    if RENORM:
        total = tl.sum(tl.where(km, wsum, 0.0), axis=0) + 1e-20
        wsum = tl.where(km, wsum * (scale / total), 0.0)
    tl.store(w_ptr + row * stride_w + kslots, wsum, mask=km)
    tl.store(i_ptr + row * stride_i + kslots, idx, mask=km)


def topk_sigmoid(
    topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor
):
    num_tokens, n_experts = gating_output.shape
    k = topk_weights.shape[1]
    if num_tokens == 0 or k == 0:
        return topk_weights, topk_ids

    x = (
        gating_output
        if gating_output.is_contiguous()
        else gating_output.contiguous()
    )

    scale = float(routed_scaling_factor)
    stride_x = x.stride(0)
    stride_w = topk_weights.stride(0)
    stride_i = topk_ids.stride(0)
    renorm = bool(renormalize)

    # Which kernel the dispatch below selects depends only on the input
    # configuration itself.
    can_tuple = n_experts <= 256 and k <= 8  # == (BLOCK <= 256 and k <= 8)
    use_tuple = can_tuple and num_tokens <= 512
    # fp32 rows can be loaded directly as int32 words (monotonic key needs
    # only the bit pattern); other dtypes keep the float-load path.
    i32load = x.dtype == torch.float32

    # Standard JIT dispatch: compiles on first sight of a configuration and
    # replays Triton's own kernel cache afterwards.
    BLOCK = max(16, triton.next_power_of_2(n_experts))
    KSLOT = triton.next_power_of_2(k)
    NOMASK = n_experts == BLOCK
    fast32 = BLOCK <= 256 and k <= 8
    use_tuple = can_tuple and num_tokens <= 512
    if use_tuple:
        _topk_sigmoid_tuple_kernel[(num_tokens,)](
            x,
            topk_weights,
            topk_ids,
            n_experts,
            scale,
            stride_x,
            stride_w,
            stride_i,
            k=k,
            RENORM=renorm,
            BLOCK=BLOCK,
            KSLOT=KSLOT,
            NOMASK=NOMASK,
            num_warps=1,
            num_stages=1,
        )
    else:
        _topk_sigmoid_kernel[(num_tokens,)](
            x,
            topk_weights,
            topk_ids,
            n_experts,
            scale,
            stride_x,
            stride_w,
            stride_i,
            k=k,
            RENORM=renorm,
            BLOCK=BLOCK,
            KSLOT=KSLOT,
            NOMASK=NOMASK,
            FAST32=fast32,
            I32LOAD=i32load,
            num_warps=1,
            num_stages=1,
        )
    return topk_weights, topk_ids


__all__ = ["topk_sigmoid"]
