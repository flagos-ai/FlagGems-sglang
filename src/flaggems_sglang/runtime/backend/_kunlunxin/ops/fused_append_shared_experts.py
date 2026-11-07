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
from triton.experimental import tle


@triton.jit
def _direct(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    SCALE: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    I0: tl.constexpr = META[5]
    I1: tl.constexpr = META[6]
    W0: tl.constexpr = META[7]
    W1: tl.constexpr = META[8]
    P: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    x = tl.program_id(0) * B + tl.arange(0, B)
    row = x // (K + S)
    col = x % (K + S)
    rr = tl.minimum(row, M - 1)
    cc = tl.minimum(col, K - 1)
    ids = tl.load(IdsPtr + rr * I0 + cc * I1)
    weights = tl.load(W + rr * W0 + cc * W1)
    ids = tl.where(col < K, ids, N + col - K)
    weights = tl.where(col < K, weights, SCALE)
    tl.store(OI + row * P + col, ids, x < M * (K + S))
    tl.store(OW + row * P + col, weights, x < M * (K + S))


@triton.jit
def _fused(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    K: tl.constexpr = META[0]
    S: tl.constexpr = META[1]
    SCALE: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    I0: tl.constexpr = META[4]
    I1: tl.constexpr = META[5]
    W0: tl.constexpr = META[6]
    W1: tl.constexpr = META[7]
    P: tl.constexpr = META[8]
    B: tl.constexpr = META[9]
    if I1 == 1 and W1 == 1:
        lane = tl.arange(0, 2 * B)
        plane = lane // B
        x = tl.program_id(0) * B + lane % B
        row = x // K
        col = x % K
        src = tl.where(plane == 0, IdsPtr, W.to(tl.pointer_type(tl.int32)))
        dst = tl.where(plane == 0, OI, OW.to(tl.pointer_type(tl.int32)))
        s0 = tl.where(plane == 0, I0, W0)
        s1 = tl.where(plane == 0, I1, W1)
        val = tl.load(src + row * s0 + col * s1)
        tl.store(dst + row * P + col, val)
        Q: tl.constexpr = B // K * S
        fill_lane = tl.arange(0, 2 * Q)
        fill_plane = fill_lane // Q
        q = tl.program_id(0) * Q + fill_lane % Q
        fill_dst = tl.where(
            fill_plane == 0, OI, OW.to(tl.pointer_type(tl.int32))
        )
        bits = tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True)
        fill = tl.where(fill_plane == 0, N + q % S, bits)
        tl.store(fill_dst + q // S * P + K + q % S, fill)
    else:
        pid = tl.program_id(0)
        x = pid * B + tl.arange(0, B)
        row = x // K
        col = x % K
        ids = tl.load(IdsPtr + row * I0 + col * I1)
        weights = tl.load(W + row * W0 + col * W1)
        out = row * P + col
        tl.store(OI + out, ids)
        tl.store(OW + out, weights)
        q = pid * (B // K * S) + tl.arange(0, B // K * S)
        shared_out = q // S * P + K + q % S
        tl.store(OI + shared_out, N + q % S)
        tl.store(OW + shared_out, SCALE)


@triton.jit
def _segmented(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    SCALE: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    I0: tl.constexpr = META[5]
    I1: tl.constexpr = META[6]
    W0: tl.constexpr = META[7]
    W1: tl.constexpr = META[8]
    P: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    COPY: tl.constexpr = META[11]
    pid = tl.program_id(0)
    if pid < COPY:
        x = pid * B + tl.arange(0, B)
        row = x // K
        col = x % K
        ids = tl.load(IdsPtr + row * I0 + col * I1)
        weights = tl.load(W + row * W0 + col * W1)
        out = row * P + col
        tl.store(OI + out, ids)
        tl.store(OW + out, weights)
    else:
        x = (pid - COPY) * B + tl.arange(0, B)
        out = x // S * P + K + x % S
        tl.store(OI + out, N + x % S, x < M * S)
        tl.store(OW + out, SCALE, x < M * S)


def fused_append_shared_experts(
    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
):
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return (topk_ids, topk_weights)
    if (
        (m >= 12288 or (m >= 4096 and m % 16 == 0))
        and k == 8
        and (s == 1)
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        p = k + s
        ids = topk_ids.new_empty_strided((m, k + s), (p, 1))
        weights = topk_weights.new_empty_strided((m, k + s), (p, 1))
        blocks = min(12, (m + 511) // 512)
        _persistent_pipeline.run(
            topk_ids,
            topk_weights,
            ids,
            weights,
            (m, k, s, p, N, scale_factor, blocks),
            grid=(blocks,),
            warmup=False,
        )
        return (ids, weights)
    if (
        m >= 4096
        and k == 8
        and (s == 1)
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        p = k + s
        ids = topk_ids.new_empty_strided((m, k + s), (p, 1))
        weights = topk_weights.new_empty_strided((m, k + s), (p, 1))
        _pipeline.run(
            topk_ids,
            topk_weights,
            ids,
            weights,
            (m, k, s, p, N, scale_factor),
            grid=((m + 511) // 512,),
            warmup=False,
        )
        return (ids, weights)
    if (
        k + s <= 64
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        p = k + s
        rows = 8 if m >= 4096 and p <= 32 else 4
        kernel = _raw_dma8 if m >= 4096 and p <= 32 else _raw_dma
        ids = topk_ids.new_empty_strided((m, k + s), (p, 1))
        weights = topk_weights.new_empty_strided((m, k + s), (p, 1))
        if m:
            kernel.run(
                topk_ids,
                topk_weights,
                ids,
                weights,
                (m, k, s, p, N, scale_factor),
                grid=((m + 64 * rows - 1) // (64 * rows),),
                warmup=False,
            )
        return (ids, weights)
    if m * k >= 32768:
        i0, i1 = topk_ids.stride()
        w0, w1 = topk_weights.stride()
        if (
            k + s <= 64
            and i1 > 0
            and (w1 > 0)
            and (i0 == k * i1)
            and (w0 == k * w1)
            and (k * max(i1, w1) <= 64)
        ):
            p = k + s
            rows = 8 if max(p, k * max(i1, w1)) <= 32 else 4
            ids = topk_ids.new_empty_strided((m, k + s), (p, 1))
            weights = topk_weights.new_empty_strided((m, k + s), (p, 1))
            if m:
                _strided_raw_dma.run(
                    topk_ids,
                    topk_weights,
                    ids,
                    weights,
                    (m, k, s, p, N, scale_factor, i1, w1, rows),
                    grid=((m + 64 * rows - 1) // (64 * rows),),
                    warmup=False,
                )
            return (ids, weights)
    if (
        k <= 32
        and s <= k
        and (m >= 8192 or (m >= 512 and m * k >= 4096 and (m * k % 4096 != 0)))
    ):
        i0, i1 = topk_ids.stride()
        w0, w1 = topk_weights.stride()
        p = (m + 256 - 1) // 256 * 256
        ids = topk_ids.new_empty_strided((m, k + s), (1, p))
        weights = topk_weights.new_empty_strided((m, k + s), (1, p))
        if (
            s * 4 <= k
            and (m >= 16384 or k >= 16)
            and (k >= 2)
            and (i1 == 1)
            and (w1 == 1)
            and ((i0 | w0) & 1 == 0)
            and ((topk_ids.data_ptr() | topk_weights.data_ptr()) & 7 == 0)
        ):
            block = 2048 if max(i0, w0) <= 8 else 512
            _pairs.run(
                topk_ids,
                topk_weights,
                ids,
                weights,
                (m, k, s, scale_factor, N, *(i0, i1), *(w0, w1), p, block),
                grid=((m + block - 1) // block, k // 2 + s),
                warmup=False,
            )
        else:
            block = 1024 if max(i0, w0) <= 8 else 512 if m >= 8192 else 256
            _columns.run(
                topk_ids,
                topk_weights,
                ids,
                weights,
                (m, k, s, scale_factor, N, *(i0, i1), *(w0, w1), p, block),
                grid=((m + block - 1) // block, k + s),
                warmup=False,
            )
        return (ids, weights)
    fused = k <= 4096 and s <= k and (m * k >= 4096) and (m * k % 4096 == 0)
    p = 1 << (k + s - 1).bit_length() if fused else k + s
    ids = topk_ids.new_empty_strided((m, k + s), (p, 1))
    weights = topk_weights.new_empty_strided((m, k + s), (p, 1))
    if m:
        if fused:
            if m * k >= 131072 and s * 8 <= k:
                copies = m * k // 4096
                _segmented.run(
                    topk_ids,
                    topk_weights,
                    ids,
                    weights,
                    (
                        m,
                        k,
                        s,
                        scale_factor,
                        N,
                        *topk_ids.stride(),
                        *topk_weights.stride(),
                        p,
                        4096,
                        copies,
                    ),
                    grid=(copies + (m * s + 4096 - 1) // 4096,),
                    warmup=False,
                )
            else:
                _fused.run(
                    topk_ids,
                    topk_weights,
                    ids,
                    weights,
                    (
                        k,
                        s,
                        scale_factor,
                        N,
                        *topk_ids.stride(),
                        *topk_weights.stride(),
                        p,
                        4096,
                    ),
                    grid=(m * k // 4096,),
                    warmup=False,
                )
        elif m * p <= 32:
            _direct.run(
                topk_ids,
                topk_weights,
                ids,
                weights,
                (
                    m,
                    k,
                    s,
                    scale_factor,
                    N,
                    *topk_ids.stride(),
                    *topk_weights.stride(),
                    p,
                    128,
                ),
                grid=(1,),
                warmup=False,
            )
        else:
            _lane.run(
                topk_ids,
                topk_weights,
                ids,
                weights,
                (
                    m,
                    k,
                    s,
                    scale_factor,
                    N,
                    *topk_ids.stride(),
                    *topk_weights.stride(),
                    p,
                    256,
                ),
                grid=((m * p + 255) // 256,),
                warmup=False,
            )
    return (ids, weights)


@triton.jit
def _columns(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    SCALE: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    I0: tl.constexpr = META[5]
    I1: tl.constexpr = META[6]
    W0: tl.constexpr = META[7]
    W1: tl.constexpr = META[8]
    P: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    row = tl.program_id(0) * B + tl.arange(0, B)
    col = tl.program_id(1)
    if col < K:
        ids = tl.load(IdsPtr + row * I0 + col * I1, row < M, other=0)
        weights = tl.load(W + row * W0 + col * W1, row < M, other=0)
        tl.store(OI + col * P + row, ids, row < M)
        tl.store(OW + col * P + row, weights, row < M)
    else:
        tl.store(OI + col * P + row, N + col - K, row < M)
        tl.store(OW + col * P + row, SCALE, row < M)


@triton.jit
def _pairs(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    SCALE: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    I0: tl.constexpr = META[5]
    W0: tl.constexpr = META[7]
    P: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    if M % B == 0:
        lane = tl.arange(0, 2 * B)
        plane = lane // B
        row = tl.program_id(0) * B + lane % B
        pair = tl.program_id(1)
        target = tl.where(plane == 0, OI, OW.to(tl.pointer_type(tl.int32)))
        if pair < K // 2:
            source = tl.where(
                plane == 0,
                IdsPtr.to(tl.pointer_type(tl.uint64)),
                W.to(tl.pointer_type(tl.uint64)),
            )
            s0 = tl.where(plane == 0, I0 // 2, W0 // 2)
            value = tl.load(source + row * s0 + pair)
            low = value.to(tl.uint32).to(tl.int32, bitcast=True)
            high = (value >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
            out = pair * 2 * P + row
            tl.store(target + out, low)
            tl.store(target + out + P, high)
        else:
            shared = pair - K // 2
            bits = tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True)
            shared_value = tl.where(plane == 0, N + shared, bits)
            tl.store(target + (K + shared) * P + row, shared_value)
    else:
        row = tl.program_id(0) * B + tl.arange(0, B)
        pair = tl.program_id(1)
        if pair < K // 2:
            ip = tl.load(
                IdsPtr.to(tl.pointer_type(tl.uint64)) + row * (I0 // 2) + pair,
                row < M,
                other=0,
            )
            wp = tl.load(
                W.to(tl.pointer_type(tl.uint64)) + row * (W0 // 2) + pair,
                row < M,
                other=0,
            )
            low_i = ip.to(tl.uint32).to(tl.int32, bitcast=True)
            high_i = (ip >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
            low_w = wp.to(tl.uint32).to(tl.float32, bitcast=True)
            high_w = (wp >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
            out = pair * 2 * P + row
            tl.store(OI + out, low_i, row < M)
            tl.store(OI + out + P, high_i, row < M)
            tl.store(OW + out, low_w, row < M)
            tl.store(OW + out + P, high_w, row < M)
        else:
            shared = pair - K // 2
            out = (K + shared) * P + row
            tl.store(OI + out, N + shared, row < M)
            tl.store(OW + out, SCALE, row < M)


@triton.jit
def _lane(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    SCALE: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    I0: tl.constexpr = META[5]
    I1: tl.constexpr = META[6]
    W0: tl.constexpr = META[7]
    W1: tl.constexpr = META[8]
    P: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    lane = tl.arange(0, 2 * B)
    plane = lane // B
    x = tl.program_id(0) * B + lane % B
    row = x // P
    col = x % P
    source = tl.where(plane == 0, IdsPtr, W.to(tl.pointer_type(tl.int32)))
    target = tl.where(plane == 0, OI, OW.to(tl.pointer_type(tl.int32)))
    s0 = tl.where(plane == 0, I0, W0)
    s1 = tl.where(plane == 0, I1, W1)
    value = tl.load(
        source + row * s0 + col * s1, (row < M) & (col < K), other=0
    )
    bits = tl.full((), SCALE, tl.float32).to(tl.int32, bitcast=True)
    fill = tl.where(plane == 0, N + col - K, bits)
    value = tl.where(col < K, value, fill)
    tl.store(target + x, value, x < M * P)


@tle.raw.dialect(
    "xpu3",
    source=(
        'extern "C" __attribute__((device)) void p146_raw_dma4(\n'
        "__attribute__((address_space(1))) int* oi,\n"
        "__attribute__((address_space(1))) float* ow,\n"
        "__attribute__((address_space(1))) const int* ids,\n"
        "__attribute__((address_space(1))) const float* weights,\n"
        "int pid, int m, int k, int s, int p, int n, float scale) {\n"
        "    int r = (pid * 64 + __builtin_xpu_core_id()) * 4;\n"
        "    if (r >= m) return;\n"
        "    int count = m - r < 4 ? m - r : 4;\n"
        "    int li[64 * 4], outi[64 * 4];\n"
        "    float lw[64 * 4], outw[64 * 4];\n"
        "    __builtin_xpu_gm2lm_v3(li, ids + (long long)r * k, 0, co"
        "unt * k * 4);\n"
        "    __builtin_xpu_gm2lm_v3(lw, weights + (long long)r * k, 0"
        ", count * k * 4);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "    for (int row = 0; row < count; ++row) {\n"
        "        __builtin_memcpy(outi + row * p, li + row * k, k * 4"
        ");\n"
        "        __builtin_memcpy(outw + row * p, lw + row * k, k * 4"
        ");\n"
        "        for (int col = k; col < p; ++col) {\n"
        "            outi[row * p + col] = n + col - k;\n"
        "            outw[row * p + col] = scale;\n"
        "        }\n"
        "    }\n"
        "    __builtin_xpu_lm2gm_v3(oi + (long long)r * p, outi, 0, ("
        "(count - 1) * p + k + s) * 4);\n"
        "    __builtin_xpu_lm2gm_v3(ow + (long long)r * p, outw, 0, ("
        "(count - 1) * p + k + s) * 4);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "}\n"
    ),
)
def p146_raw_dma4():
    pass


@triton.jit
def _raw_dma(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    tle.raw.call(
        p146_raw_dma4,
        outputs=(OI, OW),
        inputs=(
            IdsPtr,
            W,
            tl.program_id(0),
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.int32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.full((), META[4], tl.int32),
            tl.full((), META[5], tl.float32),
        ),
    )


@tle.raw.dialect(
    "xpu3",
    source=(
        'extern "C" __attribute__((device)) void _stridedp146_raw_dma'
        "4(\n"
        "__attribute__((address_space(1))) int* oi,\n"
        "__attribute__((address_space(1))) float* ow,\n"
        "__attribute__((address_space(1))) const int* ids,\n"
        "__attribute__((address_space(1))) const float* weights,\n"
        "int pid, int m, int k, int s, int p, int n, float scale, int"
        " si, int sw, int rows) {\n"
        "    int r = (pid * 64 + __builtin_xpu_core_id()) * rows;\n"
        "    if (r >= m) return;\n"
        "    int count = m - r < rows ? m - r : rows;\n"
        "    int li[64 * 4], outi[64 * 4];\n"
        "    float lw[64 * 4], outw[64 * 4];\n"
        "    __builtin_xpu_gm2lm_v3(li, ids + (long long)r * k * si, "
        "0, (count * k * si - si + 1) * 4);\n"
        "    __builtin_xpu_gm2lm_v3(lw, weights + (long long)r * k * "
        "sw, 0, (count * k * sw - sw + 1) * 4);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "    for (int row = 0; row < count; ++row) {\n"
        "        for (int col = 0; col < k; ++col) {\n"
        "            outi[row * p + col] = li[(row * k + col) * si];\n"
        "            outw[row * p + col] = lw[(row * k + col) * sw];\n"
        "        }\n"
        "        for (int col = k; col < p; ++col) {\n"
        "            outi[row * p + col] = n + col - k;\n"
        "            outw[row * p + col] = scale;\n"
        "        }\n"
        "    }\n"
        "    __builtin_xpu_lm2gm_v3(oi + (long long)r * p, outi, 0, ("
        "(count - 1) * p + k + s) * 4);\n"
        "    __builtin_xpu_lm2gm_v3(ow + (long long)r * p, outw, 0, ("
        "(count - 1) * p + k + s) * 4);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "}\n"
    ),
)
def _stridedp146_raw_dma4():
    pass


@triton.jit
def _strided_raw_dma(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    tle.raw.call(
        _stridedp146_raw_dma4,
        outputs=(OI, OW),
        inputs=(
            IdsPtr,
            W,
            tl.program_id(0),
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.int32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.full((), META[4], tl.int32),
            tl.full((), META[5], tl.float32),
            tl.full((), META[6], tl.int32),
            tl.full((), META[7], tl.int32),
            tl.full((), META[8], tl.int32),
        ),
    )


@tle.raw.dialect(
    "xpu3",
    source=(
        'extern "C" __attribute__((device)) void p146_raw_dma8(\n'
        "__attribute__((address_space(1))) int* oi,\n"
        "__attribute__((address_space(1))) float* ow,\n"
        "__attribute__((address_space(1))) const int* ids,\n"
        "__attribute__((address_space(1))) const float* weights,\n"
        "int pid, int m, int k, int s, int p, int n, float scale) {\n"
        "    int r = (pid * 64 + __builtin_xpu_core_id()) * 8;\n"
        "    if (r >= m) return;\n"
        "    int count = m - r < 8 ? m - r : 8;\n"
        "    int li[64 * 4], outi[64 * 4];\n"
        "    float lw[64 * 4], outw[64 * 4];\n"
        "    __builtin_xpu_gm2lm_v3(li, ids + (long long)r * k, 0, co"
        "unt * k * 4);\n"
        "    __builtin_xpu_gm2lm_v3(lw, weights + (long long)r * k, 0"
        ", count * k * 4);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "    for (int row = 0; row < count; ++row) {\n"
        "        __builtin_memcpy(outi + row * p, li + row * k, k * 4"
        ");\n"
        "        __builtin_memcpy(outw + row * p, lw + row * k, k * 4"
        ");\n"
        "        for (int col = k; col < p; ++col) {\n"
        "            outi[row * p + col] = n + col - k;\n"
        "            outw[row * p + col] = scale;\n"
        "        }\n"
        "    }\n"
        "    __builtin_xpu_lm2gm_v3(oi + (long long)r * p, outi, 0, ("
        "(count - 1) * p + k + s) * 4);\n"
        "    __builtin_xpu_lm2gm_v3(ow + (long long)r * p, outw, 0, ("
        "(count - 1) * p + k + s) * 4);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "}\n"
    ),
)
def p146_raw_dma8():
    pass


@triton.jit
def _raw_dma8(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    tle.raw.call(
        p146_raw_dma8,
        outputs=(OI, OW),
        inputs=(
            IdsPtr,
            W,
            tl.program_id(0),
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.int32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.full((), META[4], tl.int32),
            tl.full((), META[5], tl.float32),
        ),
    )


@tle.raw.dialect(
    "xpu3",
    source=(
        'extern "C" __attribute__((device)) void p146_pipeline(\n'
        "__attribute__((address_space(1))) int* oi,\n"
        "__attribute__((address_space(1))) float* ow,\n"
        "__attribute__((address_space(1))) const int* ids,\n"
        "__attribute__((address_space(1))) const float* weights,\n"
        "int pid, int m, int k, int s, int p, int n, float scale) {\n"
        "    int r = (pid * 64 + __builtin_xpu_core_id()) * 8;\n"
        "    if (r >= m) return;\n"
        "    int count = m - r < 8 ? m - r : 8;\n"
        "    typedef int vec __attribute__((ext_vector_type(16)));\n"
        "    int li[64] __attribute__((aligned(64)));\n"
        "    int lw[64] __attribute__((aligned(64)));\n"
        "    int outi[144] __attribute__((aligned(64)));\n"
        "    int outw[144] __attribute__((aligned(64)));\n"
        "    int bits;\n"
        "    __builtin_memcpy(&bits, &scale, 4);\n"
        "    __builtin_xpu_gm2lm_v3(li, ids + (long long)r * 8, 0, co"
        "unt * 32);\n"
        "    __builtin_xpu_gm2lm_v3(lw, weights + (long long)r * 8, 0"
        ", count * 32);\n"
        "    vec offset0 = {0};\n"
        "    offset0 = __builtin_xpu2_svor_s_mh(21930u, 4, offset0, o"
        "ffset0);\n"
        "    offset0 = __builtin_xpu2_svor_s_mh(26316u, 8, offset0, o"
        "ffset0);\n"
        "    offset0 = __builtin_xpu2_svor_s_mh(30960u, 16, offset0, "
        "offset0);\n"
        "    offset0 = __builtin_xpu2_svor_s_mh(32512u, 32, offset0, "
        "offset0);\n"
        "    offset0 = __builtin_xpu2_svor_s_mh(32768u, 64, offset0, "
        "offset0);\n"
        "    vec suffix = {0};\n"
        "    suffix = __builtin_xpu2_svor_s_mh(43690u, 4, suffix, suf"
        "fix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(52428u, 8, suffix, suf"
        "fix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(61680u, 16, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(43605u, 32, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(13158u, 64, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(15480u, 128, suffix, s"
        "uffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(16256u, 256, suffix, s"
        "uffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(49152u, 512, suffix, s"
        "uffix);\n"
        "    vec fi = __builtin_xpu2_svor_s_mr1(n, (vec){0});\n"
        "    vec fw = __builtin_xpu2_svor_s_mr1(bits, (vec){0});\n"
        "    __builtin_xpu2_vscatter_mask16_mz(fi, -1, outi, suffix);"
        "\n"
        "    __builtin_xpu2_vscatter_mask16_mz(fw, -1, outw, suffix);"
        "\n"
        "    __builtin_xpu2_mfence(5);\n"
        "    int mask0 = (count > 0 ? 255u : 0u) | (count > 1 ? 65280"
        "u : 0u);\n"
        "    vec vi0 = __builtin_xpu2_vload_mask16_mz(mask0, li + 0, "
        "0);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(vi0, mask0, outi + 0, "
        "offset0);\n"
        "    vec vw0 = __builtin_xpu2_vload_mask16_mz(mask0, lw + 0, "
        "0);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(vw0, mask0, outw + 0, "
        "offset0);\n"
        "    int mask1 = (count > 2 ? 255u : 0u) | (count > 3 ? 65280"
        "u : 0u);\n"
        "    vec vi1 = __builtin_xpu2_vload_mask16_mz(mask1, li + 16,"
        " 0);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(vi1, mask1, outi + 18,"
        " offset0);\n"
        "    vec vw1 = __builtin_xpu2_vload_mask16_mz(mask1, lw + 16,"
        " 0);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(vw1, mask1, outw + 18,"
        " offset0);\n"
        "    int mask2 = (count > 4 ? 255u : 0u) | (count > 5 ? 65280"
        "u : 0u);\n"
        "    vec vi2 = __builtin_xpu2_vload_mask16_mz(mask2, li + 32,"
        " 0);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(vi2, mask2, outi + 36,"
        " offset0);\n"
        "    vec vw2 = __builtin_xpu2_vload_mask16_mz(mask2, lw + 32,"
        " 0);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(vw2, mask2, outw + 36,"
        " offset0);\n"
        "    int mask3 = (count > 6 ? 255u : 0u) | (count > 7 ? 65280"
        "u : 0u);\n"
        "    vec vi3 = __builtin_xpu2_vload_mask16_mz(mask3, li + 48,"
        " 0);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(vi3, mask3, outi + 54,"
        " offset0);\n"
        "    vec vw3 = __builtin_xpu2_vload_mask16_mz(mask3, lw + 48,"
        " 0);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(vw3, mask3, outw + 54,"
        " offset0);\n"
        "    __builtin_xpu2_mfence(1);\n"
        "    __builtin_xpu2_mfence(1);\n"
        "    __builtin_xpu_lm2gm_v3(oi + (long long)r * 9, outi, 0, c"
        "ount * 36);\n"
        "    __builtin_xpu_lm2gm_v3(ow + (long long)r * 9, outw, 0, c"
        "ount * 36);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "}\n"
    ),
)
def p146_pipeline():
    pass


@triton.jit
def _pipeline(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    tle.raw.call(
        p146_pipeline,
        outputs=(OI, OW),
        inputs=(
            IdsPtr,
            W,
            tl.program_id(0),
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.int32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.full((), META[4], tl.int32),
            tl.full((), META[5], tl.float32),
        ),
    )


@tle.raw.dialect(
    "xpu3",
    source=(
        'extern "C" __attribute__((device)) void _persistentp146_pipe'
        "line(\n"
        "__attribute__((address_space(1))) int* oi,\n"
        "__attribute__((address_space(1))) float* ow,\n"
        "__attribute__((address_space(1))) const int* ids,\n"
        "__attribute__((address_space(1))) const float* weights,\n"
        "int pid, int m, int k, int s, int p, int n, float scale, int"
        " blocks) {\n"
        "    int core = __builtin_xpu_core_id();\n"
        "    int plane = core / 32;\n"
        "    int r = (pid * 32 + core % 32) * 16;\n"
        "    if (r >= m) return;\n"
        "    typedef int vec __attribute__((ext_vector_type(16)));\n"
        "    const __attribute__((address_space(1))) int* src = plane"
        " == 0 ? ids : (const __attribute__((address_space(1))) int*)"
        "weights;\n"
        "    __attribute__((address_space(1))) int* dst = plane == 0 "
        "? oi : (__attribute__((address_space(1))) int*)ow;\n"
        "    int in0[128] __attribute__((aligned(64)));\n"
        "    int in1[128] __attribute__((aligned(64)));\n"
        "    int out0[144] __attribute__((aligned(64)));\n"
        "    int out1[144] __attribute__((aligned(64)));\n"
        "    int* current = in0;\n"
        "    int* next = in1;\n"
        "    int* output = out0;\n"
        "    int* spare = out1;\n"
        "    int bits;\n"
        "    __builtin_memcpy(&bits, &scale, 4);\n"
        "    int count = m - r < 16 ? m - r : 16;\n"
        "    __builtin_xpu_gm2lm_v3(current, src + (long long)r * 8, "
        "0, count * 32);\n"
        "    vec prefix = {0};\n"
        "    prefix = __builtin_xpu2_svor_s_mh(21930u, 4, prefix, pre"
        "fix);\n"
        "    prefix = __builtin_xpu2_svor_s_mh(26316u, 8, prefix, pre"
        "fix);\n"
        "    prefix = __builtin_xpu2_svor_s_mh(30960u, 16, prefix, pr"
        "efix);\n"
        "    prefix = __builtin_xpu2_svor_s_mh(32512u, 32, prefix, pr"
        "efix);\n"
        "    prefix = __builtin_xpu2_svor_s_mh(32768u, 64, prefix, pr"
        "efix);\n"
        "    vec suffix = {0};\n"
        "    suffix = __builtin_xpu2_svor_s_mh(43690u, 4, suffix, suf"
        "fix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(52428u, 8, suffix, suf"
        "fix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(61680u, 16, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(43605u, 32, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(13158u, 64, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(15480u, 128, suffix, s"
        "uffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(16256u, 256, suffix, s"
        "uffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(49152u, 512, suffix, s"
        "uffix);\n"
        "    vec fill = __builtin_xpu2_svor_s_mr1(plane == 0 ? n : bi"
        "ts, (vec){0});\n"
        "    __builtin_xpu2_vscatter_mask16_mz(fill, -1, out0 + 0, su"
        "ffix);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(fill, -1, out1 + 0, su"
        "ffix);\n"
        "    __builtin_xpu2_mfence(1);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "    for (; r < m; r += blocks * 512) {\n"
        "        count = m - r < 16 ? m - r : 16;\n"
        "        int next_r = r + blocks * 512;\n"
        "        if (next_r < m) {\n"
        "            int next_count = m - next_r < 16 ? m - next_r : "
        "16;\n"
        "            __builtin_xpu_gm2lm_v3(next, src + (long long)ne"
        "xt_r * 8, 0, next_count * 32);\n"
        "        }\n"
        "        int mask0 = (count > 0 ? 255u : 0u) | (count > 1 ? 6"
        "5280u : 0u);\n"
        "        vec value0 = __builtin_xpu2_vload_mask16_mz(mask0, c"
        "urrent + 0, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value0, mask0, out"
        "put + 0, prefix);\n"
        "        int mask1 = (count > 2 ? 255u : 0u) | (count > 3 ? 6"
        "5280u : 0u);\n"
        "        vec value1 = __builtin_xpu2_vload_mask16_mz(mask1, c"
        "urrent + 16, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value1, mask1, out"
        "put + 18, prefix);\n"
        "        int mask2 = (count > 4 ? 255u : 0u) | (count > 5 ? 6"
        "5280u : 0u);\n"
        "        vec value2 = __builtin_xpu2_vload_mask16_mz(mask2, c"
        "urrent + 32, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value2, mask2, out"
        "put + 36, prefix);\n"
        "        int mask3 = (count > 6 ? 255u : 0u) | (count > 7 ? 6"
        "5280u : 0u);\n"
        "        vec value3 = __builtin_xpu2_vload_mask16_mz(mask3, c"
        "urrent + 48, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value3, mask3, out"
        "put + 54, prefix);\n"
        "        int mask4 = (count > 8 ? 255u : 0u) | (count > 9 ? 6"
        "5280u : 0u);\n"
        "        vec value4 = __builtin_xpu2_vload_mask16_mz(mask4, c"
        "urrent + 64, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value4, mask4, out"
        "put + 72, prefix);\n"
        "        int mask5 = (count > 10 ? 255u : 0u) | (count > 11 ?"
        " 65280u : 0u);\n"
        "        vec value5 = __builtin_xpu2_vload_mask16_mz(mask5, c"
        "urrent + 80, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value5, mask5, out"
        "put + 90, prefix);\n"
        "        int mask6 = (count > 12 ? 255u : 0u) | (count > 13 ?"
        " 65280u : 0u);\n"
        "        vec value6 = __builtin_xpu2_vload_mask16_mz(mask6, c"
        "urrent + 96, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value6, mask6, out"
        "put + 108, prefix);\n"
        "        int mask7 = (count > 14 ? 255u : 0u) | (count > 15 ?"
        " 65280u : 0u);\n"
        "        vec value7 = __builtin_xpu2_vload_mask16_mz(mask7, c"
        "urrent + 112, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value7, mask7, out"
        "put + 126, prefix);\n"
        "        __builtin_xpu2_mfence(1);\n"
        "        __builtin_xpu2_mfence(5);\n"
        "        __builtin_xpu_lm2gm_v3(dst + (long long)r * 9, outpu"
        "t, 0, count * 36);\n"
        "        int* old = current;\n"
        "        current = next;\n"
        "        next = old;\n"
        "        old = output;\n"
        "        output = spare;\n"
        "        spare = old;\n"
        "    }\n"
        "    __builtin_xpu2_mfence(5);\n"
        "}\n"
    ),
)
def _persistentp146_pipeline():
    pass


@tle.raw.dialect(
    "xpu3",
    source=(
        'extern "C" __attribute__((device)) void _persistentp146_full'
        "(\n"
        "__attribute__((address_space(1))) int* oi,\n"
        "__attribute__((address_space(1))) float* ow,\n"
        "__attribute__((address_space(1))) const int* ids,\n"
        "__attribute__((address_space(1))) const float* weights,\n"
        "int pid, int m, int k, int s, int p, int n, float scale, int"
        " blocks) {\n"
        "    int core = __builtin_xpu_core_id();\n"
        "    int plane = core / 32;\n"
        "    int r = (pid * 32 + core % 32) * 16;\n"
        "    if (r >= m) return;\n"
        "    typedef int vec __attribute__((ext_vector_type(16)));\n"
        "    const __attribute__((address_space(1))) int* src = plane"
        " == 0 ? ids : (const __attribute__((address_space(1))) int*)"
        "weights;\n"
        "    __attribute__((address_space(1))) int* dst = plane == 0 "
        "? oi : (__attribute__((address_space(1))) int*)ow;\n"
        "    int in0[128] __attribute__((aligned(64)));\n"
        "    int in1[128] __attribute__((aligned(64)));\n"
        "    int out0[144] __attribute__((aligned(64)));\n"
        "    int out1[144] __attribute__((aligned(64)));\n"
        "    int* current = in0;\n"
        "    int* next = in1;\n"
        "    int* output = out0;\n"
        "    int* spare = out1;\n"
        "    int bits;\n"
        "    __builtin_memcpy(&bits, &scale, 4);\n"
        "    int count = 16;\n"
        "    __builtin_xpu_gm2lm_v3(current, src + (long long)r * 8, "
        "0, count * 32);\n"
        "    vec prefix = {0};\n"
        "    prefix = __builtin_xpu2_svor_s_mh(21930u, 4, prefix, pre"
        "fix);\n"
        "    prefix = __builtin_xpu2_svor_s_mh(26316u, 8, prefix, pre"
        "fix);\n"
        "    prefix = __builtin_xpu2_svor_s_mh(30960u, 16, prefix, pr"
        "efix);\n"
        "    prefix = __builtin_xpu2_svor_s_mh(32512u, 32, prefix, pr"
        "efix);\n"
        "    prefix = __builtin_xpu2_svor_s_mh(32768u, 64, prefix, pr"
        "efix);\n"
        "    vec suffix = {0};\n"
        "    suffix = __builtin_xpu2_svor_s_mh(43690u, 4, suffix, suf"
        "fix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(52428u, 8, suffix, suf"
        "fix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(61680u, 16, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(43605u, 32, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(13158u, 64, suffix, su"
        "ffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(15480u, 128, suffix, s"
        "uffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(16256u, 256, suffix, s"
        "uffix);\n"
        "    suffix = __builtin_xpu2_svor_s_mh(49152u, 512, suffix, s"
        "uffix);\n"
        "    vec fill = __builtin_xpu2_svor_s_mr1(plane == 0 ? n : bi"
        "ts, (vec){0});\n"
        "    __builtin_xpu2_vscatter_mask16_mz(fill, -1, out0 + 0, su"
        "ffix);\n"
        "    __builtin_xpu2_vscatter_mask16_mz(fill, -1, out1 + 0, su"
        "ffix);\n"
        "    __builtin_xpu2_mfence(1);\n"
        "    __builtin_xpu2_mfence(5);\n"
        "    for (; r < m; r += blocks * 512) {\n"
        "        count = 16;\n"
        "        int next_r = r + blocks * 512;\n"
        "        if (next_r < m) {\n"
        "            int next_count = 16;\n"
        "            __builtin_xpu_gm2lm_v3(next, src + (long long)ne"
        "xt_r * 8, 0, next_count * 32);\n"
        "        }\n"
        "        int mask0 = 65535u;\n"
        "        vec value0 = __builtin_xpu2_vload_mask16_mz(mask0, c"
        "urrent + 0, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value0, mask0, out"
        "put + 0, prefix);\n"
        "        int mask1 = 65535u;\n"
        "        vec value1 = __builtin_xpu2_vload_mask16_mz(mask1, c"
        "urrent + 16, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value1, mask1, out"
        "put + 18, prefix);\n"
        "        int mask2 = 65535u;\n"
        "        vec value2 = __builtin_xpu2_vload_mask16_mz(mask2, c"
        "urrent + 32, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value2, mask2, out"
        "put + 36, prefix);\n"
        "        int mask3 = 65535u;\n"
        "        vec value3 = __builtin_xpu2_vload_mask16_mz(mask3, c"
        "urrent + 48, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value3, mask3, out"
        "put + 54, prefix);\n"
        "        int mask4 = 65535u;\n"
        "        vec value4 = __builtin_xpu2_vload_mask16_mz(mask4, c"
        "urrent + 64, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value4, mask4, out"
        "put + 72, prefix);\n"
        "        int mask5 = 65535u;\n"
        "        vec value5 = __builtin_xpu2_vload_mask16_mz(mask5, c"
        "urrent + 80, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value5, mask5, out"
        "put + 90, prefix);\n"
        "        int mask6 = 65535u;\n"
        "        vec value6 = __builtin_xpu2_vload_mask16_mz(mask6, c"
        "urrent + 96, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value6, mask6, out"
        "put + 108, prefix);\n"
        "        int mask7 = 65535u;\n"
        "        vec value7 = __builtin_xpu2_vload_mask16_mz(mask7, c"
        "urrent + 112, 0);\n"
        "        __builtin_xpu2_vscatter_mask16_mz(value7, mask7, out"
        "put + 126, prefix);\n"
        "        __builtin_xpu2_mfence(1);\n"
        "        __builtin_xpu2_mfence(5);\n"
        "        __builtin_xpu_lm2gm_v3(dst + (long long)r * 9, outpu"
        "t, 0, count * 36);\n"
        "        int* old = current;\n"
        "        current = next;\n"
        "        next = old;\n"
        "        old = output;\n"
        "        output = spare;\n"
        "        spare = old;\n"
        "    }\n"
        "    __builtin_xpu2_mfence(5);\n"
        "}\n"
    ),
)
def _persistentp146_full():
    pass


@triton.jit
def _persistent_pipeline(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    if META[0] % 16 == 0:
        tle.raw.call(
            _persistentp146_full,
            outputs=(OI, OW),
            inputs=(
                IdsPtr,
                W,
                tl.program_id(0),
                tl.full((), META[0], tl.int32),
                tl.full((), META[1], tl.int32),
                tl.full((), META[2], tl.int32),
                tl.full((), META[3], tl.int32),
                tl.full((), META[4], tl.int32),
                tl.full((), META[5], tl.float32),
                tl.full((), META[6], tl.int32),
            ),
        )
    else:
        tle.raw.call(
            _persistentp146_pipeline,
            outputs=(OI, OW),
            inputs=(
                IdsPtr,
                W,
                tl.program_id(0),
                tl.full((), META[0], tl.int32),
                tl.full((), META[1], tl.int32),
                tl.full((), META[2], tl.int32),
                tl.full((), META[3], tl.int32),
                tl.full((), META[4], tl.int32),
                tl.full((), META[5], tl.float32),
                tl.full((), META[6], tl.int32),
            ),
        )


__all__ = ["fused_append_shared_experts"]
