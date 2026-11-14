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
def _load_rope(Q, index, mask):
    lo = tl.load(Q + index, mask, other=0).to(tl.uint16)
    hi = tl.load(Q + index + 1, mask, other=0).to(tl.uint16)
    return (lo | hi << 8).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize_on_alignment=["Q", "S", "P", "Dest"])
def _small_kernel(Q, S, P, Dest, META: tl.constexpr):
    Q = Q.to(tl.pointer_type(tl.uint8))
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    QS: tl.constexpr = META[2]
    PS: tl.constexpr = META[3]
    SS: tl.constexpr = QS // 4
    chunk = tl.cdiv(N, tl.num_programs(0) * 1) * 1
    start = tl.program_id(0) * chunk
    stop = tl.minimum(start + chunk, N)
    exponents = tl.arange(0, 16)
    table = (1 << tl.maximum(exponents - 1, 0)).to(tl.float32) * 0.001953125
    for base_row in range(start, stop, 1):
        for offset in tl.static_range(1):
            row = base_row + offset
            if N % 1 == 0 or row < N:
                src = tl.load(P + row * PS).to(tl.int64)
                src = tl.where(src < 0, src + M, src)
                group = tl.arange(0, 4)
                col = group[:, None] * 128 + tl.arange(0, 128)[None, :]
                q = tl.load(Q + src * QS + tl.arange(0, 512)).to(tl.int32)
                qe = q >> 3 & 15
                qm = q & 7
                qfactor = tl.gather(table, qe, axis=0)
                decoded = (
                    tl.where(qe == 0, qm, qm + 8).to(tl.float32) * qfactor
                )
                decoded = tl.where(q & 128 != 0, -decoded, decoded)
                decoded = tl.where(q & 127 == 127, float("nan"), decoded)
                value = decoded.reshape(4, 128)
                scale = tl.load(S + src * SS + 128 + group)
                tl.store(
                    Dest
                    + (row.to(tl.int64) if N * 576 >= 2147483648 else row)
                    * 576
                    + col,
                    value * scale[:, None],
                )
                rc = tl.arange(0, 64)
                rope = _load_rope(Q, src * QS + 528 + rc * 2, True)
                tl.store(
                    Dest
                    + (row.to(tl.int64) if N * 576 >= 2147483648 else row)
                    * 576
                    + 512
                    + rc,
                    rope,
                )


@triton.jit(do_not_specialize_on_alignment=["Q", "S", "R", "P", "Dest"])
def _kernel(
    Q,
    S,
    R,
    P,
    Dest,
    META: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    N: tl.constexpr = META[0]
    if N >= 4096:
        M: tl.constexpr = META[1]
        QS: tl.constexpr = META[2]
        PS: tl.constexpr = META[3]
        BR: tl.constexpr = 16
        SS: tl.constexpr = QS // 4
        RS: tl.constexpr = QS // 2
        chunk = tl.cdiv(N, tl.num_programs(0) * BR) * BR
        start = tl.program_id(0) * chunk
        stop = tl.minimum(start + chunk, N)
        byte = tl.arange(0, 256)
        a = byte & 127
        e = a >> 3
        m = a & 7
        factor = (1 << tl.maximum(e - 1, 0)).to(tl.float32) * 0.001953125
        table = tl.where(e == 0, m, m + 8).to(tl.float32) * factor
        table = tl.where(byte & 128 != 0, -table, table)
        table = tl.where(a == 127, float("nan"), table)
        group = tl.arange(0, 4)
        col = tl.arange(0, 512)
        rc = tl.arange(0, 64)
        for base_row in range(start, stop, BR):
            rows = base_row + tl.arange(0, BR)
            live = rows < stop
            src = tl.load(P + rows * PS, live, other=0).to(tl.int64)
            src = tl.where(src < 0, src + M, src)
            q = tl.load(
                Q + src[:, None] * QS + col[None, :], live[:, None], other=0
            ).to(tl.int32)
            value = tl.gather(table, q.reshape(BR * 512), axis=0).reshape(
                BR, 4, 128
            )
            scale = tl.load(
                S + src[:, None] * SS + 128 + group[None, :],
                live[:, None],
                other=0,
            )
            product = (value * scale[:, :, None]).reshape(BR, 512)
            offsets = rows.to(tl.int64) * 576
            tl.store(
                Dest + offsets[:, None] + col[None, :], product, live[:, None]
            )
            rope = tl.load(
                R + src[:, None] * RS + 264 + rc[None, :],
                live[:, None],
                other=0,
            )
            tl.store(
                Dest + offsets[:, None] + 512 + rc[None, :],
                rope,
                live[:, None],
            )
    else:
        M: tl.constexpr = META[1]
        QS: tl.constexpr = META[2]
        PS: tl.constexpr = META[3]
        SS: tl.constexpr = QS // 4
        RS: tl.constexpr = QS // 2
        chunk = tl.cdiv(N, tl.num_programs(0) * 1) * 1
        start = tl.program_id(0) * chunk
        stop = tl.minimum(start + chunk, N)
        for base_row in range(start, stop, 1):
            for offset in tl.static_range(1):
                row = base_row + offset
                if N % 1 == 0 or row < N:
                    src = tl.load(P + row * PS).to(tl.int64)
                    src = tl.where(src < 0, src + M, src)
                    group = tl.arange(0, 4)
                    col = group[:, None] * 128 + tl.arange(0, 128)[None, :]
                    q = tl.load(Q + src * QS + tl.arange(0, 512)).to(tl.int32)
                    byte = tl.arange(0, 256)
                    a = byte & 127
                    e = a >> 3
                    m = a & 7
                    factor = (1 << tl.maximum(e - 1, 0)).to(
                        tl.float32
                    ) * 0.001953125
                    table = tl.where(e == 0, m, m + 8).to(tl.float32) * factor
                    table = tl.where(byte & 128 != 0, -table, table)
                    table = tl.where(a == 127, float("nan"), table)
                    value = tl.gather(table, q, axis=0).reshape(4, 128)
                    scale = tl.load(S + src * SS + 128 + group)
                    tl.store(
                        Dest
                        + (row.to(tl.int64) if N * 576 >= 2147483648 else row)
                        * 576
                        + col,
                        value * scale[:, None],
                    )
                    rc = tl.arange(0, 64)
                    rope = tl.load(R + src * RS + 264 + rc)
                    tl.store(
                        Dest
                        + (row.to(tl.int64) if N * 576 >= 2147483648 else row)
                        * 576
                        + 512
                        + rc,
                        rope,
                    )


def dequantize_k_cache_paged(
    quant_k_cache, page_table_1_flattened, group_size=128
):
    quant_k_cache = quant_k_cache.view(torch.uint8)
    if quant_k_cache.ndim == 3 and quant_k_cache.shape[1] == 1:
        cache = quant_k_cache.contiguous()
    else:
        cache = quant_k_cache.view(-1, quant_k_cache.shape[-1]).contiguous()
    strides = cache.stride()
    if strides[0] % 4 or strides[-2] % 4:
        cache = cache.view(-1, cache.shape[-1])
        strides = cache.stride()
    if cache.storage_offset() % 4:
        cache = cache.clone()
        strides = cache.stride()
    n = page_table_1_flattened.numel()
    out = cache.new_empty((n, 1, 576), dtype=torch.bfloat16)
    if n:
        scales = cache.view(torch.float32)
        if n < 4096:
            _small_kernel.run(
                cache,
                scales,
                page_table_1_flattened,
                out,
                (
                    n,
                    cache.shape[0],
                    strides[0],
                    page_table_1_flattened.stride(0),
                ),
                num_warps=4,
                grid=(min(n, 40),),
                warmup=False,
            )
        else:
            rope = cache.view(torch.bfloat16)
            _kernel.run(
                cache,
                scales,
                rope,
                page_table_1_flattened,
                out,
                (
                    n,
                    cache.shape[0],
                    strides[0],
                    page_table_1_flattened.stride(0),
                ),
                num_warps=4,
                grid=(min(n, 40),),
                warmup=False,
            )
    return out


__all__ = ["dequantize_k_cache_paged"]
