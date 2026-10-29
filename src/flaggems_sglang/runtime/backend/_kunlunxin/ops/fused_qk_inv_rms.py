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
def _add_pair(a, b, c, d):
    return (a + c, b + d)


@triton.jit
def _tuple_group(
    X,
    Q,
    K,
    M: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, COLS)
    off = row[:, None] * (3 * C) + col[None, :]
    mask = (row[:, None] < M) & (col[None, :] < C)
    q = tl.load(X + off, mask, 0).to(tl.float32)
    k = tl.load(X + off + C, mask, 0).to(tl.float32)
    qs, ks = tl.reduce((q * q, k * k), 1, _add_pair)
    tl.store(Q + row, tl.rsqrt(qs / C + EPS), row < M)
    tl.store(K + row, tl.rsqrt(ks / C + EPS), row < M)


@triton.jit
def _interleaved(
    X,
    O,
    M: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    logical = tl.program_id(0) * (2 * ROWS) + tl.arange(0, 2 * ROWS)
    row = logical // 2
    component = logical % 2
    col = tl.arange(0, COLS)
    off = (row[:, None] * 3 + component[:, None]) * C + col[None, :]
    value = tl.load(X + off, (row[:, None] < M) & (col[None, :] < C), 0).to(
        tl.float32
    )
    inv = tl.rsqrt(tl.sum(value * value, 1) / C + EPS)
    tl.store(O + logical, inv, logical < 2 * M)


def _baseline(qkv, eps):
    b, n, _, h, d = qkv.shape
    m, c = (b * n, h * d)
    cols = triton.next_power_of_2(c)
    buffer = (
        8192
        if m > 512 and 1024 <= c <= 2048
        else 1024 if 256 <= m <= 512 and 1024 <= c <= 2048 else 512
    )
    if 512 < m < 8192 and c <= 2048 or (m >= 8192 and 1024 <= c <= 2048):
        output = torch.empty((b, n, 2), device=qkv.device, dtype=torch.float32)
        _interleaved.run(
            qkv,
            output,
            m,
            c,
            eps,
            32,
            cols,
            grid=(triton.cdiv(m, 32),),
            warmup=False,
            num_warps=4,
            buffer_size_limit=buffer,
        )
        return output.unbind(-1)
    rows = (
        64
        if m >= 4096 and c <= 2048 or (m == 512 and c == 2048)
        else 32 if m >= 256 and c <= 2048 else 4
    )
    q = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    k = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    _tuple_group.run(
        qkv,
        q,
        k,
        m,
        c,
        eps,
        rows,
        cols,
        grid=(triton.cdiv(m, rows),),
        warmup=False,
        num_warps=4,
        buffer_size_limit=buffer,
    )
    return (q, k)


@triton.jit
def _inv_stream(X, Q, K, N: tl.constexpr, D: tl.constexpr, E: tl.constexpr):
    c = tl.arange(0, 512)
    which = tl.program_id(1)
    out = tl.where(which == 0, Q, K)
    for tile in range(tl.program_id(0), N // 64, tl.num_programs(0)):
        row = tile * 64 + tl.arange(0, 64)
        base = row[:, None] * (3 * D) + which * D
        x = tl.load(X + base + c[None, :]).to(tl.float32)
        total = x * x
        for start in tl.range(512, D, 512, num_stages=1):
            x = tl.load(X + base + start + c[None, :]).to(tl.float32)
            total += x * x
        inv = tl.rsqrt(tl.sum(total, 1) / D + E)
        tl.store(out + row, inv)


def fused_qk_inv_rms(qkv, eps):
    b, n, _, h, d = qkv.shape
    width = h * d
    rows = b * n
    if not (
        qkv.dtype == torch.bfloat16
        and width == 2048
        and (rows >= 512)
        and (rows % 64 == 0)
    ):
        return _baseline(qkv, eps)
    q = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    k = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    _inv_stream.run(
        qkv,
        q,
        k,
        rows,
        width,
        float(eps),
        grid=(min(rows // 64, 12), 2),
        warmup=False,
        groups_per_cluster=64,
        isCloseUnrollControl=True,
        enable_fp_fusion=False,
    )
    return (q, k)


__all__ = ["fused_qk_inv_rms"]
