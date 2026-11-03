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
def _grouped(
    X,
    Q,
    K,
    M: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    col = tl.arange(0, COLS)
    for tile in range(tl.program_id(0), tl.cdiv(M, ROWS), tl.num_programs(0)):
        row = tile * ROWS + tl.arange(0, ROWS)
        off = row[:, None] * (3 * C) + col[None, :]
        mask = (row[:, None] < M) & (col[None, :] < C)
        q = tl.load(X + off, mask, 0).to(tl.float32)
        k = tl.load(X + off + C, mask, 0).to(tl.float32)
        tl.store(Q + row, tl.rsqrt(tl.sum(q * q, 1) / C + EPS), row < M)
        tl.store(K + row, tl.rsqrt(tl.sum(k * k, 1) / C + EPS), row < M)


@triton.jit
def _grouped_full(
    X, Q, K, C: tl.constexpr, EPS: tl.constexpr, ROWS: tl.constexpr
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    off = row[:, None] * (3 * C) + tl.arange(0, C)[None, :]
    q = tl.load(X + off).to(tl.float32)
    k = tl.load(X + off + C).to(tl.float32)
    tl.store(Q + row, tl.rsqrt(tl.sum(q * q, 1) / C + EPS))
    tl.store(K + row, tl.rsqrt(tl.sum(k * k, 1) / C + EPS))


@triton.jit
def _persist_full(
    X,
    Q,
    K,
    TILES: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
):
    for block in tl.range(
        tl.program_id(0), TILES, tl.num_programs(0), num_stages=1
    ):
        row = block * ROWS + tl.arange(0, ROWS)
        off = row[:, None] * (3 * C) + tl.arange(0, C)[None, :]
        q = tl.load(X + off).to(tl.float32)
        k = tl.load(X + off + C).to(tl.float32)
        tl.store(Q + row, tl.rsqrt(tl.sum(q * q, 1) / C + EPS))
        tl.store(K + row, tl.rsqrt(tl.sum(k * k, 1) / C + EPS))


def fused_qk_inv_rms(qkv, eps):
    b, n, _, h, d = qkv.shape
    m, c = (b * n, h * d)
    q = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    k = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    if not m:
        return q, k
    block = triton.next_power_of_2(c)
    exact = c == block
    if m >= 4096:
        rows = min(8, max(1, 16384 // block))
        if exact and m % rows == 0:
            _persist_full[min(m // rows, 40),](
                qkv, q, k, m // rows, c, eps, rows, num_warps=4, num_stages=1
            )
        else:
            _grouped[min(triton.cdiv(m, rows), 40),](
                qkv, q, k, m, c, eps, rows, block, num_warps=4, num_stages=1
            )
    else:
        rows = max(1, min(4, 16384 // block))
        if exact and m % rows == 0:
            _grouped_full[triton.cdiv(m, rows),](
                qkv, q, k, c, eps, rows, num_warps=4
            )
        else:
            _grouped[min(triton.cdiv(m, rows), 40),](
                qkv, q, k, m, c, eps, rows, block, num_warps=4
            )
    return (q, k)


__all__ = ["fused_qk_inv_rms"]
