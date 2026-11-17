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

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_inv_rms_tile(
    X,
    Q,
    K,
    M,
    EPS: tl.constexpr,
    C: tl.constexpr,
    RT: tl.constexpr,
    INV_C: tl.constexpr,
    FULL_BLOCKS,
):
    """2D tile path (power-of-two C): [RT, C] block_ptr loads of the q/k
    planes lower to the ``tops_slice_async`` DMA (vs the per-512B
    ``tar_load`` descriptor-relay chain of the 1D arange form), ~3.9x at
    M=4096/C=2048. Full blocks take the tile path; one trailing program
    covers M % RT rows with a 1D loop (block_ptr without boundary_check
    must never run past M)."""
    pid = tl.program_id(0)
    if pid < FULL_BLOCKS:
        row0 = pid * RT
        qbp = tl.make_block_ptr(
            base=X,
            shape=(M, 3 * C),
            strides=(3 * C, 1),
            offsets=(row0, 0),
            block_shape=(RT, C),
            order=(1, 0),
        )
        kbp = tl.make_block_ptr(
            base=X,
            shape=(M, 3 * C),
            strides=(3 * C, 1),
            offsets=(row0, C),
            block_shape=(RT, C),
            order=(1, 0),
        )
        qv = tl.load(qbp).to(tl.float32)
        kv = tl.load(kbp).to(tl.float32)
        qi = tl.rsqrt(tl.sum(qv * qv, 1) * INV_C + EPS)
        ki = tl.rsqrt(tl.sum(kv * kv, 1) * INV_C + EPS)
        qobp = tl.make_block_ptr(
            base=Q,
            shape=(M,),
            strides=(1,),
            offsets=(row0,),
            block_shape=(RT,),
            order=(0,),
        )
        kobp = tl.make_block_ptr(
            base=K,
            shape=(M,),
            strides=(1,),
            offsets=(row0,),
            block_shape=(RT,),
            order=(0,),
        )
        tl.store(qobp, qi)
        tl.store(kobp, ki)
    else:
        base_row = FULL_BLOCKS * RT
        cols = tl.arange(0, C)
        for i in tl.static_range(RT):
            row = base_row + i
            if row < M:
                qv = tl.load(X + (row * 3) * C + cols).to(tl.float32)
                kv = tl.load(X + (row * 3 + 1) * C + cols).to(tl.float32)
                qi = tl.rsqrt(tl.sum(qv * qv, 0) * INV_C + EPS)
                ki = tl.rsqrt(tl.sum(kv * kv, 0) * INV_C + EPS)
                tl.store(Q + row, qi)
                tl.store(K + row, ki)


@triton.jit
def _qk_inv_rms_row(
    X,
    Q,
    K,
    M,
    EPS: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    RT: tl.constexpr,
    INV_C: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """1D row-batch path (non-power-of-two C fallback): per-row affine
    loads with a column mask; fp32 mixed-width squares and a per-row
    hardware redsum."""
    pid = tl.program_id(0)
    for i in tl.static_range(RT):
        row = pid * RT + i
        if EVEN_M or row < M:
            cols = tl.arange(0, BLOCK)
            if EVEN_N:
                qv = tl.load(X + (row * 3) * C + cols).to(tl.float32)
                kv = tl.load(X + (row * 3 + 1) * C + cols).to(tl.float32)
            else:
                cmask = cols < C
                qv = tl.load(X + (row * 3) * C + cols, cmask, 0).to(tl.float32)
                kv = tl.load(X + (row * 3 + 1) * C + cols, cmask, 0).to(
                    tl.float32
                )
            qi = tl.rsqrt(tl.sum(qv * qv, 0) * INV_C + EPS)
            ki = tl.rsqrt(tl.sum(kv * kv, 0) * INV_C + EPS)
            tl.store(Q + row, qi)
            tl.store(K + row, ki)


_MIN_BATCH_BLOCKS = 32
_MAX_ROWS_PER_PROGRAM = 16
_GRID_X_MAX = 65535
_DSM_DUAL_PROGRAM_BUDGET = 458752  # 448KB: 2 programs/SP threshold (gcu400)


def _pick_rt_tile(m, c):
    if m <= 128:
        rt = 4
    elif m <= 768:
        rt = 16
    else:
        rt = 32
    # DSM guard (measured: two bf16 tiles + 2KB metadata): stay inside the
    # 2-programs-per-SP window; shrink RT for wide C.
    while rt > 1 and 2 * rt * c * 2 + 2048 > _DSM_DUAL_PROGRAM_BUDGET:
        rt //= 2
    while rt < 64 and triton.cdiv(m, rt) > _GRID_X_MAX:
        rt *= 2
    return rt


def _pick_rt_row(m):
    rt = 1
    while (
        rt < _MAX_ROWS_PER_PROGRAM
        and triton.cdiv(m, rt * 2) >= _MIN_BATCH_BLOCKS
    ):
        rt *= 2
    while rt < _MAX_ROWS_PER_PROGRAM and triton.cdiv(m, rt) > _GRID_X_MAX:
        rt *= 2
    return rt


def fused_qk_inv_rms(qkv, eps):
    b, n, _, h, d = qkv.shape
    c = h * d
    m = b * n
    q = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    k = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    if m:
        inv_c = 1.0 / c
        if c & (c - 1) == 0:
            rt = _pick_rt_tile(m, c)
            full_blocks = m // rt
            grid = (full_blocks + (1 if m % rt else 0),)
            _qk_inv_rms_tile.run(
                qkv,
                q,
                k,
                m,
                eps,
                c,
                rt,
                inv_c,
                full_blocks,
                num_warps=1,
                grid=grid,
                warmup=False,
            )
        else:
            block = triton.next_power_of_2(c)
            rt = _pick_rt_row(m)
            blocks = m // rt if m % rt == 0 else triton.cdiv(m, rt)
            _qk_inv_rms_row.run(
                qkv,
                q,
                k,
                m,
                eps,
                c,
                block,
                rt,
                inv_c,
                block == c,
                m % rt == 0,
                num_warps=1,
                grid=(blocks,),
                warmup=False,
            )
    return q, k


__all__ = ["fused_qk_inv_rms"]
