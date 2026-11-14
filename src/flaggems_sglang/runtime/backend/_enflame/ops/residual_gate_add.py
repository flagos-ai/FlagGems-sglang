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
def _residual_gate_add_flat(
    R,
    U,
    G,
    O,
    n,
    nblk,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
):
    for blk in range(tl.program_id(0), nblk, tl.num_programs(0)):
        if WIDE:
            offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        else:
            offs = blk * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            r = tl.load(R + offs)
            u = tl.load(U + offs)
            g = tl.load(G + offs)
            tl.store(O + offs, r + (u * g).to(O.dtype.element_ty))
        else:
            m = offs < n
            r = tl.load(R + offs, m, 0)
            u = tl.load(U + offs, m, 0)
            g = tl.load(G + offs, m, 0)
            tl.store(O + offs, r + (u * g).to(O.dtype.element_ty), m)


@triton.jit
def _residual_gate_add_broadcast(
    R,
    U,
    G,
    O,
    ROWS,
    D,
    cols,
    B: tl.constexpr,
    MASKCOL: tl.constexpr,
    WIDE: tl.constexpr,
):
    for job in range(tl.program_id(0), tl.cdiv(ROWS, 8) * cols, tl.num_programs(0)):
        rg = job // cols
        cb = job % cols
        col = cb * B
        if WIDE:
            r0 = (rg * 8).to(tl.int64)
            last = (ROWS - 1).to(tl.int64)
        else:
            r0 = rg * 8
            last = ROWS - 1
        rw0 = tl.minimum(r0 + 0, last)
        rw1 = tl.minimum(r0 + 1, last)
        rw2 = tl.minimum(r0 + 2, last)
        rw3 = tl.minimum(r0 + 3, last)
        rw4 = tl.minimum(r0 + 4, last)
        rw5 = tl.minimum(r0 + 5, last)
        rw6 = tl.minimum(r0 + 6, last)
        rw7 = tl.minimum(r0 + 7, last)
        ar = tl.arange(0, B)
        b0 = rw0 * D + col
        b1 = rw1 * D + col
        b2 = rw2 * D + col
        b3 = rw3 * D + col
        b4 = rw4 * D + col
        b5 = rw5 * D + col
        b6 = rw6 * D + col
        b7 = rw7 * D + col
        if MASKCOL:
            cm = col + ar < D
            g = tl.load(G + col + ar, cm, 0)
            r0v = tl.load(R + b0 + ar, cm, 0)
            r1v = tl.load(R + b1 + ar, cm, 0)
            r2v = tl.load(R + b2 + ar, cm, 0)
            r3v = tl.load(R + b3 + ar, cm, 0)
            r4v = tl.load(R + b4 + ar, cm, 0)
            r5v = tl.load(R + b5 + ar, cm, 0)
            r6v = tl.load(R + b6 + ar, cm, 0)
            r7v = tl.load(R + b7 + ar, cm, 0)
            u0v = tl.load(U + b0 + ar, cm, 0)
            u1v = tl.load(U + b1 + ar, cm, 0)
            u2v = tl.load(U + b2 + ar, cm, 0)
            u3v = tl.load(U + b3 + ar, cm, 0)
            u4v = tl.load(U + b4 + ar, cm, 0)
            u5v = tl.load(U + b5 + ar, cm, 0)
            u6v = tl.load(U + b6 + ar, cm, 0)
            u7v = tl.load(U + b7 + ar, cm, 0)
            tl.store(O + b0 + ar, r0v + (u0v * g).to(O.dtype.element_ty), cm)
            tl.store(O + b1 + ar, r1v + (u1v * g).to(O.dtype.element_ty), cm)
            tl.store(O + b2 + ar, r2v + (u2v * g).to(O.dtype.element_ty), cm)
            tl.store(O + b3 + ar, r3v + (u3v * g).to(O.dtype.element_ty), cm)
            tl.store(O + b4 + ar, r4v + (u4v * g).to(O.dtype.element_ty), cm)
            tl.store(O + b5 + ar, r5v + (u5v * g).to(O.dtype.element_ty), cm)
            tl.store(O + b6 + ar, r6v + (u6v * g).to(O.dtype.element_ty), cm)
            tl.store(O + b7 + ar, r7v + (u7v * g).to(O.dtype.element_ty), cm)
        else:
            g = tl.load(G + col + ar)
            r0v = tl.load(R + b0 + ar)
            r1v = tl.load(R + b1 + ar)
            r2v = tl.load(R + b2 + ar)
            r3v = tl.load(R + b3 + ar)
            r4v = tl.load(R + b4 + ar)
            r5v = tl.load(R + b5 + ar)
            r6v = tl.load(R + b6 + ar)
            r7v = tl.load(R + b7 + ar)
            u0v = tl.load(U + b0 + ar)
            u1v = tl.load(U + b1 + ar)
            u2v = tl.load(U + b2 + ar)
            u3v = tl.load(U + b3 + ar)
            u4v = tl.load(U + b4 + ar)
            u5v = tl.load(U + b5 + ar)
            u6v = tl.load(U + b6 + ar)
            u7v = tl.load(U + b7 + ar)
            tl.store(O + b0 + ar, r0v + (u0v * g).to(O.dtype.element_ty))
            tl.store(O + b1 + ar, r1v + (u1v * g).to(O.dtype.element_ty))
            tl.store(O + b2 + ar, r2v + (u2v * g).to(O.dtype.element_ty))
            tl.store(O + b3 + ar, r3v + (u3v * g).to(O.dtype.element_ty))
            tl.store(O + b4 + ar, r4v + (u4v * g).to(O.dtype.element_ty))
            tl.store(O + b5 + ar, r5v + (u5v * g).to(O.dtype.element_ty))
            tl.store(O + b6 + ar, r6v + (u6v * g).to(O.dtype.element_ty))
            tl.store(O + b7 + ar, r7v + (u7v * g).to(O.dtype.element_ty))


def residual_gate_add(residual, update, gate):
    out = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    n = residual.numel()
    if n:
        d = residual.shape[-1]
        rows = n // d
        wide = n >= 2147483648
        if gate.numel() == n:
            cap = 8192 if residual.element_size() <= 2 else 4096
            block = min(cap, triton.next_power_of_2(max(n, 256)))
            nblk = triton.cdiv(n, block)
            _residual_gate_add_flat[(min(48, nblk),)](
                residual,
                update,
                gate,
                out,
                n,
                nblk,
                block,
                n % block == 0,
                wide,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            if d >= 1024:
                b = 2048 if d % 2048 == 0 else 1024
            else:
                b = max(16, triton.next_power_of_2(d))
            cols = triton.cdiv(d, b)
            jobs = triton.cdiv(rows, 8) * cols
            _residual_gate_add_broadcast[(min(48, jobs),)](
                residual,
                update,
                gate,
                out,
                rows,
                d,
                cols,
                b,
                d % b != 0,
                wide,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out


__all__ = ["residual_gate_add"]
