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
# implied. See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _group(
    R,
    U,
    G,
    O,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    BROADCAST: tl.constexpr,
    B: tl.constexpr,
    RG: tl.constexpr,
    WIDE: tl.constexpr,
):
    cols = tl.cdiv(D, B)
    for job in range(
        tl.program_id(0), tl.cdiv(ROWS, RG) * cols, tl.num_programs(0)
    ):
        tile = job.to(tl.int64 if WIDE else tl.int32)
        row = (tile // cols) * RG
        col = (tile % cols) * B
        rp = tl.make_block_ptr(
            R, (ROWS, D), (D, 1), (row, col), (RG, B), (1, 0)
        )
        up = tl.make_block_ptr(
            U, (ROWS, D), (D, 1), (row, col), (RG, B), (1, 0)
        )
        r = tl.load(rp, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        u = tl.load(up, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        if BROADCAST:
            gp = tl.make_block_ptr(G, (D,), (1,), (col,), (B,), (0,))
            g = tl.load(gp, boundary_check=(0,), padding_option="zero").to(
                tl.float32
            )[None, :]
        else:
            gp = tl.make_block_ptr(
                G, (ROWS, D), (D, 1), (row, col), (RG, B), (1, 0)
            )
            g = tl.load(gp, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
        product = (u * g).to(O.dtype.element_ty).to(tl.float32)
        value = r + product
        op = tl.make_block_ptr(
            O, (ROWS, D), (D, 1), (row, col), (RG, B), (1, 0)
        )
        tl.store(op, value.to(O.dtype.element_ty), boundary_check=(0, 1))


def residual_gate_add(residual, update, gate):
    out = torch.empty(
        residual.shape, dtype=residual.dtype, device=residual.device
    )
    n = residual.numel()
    if n:
        d = residual.shape[-1]
        rows = n // d
        b = min(8192, triton.next_power_of_2(d))
        rg = min(16, triton.next_power_of_2(rows), 32768 // b)
        _group[(min(triton.cdiv(rows, rg) * triton.cdiv(d, b), 96),)](
            residual,
            update,
            gate,
            out,
            rows,
            d,
            gate.numel() != n,
            b,
            rg,
            n >= 2147483648,
            num_warps=2,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["residual_gate_add"]
