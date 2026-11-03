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
def _pair(X, Idx, Y, R: tl.constexpr, HD: tl.constexpr, B: tl.constexpr):
    cols = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    packed = Idx.to(tl.pointer_type(tl.uint64))
    for pair in range(
        tl.program_id(0).to(tl.int64), R // 2, tl.num_programs(0)
    ):
        bits = tl.load(packed + pair)
        for part in tl.static_range(2):
            idx = (bits >> part * 32).to(tl.int32).to(tl.int64)
            dst = Y + (pair.to(tl.int64) * 2 + part) * HD + cols
            if idx >= 0:
                value = tl.load(X + idx * HD + cols, mask=cols < HD, other=0)
                tl.store(dst, value, mask=cols < HD)
            else:
                tl.store(
                    dst, tl.full((B,), 0, Y.dtype.element_ty), mask=cols < HD
                )


@triton.jit
def _copy(
    X,
    I,
    Y,
    R: tl.constexpr,
    HD: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    SI: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
):
    cols = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    for row in range(tl.program_id(0).to(tl.int64), R, tl.num_programs(0)):
        idx = tl.load(I + row.to(tl.int64) * SI).to(tl.int64)
        value = tl.full((B,), 0, Y.dtype.element_ty)
        if N > 0:
            ptr = X + tl.maximum(idx, 0) * S0 + cols // D * S1 + cols % D * S2
            value = tl.load(ptr, mask=cols < HD, other=0)
            value = tl.where(idx >= 0, value, 0)
        tl.store(Y + row.to(tl.int64) * HD + cols, value, mask=cols < HD)


@triton.jit
def _cluster(X, Idx, Y, HD: tl.constexpr, N: tl.constexpr):
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)[:, None]
    X = X.to(tl.pointer_type(tl.uint64))
    Y = Y.to(tl.pointer_type(tl.uint64))
    cols = tl.arange(0, HD)[None, :]
    idx = tl.load(Idx + rows).to(tl.int64)
    value = tl.full((64, HD), 0, Y.dtype.element_ty)
    if N > 0:
        value = tl.load(
            X + tl.maximum(idx, 0) * HD + cols,
            offset_state_policy="locally_continuous",
        )
        value = tl.where(idx >= 0, value, 0)
    tl.store(Y + rows.to(tl.int64) * HD + cols, value)


def scatter_to_padded(out_unpad, inv_indices, batch_size, seqlen):
    n, h, d = out_unpad.shape
    hd = h * d
    rows = batch_size * seqlen
    out = torch.empty(
        (batch_size, seqlen, h, d),
        dtype=out_unpad.dtype,
        device=out_unpad.device,
    )
    if rows * hd:
        if (
            rows % 64 == 0
            and rows <= 2147483648
            and (hd <= 2048)
            and (hd * out_unpad.element_size() % 8 == 0)
            and (
                out_unpad.storage_offset() * out_unpad.element_size() % 8 == 0
            )
            and (hd & hd - 1 == 0)
            and (out_unpad.stride() == (hd, d, 1))
            and (inv_indices.stride(0) == 1)
        ):
            _cluster[rows // 64,](
                out_unpad,
                inv_indices,
                out,
                hd * out_unpad.element_size() // 8,
                n,
                groups_per_cluster=64,
            )
            return out
        block = min(2048, triton.next_power_of_2(hd))
        if (
            rows % 2 == 0
            and out_unpad.stride() == (hd, d, 1)
            and (inv_indices.dtype == torch.int32)
            and (inv_indices.stride(0) == 1)
            and (inv_indices.storage_offset() % 2 == 0)
        ):
            _pair[min(rows // 2, 192), triton.cdiv(hd, block)](
                out_unpad, inv_indices, out, rows, hd, block
            )
        else:
            _copy[min(rows, 192), triton.cdiv(hd, block)](
                out_unpad,
                inv_indices,
                out,
                rows,
                hd,
                d,
                *out_unpad.stride(),
                inv_indices.stride(0),
                n,
                block,
            )
    return out


__all__ = ["scatter_to_padded"]
