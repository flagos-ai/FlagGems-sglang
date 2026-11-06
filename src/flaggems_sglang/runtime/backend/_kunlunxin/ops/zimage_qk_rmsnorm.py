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
def _norm(
    X,
    W,
    Y,
    S: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    EPS: tl.constexpr,
    BR: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
):
    col = tl.arange(0, 128)
    w = tl.load(W + col[None, :]).to(tl.float32)
    for tile in range(tl.program_id(0), T, tl.num_programs(0)):
        row = tile * BR + tl.arange(0, BR)
        if CONTIGUOUS:
            offset = row[:, None] * 128 + col[None, :]
        else:
            offset = (
                row[:, None] // (S * H) * X0
                + row[:, None] // H % S * X1
                + row[:, None] % H * 128
                + col[None, :]
            )
        x = tl.load(X + offset, offset_state_policy="continuous").to(
            tl.float32
        )
        inv = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / 128) + EPS)
        norm = (x * inv[:, None]).to(Y.dtype.element_ty)
        for _ in range(1):
            tl.store(
                Y + row[:, None] * 128 + col[None, :],
                norm.to(tl.float32) * w,
                offset_state_policy="continuous",
            )


def zimage_qk_rmsnorm(x, weight, eps):
    b, s, h, d = x.shape
    if d != 128 or not weight.is_contiguous():
        return None
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = b * s * h
    if rows:
        contiguous = x.is_contiguous()
        if contiguous:
            br = min(rows & -rows, 64)
        else:
            br = min(h & -h, 64)
        tiles = rows // br
        _norm[(min(tiles, 12),)](
            x,
            weight,
            out,
            s,
            h,
            tiles,
            x.stride(0),
            x.stride(1),
            eps,
            br,
            contiguous,
            groups_per_cluster=64 if contiguous else br,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["zimage_qk_rmsnorm"]
