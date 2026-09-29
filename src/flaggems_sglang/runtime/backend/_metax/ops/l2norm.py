# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP32 last-dimension L2 normalization with grouped rows."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(
    X,
    Out,
    ROWS,
    D: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    col = tl.arange(0, BLOCK)
    base = row * XS0
    valid = (row[:, None] < ROWS) & (col[None, :] < D)
    x = tl.load(X + base[:, None] + col[None, :] * XS1, valid, other=0).to(
        tl.float32
    )
    inv = tl.rsqrt(tl.sum(x * x, 1) + EPS)
    tl.store(Out + row[:, None] * D + col[None, :], x * inv[:, None], valid)


def _launch(x, eps=1e-6, group=4, warps=4):
    if x.ndim == 0 or x.shape[-1] <= 0:
        raise ValueError("x must have a positive final dimension")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    width = x.shape[-1]
    rows = x.numel() // width
    if rows:
        flat = x.reshape(rows, width)
        _norm[(triton.cdiv(rows, group),)](
            flat,
            out,
            rows,
            width,
            float(eps),
            flat.stride(0),
            flat.stride(1),
            group,
            triton.next_power_of_2(width),
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return out


def l2norm(x, eps=1e-6):
    width = x.shape[-1]
    rows = x.numel() // width if width else 0
    group = (
        8
        if rows >= 256 and width <= 512
        else 4 if rows >= 4 and width <= 512 else 1
    )
    return _launch(x, eps, group=group, warps=4 if width <= 4096 else 8)


__all__ = ["l2norm"]
