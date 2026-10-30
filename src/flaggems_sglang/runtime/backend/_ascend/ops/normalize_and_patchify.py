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
def _kernel(
    X,
    S,
    A,
    Y,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GH: tl.constexpr,
    PW: tl.constexpr,
    P: tl.constexpr,
    ST: tl.constexpr,
    SS: tl.constexpr,
    AS: tl.constexpr,
    STRIPE: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
):
    plane = tl.program_id(0)
    channel = plane % C
    batch = plane // C
    py = tl.program_id(1)
    tile = tl.program_id(2)
    col_tiles: tl.constexpr = tl.cdiv(PW, K)
    gy = tile // col_tiles * R + tl.arange(0, R)[:, None]
    x = tile % col_tiles * K + tl.arange(0, K)[None, :]
    y = gy * P + py
    scale = tl.load(S + channel * SS).to(tl.float32)
    bias = tl.load(A + channel * AS).to(tl.float32)
    if H == GH * P and W == PW:
        value = tl.load(
            X + batch * ST[0] + channel * ST[1] + y * ST[2] + x * ST[3],
            x < PW,
            other=0,
        ).to(tl.float32)
    else:
        value = tl.load(
            X
            + batch * ST[0]
            + channel * ST[1]
            + tl.minimum(y, H - 1) * ST[2]
            + tl.minimum(x, W - 1) * ST[3]
        ).to(tl.float32)
        value = tl.where((y < H) & (x < W), value, 0.0)
    tl.store(
        Y + (plane * P + py) * STRIPE + gy * PW + x,
        tl.fma(value, scale, bias),
        x < PW,
    )


def normalize_and_patchify(
    image, image_scale, image_bias, patch_size, padded_height, padded_width
):
    b, c, h, w = image.shape
    p = patch_size
    gh, gw = padded_height // p, padded_width // p
    stripe = max(64, triton.next_power_of_2(gh * padded_width))
    dtype = torch.promote_types(
        torch.promote_types(image.dtype, image_scale.dtype), image_bias.dtype
    )
    storage = torch.empty_strided(
        (b, gh * gw, c, p, p),
        (c * p * stripe, p, p * stripe, stripe, 1),
        device=image.device,
        dtype=dtype,
    )
    if b * c * gh * gw:
        rows = min(16, gh & -gh)
        cols = min(512, triton.next_power_of_2(padded_width))
        _kernel[(b * c, p, gh // rows * triton.cdiv(padded_width, cols))](
            image,
            image_scale,
            image_bias,
            storage,
            c,
            h,
            w,
            gh,
            padded_width,
            p,
            image.stride(),
            image_scale.stride(0),
            image_bias.stride(0),
            stripe,
            rows,
            cols,
            num_warps=4,
        )
    return storage


__all__ = ["normalize_and_patchify"]
