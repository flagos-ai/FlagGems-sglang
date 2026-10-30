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
    START: tl.constexpr,
    FULL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    plane = tl.program_id(0)
    channel = plane % C
    batch = plane // C
    py = tl.program_id(1)
    pos = START + tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    scale = tl.load(S + channel * SS).to(tl.float32)
    bias = tl.load(A + channel * AS).to(tl.float32)
    if H == GH * P and W == PW:
        y = pos // PW * P + py
        x = pos % PW
        if FULL:
            val = tl.load(
                X + batch * ST[0] + channel * ST[1] + y * ST[2] + x * ST[3]
            ).to(tl.float32)
        else:
            val = tl.load(
                X + batch * ST[0] + channel * ST[1] + y * ST[2] + x * ST[3],
                pos < GH * PW,
                other=0,
            ).to(tl.float32)
    else:
        safe = tl.minimum(pos, GH * PW - 1)
        y = safe // PW * P + py
        x = safe % PW
        val = tl.load(
            X
            + batch * ST[0]
            + channel * ST[1]
            + tl.minimum(y, H - 1) * ST[2]
            + tl.minimum(x, W - 1) * ST[3]
        ).to(tl.float32)
        val = tl.where((y < H) & (x < W), val, 0.0)
    tl.store(Y + (plane * P + py) * STRIPE + pos, tl.fma(val, scale, bias))


def normalize_and_patchify(
    image, image_scale, image_bias, patch_size, padded_height, padded_width
):
    b, c, h, w = image.shape
    p = patch_size
    gh, gw = padded_height // p, padded_width // p
    size = gh * padded_width
    stripe = max(64, triton.next_power_of_2(size))
    dtype = torch.promote_types(
        torch.promote_types(image.dtype, image_scale.dtype), image_bias.dtype
    )
    storage = torch.empty((b, c, p, stripe), device=image.device, dtype=dtype)
    if b * c * size:
        block = min(16384, stripe)
        full = size // block
        if full:
            _kernel[(b * c, p, full)](
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
                0,
                True,
                block,
                num_warps=4,
            )
        tail = size - full * block
        if tail:
            _kernel[(b * c, p, 1)](
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
                full * block,
                False,
                max(64, triton.next_power_of_2(tail)),
                num_warps=4,
            )
    return storage.as_strided(
        (b, gh * gw, c, p, p), (c * p * stripe, p, p * stripe, stripe, 1)
    )


__all__ = ["normalize_and_patchify"]
