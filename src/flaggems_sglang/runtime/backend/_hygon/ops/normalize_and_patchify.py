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
    GW: tl.constexpr,
    P: tl.constexpr,
    ST: tl.constexpr,
    SS: tl.constexpr,
    AS: tl.constexpr,
    T: tl.constexpr,
    Q: tl.constexpr,
):
    plane = tl.program_id(1)
    channel = plane % C
    batch = plane // C
    patch = tl.program_id(0) * T + tl.arange(0, T)[:, None]
    pos = tl.arange(0, Q)[None, :]
    y = patch // GW * P + pos // P
    x = patch % GW * P + pos % P
    scale = tl.load(S + channel * SS).to(tl.float32)
    bias = tl.load(A + channel * AS).to(tl.float32)
    val = tl.load(
        X + batch * ST[0] + channel * ST[1] + y * ST[2] + x * ST[3],
        (patch < GH * GW) & (pos < P * P) & (y < H) & (x < W),
        other=0,
    ).to(tl.float32)
    out = tl.fma(val, scale, bias)
    dst = (plane * GH * GW + patch) * P * P + pos
    tl.store(Y + dst, out, (patch < GH * GW) & (pos < P * P))


def normalize_and_patchify(
    image, image_scale, image_bias, patch_size, padded_height, padded_width
):
    b, c, h, w = image.shape
    p = patch_size
    gh, gw = padded_height // p, padded_width // p
    dtype = torch.promote_types(
        torch.promote_types(image.dtype, image_scale.dtype), image_bias.dtype
    )
    out = torch.empty_strided(
        (b, gh * gw, c, p, p),
        (c * gh * gw * p * p, p * p, gh * gw * p * p, p, 1),
        device=image.device,
        dtype=dtype,
    )
    if out.numel():
        _kernel[(triton.cdiv(gh * gw, 8), b * c)](
            image,
            image_scale,
            image_bias,
            out,
            c,
            h,
            w,
            gh,
            gw,
            p,
            image.stride(),
            image_scale.stride(0),
            image_bias.stride(0),
            8,
            triton.next_power_of_2(p * p),
            num_warps=8,
        )
    return out


__all__ = ["normalize_and_patchify"]
