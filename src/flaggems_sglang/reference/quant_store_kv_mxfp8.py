# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference implementation for quantization/quant_store_kv_mxfp8."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_BLOCK = 32


def _quant_one(src):
    t, nkv, d = src.shape
    sf = d // _BLOCK
    x = src.to(torch.float32).reshape(t, nkv, sf, _BLOCK)
    amax = x.abs().amax(dim=-1).clamp(min=1e-30)
    scale_biased = (torch.ceil(torch.log2(amax / _FP8_MAX)) + 127.0).clamp(
        0.0, 254.0
    )
    descale = torch.exp2(scale_biased - 127.0)
    xq = (x / descale[..., None]).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return xq.reshape(t, nkv, d), scale_biased.to(torch.uint8)


def reference(k, v, loc, k_cache, v_cache, sfk, sfv, page_size):
    t, nkv, d = k.shape
    sf = d // _BLOCK
    loc = loc.to(torch.int64)

    out = []
    for src, cache, scales in ((k, k_cache, sfk), (v, v_cache, sfv)):
        cache = cache.clone()
        scales = scales.clone()
        xq, sb = _quant_one(src)

        cache.view(-1, nkv, d)[loc] = xq

        po = loc % page_size
        pg = loc // page_size
        h = torch.arange(nkv, device=k.device)
        base = (
            (pg[:, None] * nkv + h[None, :])
            * (32 * (page_size // _BLOCK) * sf)
            + (po % 32)[:, None] * ((page_size // _BLOCK) * sf)
            + (po // 32)[:, None] * sf
        )
        idx = (
            base[:, :, None] + torch.arange(sf, device=k.device)[None, None, :]
        )
        scales.view(torch.uint8).view(-1)[idx.reshape(-1)] = sb.reshape(-1)

        out.append(cache)
        out.append(scales)

    kc, sk, vc, sv = out
    return kc, vc, sk, sv
