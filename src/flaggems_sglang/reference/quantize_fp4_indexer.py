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

"""Reference implementation for quantization/quantize_fp4_indexer."""

import torch

_GROUP_N = 32
_BLOCK_N = 128
_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _ceil_ue8m0_exp(x):
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + (mantissa != 0).to(torch.int32)
    return exp.clamp(1, 254)


def _fp4_e2m1_code(x):
    ax = x.abs().clamp(max=6.0)
    idx = torch.zeros_like(ax, dtype=torch.int32)
    for t in _THRESHOLDS:
        idx += (ax > t).to(torch.int32)
    sign = ((x < 0) & (idx != 0)).to(torch.int32)
    return idx | (sign << 3)


def reference(x):
    x = x.contiguous().view(-1, _BLOCK_N).to(torch.float32)
    n = x.shape[0]
    groups = x.view(n, _BLOCK_N // _GROUP_N, _GROUP_N)

    amax = groups.abs().amax(dim=-1)
    sf = torch.clamp(amax / 6.0, min=1.0e-4)
    exp = _ceil_ue8m0_exp(sf)

    packed_sf = (
        exp[:, 0] | (exp[:, 1] << 8) | (exp[:, 2] << 16) | (exp[:, 3] << 24)
    ).to(torch.int32)

    scale = (exp << 23).view(torch.float32)
    v = groups / scale[:, :, None]
    code = _fp4_e2m1_code(v).view(n, _BLOCK_N)

    lo, hi = code[:, 0::2], code[:, 1::2]
    packed = ((lo & 0x0F) | ((hi & 0x0F) << 4)).to(torch.uint8)
    return packed.view(torch.int8), packed_sf
