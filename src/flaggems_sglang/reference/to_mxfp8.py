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

"""Reference implementation for quantization/to_mxfp8."""

import torch

_BLOCK = 32


def reference(x):
    orig = x.shape
    x2d = x.contiguous().reshape(-1, orig[-1]).float()
    blocks = x2d.reshape(x2d.shape[0], -1, _BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-30)
    scale_biased = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(
        0.0, 254.0
    )
    descale = torch.exp2(scale_biased - 127.0)
    xq = (
        (blocks / descale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    return (
        xq.reshape(orig),
        scale_biased.to(torch.uint8).reshape(*orig[:-1], orig[-1] // _BLOCK),
    )
