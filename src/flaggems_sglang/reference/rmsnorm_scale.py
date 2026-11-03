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

"""Reference implementation for diffusion/rmsnorm_scale."""

import torch


def reference(x, weight, scale, eps):
    dim = x.shape[-1]
    xf = x.reshape(-1, dim).to(torch.float32)
    normed = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(
        x.dtype
    )
    normed = normed * weight
    # `scale` may have fewer rows than `x`; each scale row covers a contiguous
    # run of x rows (x_rows must be a multiple of scale_rows).
    s = scale.reshape(-1, dim)
    repeat = normed.shape[0] // s.shape[0]
    return (normed * s.repeat_interleave(repeat, dim=0)).reshape(x.shape)
