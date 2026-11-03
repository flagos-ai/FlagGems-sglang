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

"""Reference implementation for diffusion/rmsnorm_scale_shift."""

import torch


def reference(x, weight, scale, shift, eps):
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    # The norm and the weight multiply happen in fp32 and round **once**, at
    # the end of the norm; the modulation that follows is bf16 arithmetic.
    # This ordering is what makes the baseline bit-exact vs the eager chain.
    normed = (xf * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(
        x.dtype
    )
    # scale/shift are [B, 1, D]: one modulation vector per batch element,
    # broadcast across the sequence.
    return normed * (1 + scale) + shift
