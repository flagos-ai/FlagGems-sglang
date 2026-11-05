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

"""Reference implementation for diffusion/scale_residual_rmsnorm_scale_shift."""

import torch


def reference(residual, update, gate, weight, scale, shift, eps):
    # The gated residual add is bf16 arithmetic and is returned as-is: the next
    # block consumes it as its residual stream.
    res = residual + gate * update
    rf = res.to(torch.float32)
    var = rf.pow(2).mean(dim=-1, keepdim=True)
    normed = (rf * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(
        res.dtype
    )
    return normed * (1 + scale) + shift, res
