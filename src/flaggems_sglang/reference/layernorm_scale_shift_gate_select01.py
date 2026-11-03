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

"""Reference implementation for diffusion/layernorm_scale_shift_gate_select01."""

import torch


def reference(
    x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index, eps
):
    b, l, c = x.shape
    xf = x.to(torch.float32)
    mean = xf.mean(dim=-1, keepdim=True)
    xbar = xf - mean
    var = (xbar * xbar).mean(dim=-1, keepdim=True)
    x_hat = xbar * torch.rsqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * weight.to(torch.float32)
    if bias is not None:
        x_hat = x_hat + bias.to(torch.float32)

    # Per (batch, token), `index` picks modulation set 0 or 1.
    sel = index.to(torch.bool)[..., None]  # [B, L, 1]
    scale = torch.where(
        sel,
        scale1[:, None].to(torch.float32),
        scale0[:, None].to(torch.float32),
    )
    shift = torch.where(
        sel,
        shift1[:, None].to(torch.float32),
        shift0[:, None].to(torch.float32),
    )
    gate = torch.where(sel, gate1[:, None], gate0[:, None]).expand(b, l, c)

    y = x_hat * (1.0 + scale) + shift
    return y.to(x.dtype), gate.to(x.dtype).contiguous()
