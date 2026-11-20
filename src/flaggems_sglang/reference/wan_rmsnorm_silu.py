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

"""Reference implementation for activation_norm/wan_rmsnorm_silu."""

import math

import torch


def reference(x, gamma, bias, rms_scale, eps):
    dtype = torch.result_type(x, gamma)
    xf = x.to(torch.float32)
    b, c = x.shape[0], x.shape[1]
    xr = xf.reshape(b, c, -1)
    rms = torch.sqrt(xr.pow(2).mean(dim=1, keepdim=True) + eps)
    scale = (
        float(rms_scale)
        if rms_scale is not None
        else math.sqrt(c)
    )
    y = xr / rms * scale
    y = y * gamma.to(torch.float32).reshape(1, c, 1)
    if bias is not None:
        y = y + bias.to(torch.float32).reshape(1, c, 1)
    y = y * torch.sigmoid(y)
    return y.reshape(x.shape).to(dtype)
