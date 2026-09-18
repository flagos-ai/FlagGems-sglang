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

import torch


def reference(
    x,
    weight,
    bias,
    eps,
    z=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=True,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size
    out_dtype = x.dtype

    xf = x.float().view(M, ngroups, group_size)
    zf = z.float().view(M, ngroups, group_size) if z is not None else None

    if zf is not None and not norm_before_gate:
        xf = xf * zf * torch.sigmoid(zf)

    if is_rms_norm:
        var = (xf**2).mean(dim=-1, keepdim=True)
    else:
        mean = xf.mean(dim=-1, keepdim=True)
        xf = xf - mean
        var = (xf**2).mean(dim=-1, keepdim=True)

    rstd = torch.rsqrt(var + eps)
    x_hat = xf * rstd

    w = weight.float().view(ngroups, group_size)
    y = x_hat * w
    if bias is not None:
        b = bias.float().view(ngroups, group_size)
        y = y + b

    if zf is not None and norm_before_gate:
        y = y * zf * torch.sigmoid(zf)

    return y.reshape(M, N).to(out_dtype)
