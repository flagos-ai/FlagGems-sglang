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

"""Reference implementation for diffusion/wan_rmsnorm_silu."""

import torch
import torch.nn.functional as F


def reference(x, gamma, bias, rms_scale, eps):
    c = x.shape[1]
    scale = rms_scale if rms_scale is not None else c**0.5
    shape = (1, c, 1, 1, 1)
    # Normalization runs along the channel axis, which is the contiguous one
    # in channels_last_3d.
    y = (
        F.normalize(x.float(), dim=1, eps=eps)
        * scale
        * gamma.float().reshape(shape)
    )
    if bias is not None:
        y = y + bias.float().reshape(shape)
    out_dtype = torch.promote_types(x.dtype, gamma.dtype)
    return F.silu(y).to(out_dtype)
