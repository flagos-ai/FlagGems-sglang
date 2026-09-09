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

_EPS = 1e-10


def reference(x, group_size, dtype=torch.int8):
    iinfo = torch.iinfo(dtype)
    int8_min, int8_max = iinfo.min, iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = (
        x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=_EPS).to(torch.float32)
    )
    x_s = amax / int8_max
    x_q = (x_ / x_s).clamp(min=int8_min, max=int8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s
