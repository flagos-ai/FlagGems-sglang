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

"""Reference implementation for quantization/per_token_group_quant_fp8."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn
_EPS = 1e-10


def reference(x, group_size):
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max

    orig_shape = x.shape
    x32 = x.reshape(-1, group_size).to(torch.float32)
    amax = x32.abs().amax(dim=-1, keepdim=True).clamp(min=_EPS)
    scale = amax / fp8_max
    x_q = (x32 / scale).clamp(fp8_min, fp8_max).to(_FP8_DTYPE)

    x_q = x_q.reshape(orig_shape)
    x_s = scale.reshape(orig_shape[:-1] + (orig_shape[-1] // group_size,))
    return x_q, x_s
