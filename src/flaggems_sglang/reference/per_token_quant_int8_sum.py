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

"""Reference implementation for quantization/per_token_quant_int8_sum."""

import torch


def _round_half_away_from_zero(v):
    return torch.sign(v) * torch.floor(torch.abs(v) + 0.5)


def reference(x, scale_dtype, cal_sum):
    x32 = x.float()
    absmax = x32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax / 127
    x_q = _round_half_away_from_zero(x32 * (127 / absmax)).to(torch.int8)
    scales = scale.to(scale_dtype)
    if not cal_sum:
        return x_q, scales
    return x_q, scales, x32.sum(dim=-1).to(x.dtype)
