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

"""Reference implementation for quantization/per_tensor_quant_mla_fp8."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn


def reference(x, eps=1e-12):
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max
    # One scalar scale for the whole [num_head, num_seq, head_size] tensor.
    absmax = x.abs().float().max().clamp(min=eps)
    scale = (absmax / fp8_max).reshape(1)
    x_q = (x.to(torch.float32) / scale).clamp(fp8_min, fp8_max).to(_FP8_DTYPE)
    return x_q, scale
