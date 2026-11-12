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

"""Reference implementation for quantization/per_tensor_quant_fp8."""

import torch

_FP8 = torch.float8_e4m3fn


def reference(input, output_q, output_s, is_static):
    fp8_max = torch.finfo(_FP8).max
    if is_static:
        scale = output_s.clone().float()
    else:
        scale = (input.float().abs().amax() / fp8_max).reshape(1)
    q = (input.float() / scale).clamp(-fp8_max, fp8_max).to(_FP8)
    return q, scale.reshape(output_s.shape).float()
