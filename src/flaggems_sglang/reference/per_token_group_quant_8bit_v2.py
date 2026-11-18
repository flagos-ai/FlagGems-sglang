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

"""Reference implementation for quantization/per_token_group_quant_8bit_v2."""


def reference(input, output_q, output_s, group_size, eps, min_8bit, max_8bit):
    x = input.float().reshape(-1, group_size)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = amax / max_8bit
    q = (x / scale).clamp(min_8bit, max_8bit).to(output_q.dtype)
    return q.reshape(output_q.shape), scale.reshape(output_s.shape).float()
