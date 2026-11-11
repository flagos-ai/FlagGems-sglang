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

"""Reference for per_token_group_quant_fp8.

Reproduces sglang's arithmetic rather than a textbook definition: the group
maximum is floored at ``1e-10``, the element is multiplied by ``448 / amax``,
and only then clamped into the finite E4M3 range. Quantizing with the
reciprocal instead of dividing by the scale is what the CUDA kernel does and it
is not bit-identical to the division, so the reference follows the multiply.
"""

import torch

FP8_MAX = 448.0
ABSMAX_FLOOR = 1.0e-10


def reference(x, group_size, dtype=torch.float8_e4m3fn):
    xf = x.float()
    shape = xf.shape
    cols = shape[-1]
    num_groups = cols // group_size

    groups = xf.reshape(*shape[:-1], num_groups, group_size)
    amax = groups.abs().amax(dim=-1).clamp(min=ABSMAX_FLOOR)

    inv = FP8_MAX / amax
    q = (groups * inv.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)

    quantized = q.to(dtype).reshape(shape)
    scales = amax / FP8_MAX
    return quantized, scales
