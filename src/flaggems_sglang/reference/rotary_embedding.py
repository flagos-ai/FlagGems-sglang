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

"""Pure-torch reference implementation of rotary_embedding (diffusion).

Used as correctness ground truth and benchmark baseline.
Source: kernel-comp-baseline/problems/diffusion/rotary_embedding.
"""

import torch


def reference(x, cos, sin, interleaved):
    """Interleaved-pair rotary embedding.

    Args:
        x:   Tensor[tokens, heads, head_size], bfloat16
        cos: Tensor[tokens, head_size // 2]
        sin: Tensor[tokens, head_size // 2]
        interleaved: bool (this problem uses False / half-width form)

    Returns:
        Rotated tensor with same shape and dtype as x.
    """
    xf = x.float()
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    c = cos.float().reshape(cos.shape[0], 1, -1)
    s = sin.float().reshape(sin.shape[0], 1, -1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack([o1, o2], dim=-1).reshape(xf.shape)
    return out.to(x.dtype)
