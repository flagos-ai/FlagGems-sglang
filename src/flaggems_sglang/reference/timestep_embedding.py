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

"""Reference implementation for diffusion/timestep_embedding."""

import math

import torch


def reference(
    t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
):
    half = dim // 2
    idx = torch.arange(half, dtype=torch.float32, device=t.device)
    exponent = -math.log(max_period) * idx / (half - downscale_freq_shift)
    angle = scale * t.float()[:, None] * torch.exp(exponent)[None, :]
    sin, cos = torch.sin(angle), torch.cos(angle)
    # Always float32, whatever t's dtype was.
    return torch.cat([cos, sin] if flip_sin_to_cos else [sin, cos], dim=-1)
