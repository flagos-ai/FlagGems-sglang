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

"""Reference implementation for embedding/timestep_embedding."""

import math

import torch


def reference(t, dim, flip_sin_to_cos, downscale_freq_shift, scale,
              max_period):
    half = max(dim // 2, 1)
    exponent = (
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / (half - downscale_freq_shift)
    )
    freqs = torch.exp(exponent)
    args = t[:, None].float() * freqs[None, :] * scale
    if flip_sin_to_cos:
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    else:
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding
