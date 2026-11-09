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

"""Reference implementation for fla/kda_gate_chunk_cumsum."""

import torch


def reference(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    b, t, h, s = g.shape
    x = g.to(torch.float32)
    if dt_bias is not None:
        x = x + dt_bias.reshape(h, s).to(torch.float32)[None, None]

    a = A_log.reshape(h).to(torch.float32)[None, None, :, None]
    if lower_bound is None:
        gate = -torch.exp(a) * torch.nn.functional.softplus(x)
    else:
        # Safe gate variant.
        gate = lower_bound * torch.sigmoid(torch.exp(a) * x)

    # Cumsum along time, restarted at every chunk boundary.
    pad = (-t) % chunk_size
    padded = torch.nn.functional.pad(gate, (0, 0, 0, 0, 0, pad))
    chunks = padded.reshape(b, -1, chunk_size, h, s)
    out = chunks.cumsum(dim=2).reshape(b, -1, h, s)[:, :t]

    if scale is not None:
        out = out * scale
    return out.to(torch.float32)
