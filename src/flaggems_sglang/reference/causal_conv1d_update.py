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

import torch


def reference(x, conv_state, weight, bias=None, activation="silu"):
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    conv_state = conv_state.clone()

    x_cat = torch.cat([conv_state.float(), x.float()], dim=-1)
    out = torch.zeros_like(x, dtype=torch.float32)

    for t in range(seqlen):
        window = x_cat[:, :, t : t + state_len + 1][:, :, -width:]
        val = (window * weight.float().unsqueeze(0)).sum(-1)
        if bias is not None:
            val = val + bias.float()
        if activation in ("silu", "swish"):
            val = val * torch.sigmoid(val)
        out[:, :, t] = val

    new_conv_state = x_cat[:, :, -state_len:]
    conv_state.copy_(new_conv_state.to(conv_state.dtype))

    out = out.to(x.dtype)
    if unsqueeze:
        out = out.squeeze(-1)
    return out, conv_state
