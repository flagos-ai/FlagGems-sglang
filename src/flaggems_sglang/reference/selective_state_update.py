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

"""Pure-torch reference implementation of mamba/selective_state_update."""

import torch
import torch.nn.functional as F


def reference(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    state = state.clone()
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = nheads // ngroups

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = F.softplus(dt_f)

    dA = torch.exp(dt_f.unsqueeze(-1) * A.float().unsqueeze(0))
    B_exp = B.float().repeat_interleave(ratio, dim=1)
    dB = dt_f.unsqueeze(-1) * B_exp.unsqueeze(2)

    new_state = state.float() * dA + dB * x.float().unsqueeze(-1)
    state = new_state.to(state.dtype)

    C_exp = C.float().repeat_interleave(ratio, dim=1)
    y = torch.einsum("bhpn,bhn->bhp", new_state, C_exp)

    if D is not None:
        y = y + D.float() * x.float()
    if z is not None:
        y = y * (z.float() * torch.sigmoid(z.float()))

    return y.to(x.dtype), state
