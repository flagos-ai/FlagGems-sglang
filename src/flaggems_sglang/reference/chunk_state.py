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


def reference(B, x, dt, dA_cumsum):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups

    x_c = x.reshape(batch, nchunks, chunk_size, nheads, headdim).float()
    B_c = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).float()
    B_c = B_c.repeat_interleave(ratio, dim=3)

    dA_last = dA_cumsum[..., -1:].float()
    decay = torch.exp(dA_last - dA_cumsum.float())
    scale = (decay * dt.float()).permute(0, 2, 3, 1)

    Bs = B_c * scale.unsqueeze(-1)
    states = torch.einsum("bcthp,bcthn->bchpn", x_c, Bs)
    return states
