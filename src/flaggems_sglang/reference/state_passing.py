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

"""Pure-torch reference implementation of mamba/state_passing."""

import torch


def reference(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape

    if initial_states is None:
        cur = states.new_zeros(batch, nheads, dim, dtype=torch.float32)
    else:
        cur = initial_states.float().clone()

    out = torch.empty(
        batch,
        nchunks,
        nheads,
        dim,
        device=states.device,
        dtype=states.dtype,
    )
    states_f = states.float()
    dA_last = dA_cumsum[..., -1].float().permute(0, 2, 1)

    for c in range(nchunks):
        out[:, c] = cur.to(states.dtype)
        decay = torch.exp(dA_last[:, c]).unsqueeze(-1)
        cur = cur * decay + states_f[:, c]

    final_states = cur
    return out, final_states
