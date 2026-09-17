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


def reference(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.numel() - 1
    ratio = nheads // ngroups

    states = torch.zeros(
        batch,
        nheads,
        headdim,
        dstate,
        dtype=chunk_states.dtype,
        device=x.device,
    )

    for bidx in range(batch):
        start = int(cu_seqlens[bidx].item())
        end = int(cu_seqlens[bidx + 1].item())
        pid_c = (end - 1) // chunk_size
        chunk_start_tok = pid_c * chunk_size
        start_rel = start - chunk_start_tok
        end_rel = end - chunk_start_tok

        for h in range(nheads):
            g = h // ratio
            dA_cs_last = dA_cumsum[h, pid_c, end_rel - 1].float()
            x_seg = x[start:end, h, :].float()
            b_seg = B[start:end, g, :].float()
            dt_seg = dt[h, pid_c, start_rel:end_rel].float()
            dA_seg = dA_cumsum[h, pid_c, start_rel:end_rel].float()
            scale = torch.exp(dA_cs_last - dA_seg) * dt_seg
            b_scaled = b_seg * scale.unsqueeze(-1)
            states[bidx, h] = (x_seg.t() @ b_scaled).to(chunk_states.dtype)

    return states
