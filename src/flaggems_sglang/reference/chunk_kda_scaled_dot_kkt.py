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

"""Reference implementation for fla/chunk_kda_scaled_dot_kkt."""

import torch


def reference(q, k, gk, beta, scale, gk_scale, chunk_size):
    b, t, h, kdim = k.shape
    bt = chunk_size
    a_out = torch.zeros(b, t, h, bt, device=k.device, dtype=torch.float32)
    aqk_out = torch.zeros(b, t, h, bt, device=k.device, dtype=torch.float32)

    q32, k32 = q.to(torch.float32), k.to(torch.float32)
    g32 = gk.to(torch.float32) * gk_scale
    beta32 = beta.to(torch.float32)

    for lo in range(0, t, bt):
        hi = min(lo + bt, t)
        n = hi - lo
        gc = g32[:, lo:hi]  # [B, n, H, K]
        # exp2(g_i - g_j) per (row i, col j, head, channel)
        decay = torch.exp2(gc[:, :, None] - gc[:, None, :])  # [B, n, n, H, K]

        kk = k32[:, lo:hi]
        qq = q32[:, lo:hi]
        # A[i,j] = beta_i * sum_k k_i k_j exp2(g_i - g_j)
        a = torch.einsum("bihk,bjhk,bijhk->bijh", kk, kk, decay)
        a = a * beta32[:, lo:hi][:, :, None, :]
        aqk = torch.einsum("bihk,bjhk,bijhk->bijh", qq, kk, decay) * scale

        rows = torch.arange(n, device=k.device)
        strict = rows[:, None] > rows[None, :]
        lower = rows[:, None] >= rows[None, :]
        a = torch.where(strict[None, :, :, None], a, torch.zeros_like(a))
        aqk = torch.where(lower[None, :, :, None], aqk, torch.zeros_like(aqk))

        a_out[:, lo:hi, :, :n] = a.permute(0, 1, 3, 2)
        aqk_out[:, lo:hi, :, :n] = aqk.permute(0, 1, 3, 2)

    return a_out, aqk_out
