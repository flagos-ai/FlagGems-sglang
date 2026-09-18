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


def reference(k, beta, g_cumsum=None, chunk_size=64):
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    ratio = H // Hg
    BT = chunk_size
    NT = T // BT

    k_c = k.float().view(B, NT, BT, Hg, K)
    k_c = k_c.repeat_interleave(ratio, dim=3)  # (B, NT, BT, H, K)
    k_c = k_c.permute(0, 1, 3, 2, 4)  # (B, NT, H, BT, K)

    A = torch.einsum("bnhik,bnhjk->bnhij", k_c, k_c)

    if g_cumsum is not None:
        g_c = (
            g_cumsum.float().view(B, NT, BT, H).permute(0, 1, 3, 2)
        )  # (B, NT, H, BT)
        g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
        A = A * torch.where(
            g_diff <= 0, torch.exp(g_diff), torch.zeros_like(g_diff)
        )

    beta_c = (
        beta.float().view(B, NT, BT, H).permute(0, 1, 3, 2)
    )  # (B, NT, H, BT)
    A = A * beta_c.unsqueeze(-1)

    causal = torch.tril(
        torch.ones(BT, BT, dtype=torch.bool, device=k.device), diagonal=-1
    )
    A = torch.where(causal, A, torch.zeros_like(A))

    # (B, NT, H, BT_i, BT_j) -> (B, NT, BT_i, H, BT_j) -> (B, T, H, BT)
    out = A.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)
    return out
