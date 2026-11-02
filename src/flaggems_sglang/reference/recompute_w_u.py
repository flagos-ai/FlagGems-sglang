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


def reference(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None, "varlen path is out of scope"
    B, T, Hg, K = k.shape
    H = v.shape[-2]
    BT = A.shape[-1]
    heads_per_kv = H // Hg

    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)

    for c in range(T // BT):
        rows = slice(c * BT, (c + 1) * BT)
        # [B, H, BT, BT]
        Ac = A[:, rows].permute(0, 2, 1, 3).float()
        b_beta = beta[:, rows].permute(0, 2, 1).float()  # [B, H, BT]
        b_g = torch.exp(g_cumsum[:, rows].permute(0, 2, 1).float())

        vb = v[:, rows].permute(0, 2, 1, 3).float() * b_beta[..., None]
        u[:, rows] = (
            (Ac @ vb.to(v.dtype).float()).permute(0, 2, 1, 3).to(v.dtype)
        )

        # GQA: k carries only Hg key-value heads, while v, beta and g_cumsum
        # carry H query heads, so hk = h // (H // Hg).
        kh = k[:, rows].permute(0, 2, 1, 3).float()  # [B, Hg, BT, K]
        kh = kh.repeat_interleave(heads_per_kv, dim=1)  # [B, H, BT, K]
        kb = kh * b_beta[..., None] * b_g[..., None]
        w[:, rows] = (
            (Ac @ kb.to(k.dtype).float()).permute(0, 2, 1, 3).to(k.dtype)
        )

    return w, u
