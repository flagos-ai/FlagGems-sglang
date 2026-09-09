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


def reference(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    ratio = HV // H
    beta_headwise = beta.dim() == v.dim()

    if initial_state is not None:
        state = initial_state.float().clone()
    else:
        state = q.new_zeros(B, HV, V, K, dtype=torch.float32)

    o = q.new_zeros(B, T, HV, V, dtype=torch.float32)

    for t in range(T):
        qt = q[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        gt = g[:, t].float()

        if use_qk_l2norm_in_kernel:
            qt = qt / (qt.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
            kt = kt / (kt.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        qt = qt * scale

        kt_e = (
            kt.repeat_interleave(ratio, dim=1) if ratio > 1 else kt
        )  # (B, HV, K)
        qt_e = (
            qt.repeat_interleave(ratio, dim=1) if ratio > 1 else qt
        )  # (B, HV, K)

        state = state * gt.exp()[:, :, None, None]

        pred = torch.einsum("bhvk,bhk->bhv", state, kt_e)
        vt_corr = vt - pred

        if beta_headwise:
            bt = beta[:, t].float()
        else:
            bt = beta[:, t].float().unsqueeze(-1)
        vt_corr = vt_corr * bt

        state = state + vt_corr.unsqueeze(-1) * kt_e.unsqueeze(-2)

        o[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt_e)

    final_state = state if output_final_state else None
    return o.to(v.dtype), final_state
