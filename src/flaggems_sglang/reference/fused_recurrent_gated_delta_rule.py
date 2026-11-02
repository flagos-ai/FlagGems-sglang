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

"""Reference for the fused recurrent gated delta rule (Qwen3-Next / FLA).

Sequential delta-rule recurrence with a per-step scalar gate. For each
value head ``hv`` (which reads query/key head ``hv // (HV // H)``) and each
step ``t``::

    decay  = exp(g_t)
    pred   = decay * (S . k_t)              # S is [V, K]
    vcorr  = (v_t - pred) * beta_t
    S      = S * decay + outer(vcorr, k_t)
    o_t    = S . (q_t * scale)

Shapes: ``q, k`` are ``[B, T, H, K]``; ``v`` is ``[B, T, HV, V]``; ``g`` and
``beta`` are ``[B, T, HV]`` (scalar per value head). When
``use_qk_l2norm_in_kernel`` is set, ``q`` and ``k`` are L2-normalized per head
before the recurrence. Returns ``(output, final_state)`` where ``final_state``
is ``None`` unless ``output_final_state`` is set.
"""

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

    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = g.float()
    bf = beta.float()
    if use_qk_l2norm_in_kernel:
        qf = qf / qf.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()
        kf = kf / kf.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()

    out = torch.zeros(B, T, HV, V, device=q.device, dtype=torch.float32)
    if initial_state is None:
        state = torch.zeros(B, HV, V, K, device=q.device, dtype=torch.float32)
    else:
        state = initial_state.clone().float()

    g_per_vhead = gf.dim() == 3 and gf.shape[-1] == HV
    beta_per_vhead = bf.dim() == 3 and bf.shape[-1] == HV

    for b in range(B):
        for hv in range(HV):
            h = hv // ratio
            S = state[b, hv]
            for t in range(T):
                kt = kf[b, t, h]
                qt = qf[b, t, h] * scale
                vt = vf[b, t, hv]
                gt = gf[b, t, hv] if g_per_vhead else gf[b, t, h]
                bt = bf[b, t, hv] if beta_per_vhead else bf[b, t, h]
                decay = torch.exp(gt)
                pred = decay * (S * kt[None, :]).sum(-1)
                vcorr = (vt - pred) * bt
                S = S * decay + vcorr[:, None] * kt[None, :]
                out[b, t, hv] = (S * qt[None, :]).sum(-1)
            state[b, hv] = S

    final_state = state if output_final_state else None
    return out.to(v.dtype), final_state
