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

"""Reference for the chunked gated delta-rule forward (vLLM-Ascend chunk.py).

Chunking is only a scheduling decision: the chunked and fused-recurrent forms
compute the same gated delta-rule recurrence, so the reference is the plain
sequential recurrence. For each head ``h`` and step ``t``::

    decay  = exp(g_t)
    pred   = decay * (S . k_t)              # S is [V, K]
    vcorr  = (v_t - pred) * beta_t
    S      = S * decay + outer(vcorr, k_t)
    o_t    = S . (q_t * scale)

Shapes: ``q, k`` are ``[B, T, H, K]``; ``v`` is ``[B, T, H, V]``; ``g`` and
``beta`` are ``[B, T, H]``. ``scale`` defaults to ``K ** -0.5``. With
``use_qk_l2norm_in_kernel`` the per-step ``q``/``k`` rows are L2-normalized
before the ``scale`` multiply. Returns ``(output, final_state)`` where
``final_state`` is ``None`` unless ``output_final_state`` is set.

This reference covers the single-sequence (``cu_seqlens is None``) path used by
the correctness and benchmark shapes.
"""

import torch


def reference(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5

    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = g.float()
    bf = beta.float()
    if use_qk_l2norm_in_kernel:
        qf = qf / qf.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()
        kf = kf / kf.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()

    out = torch.zeros(B, T, H, V, device=q.device, dtype=torch.float32)
    if initial_state is None:
        state = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    else:
        state = initial_state.clone().float()

    for b in range(B):
        for h in range(H):
            # S kept as [V, K] internally; initial_state stores [K, V].
            S = state[b, h].transpose(0, 1).contiguous()
            for t in range(T):
                kt = kf[b, t, h]
                qt = qf[b, t, h] * scale
                vt = vf[b, t, h]
                decay = torch.exp(gf[b, t, h])
                pred = decay * (S * kt[None, :]).sum(-1)
                vcorr = (vt - pred) * bf[b, t, h]
                S = S * decay + vcorr[:, None] * kt[None, :]
                out[b, t, h] = (S * qt[None, :]).sum(-1)
            state[b, h] = S.transpose(0, 1).contiguous()

    final_state = state if output_final_state else None
    return out.to(v.dtype), final_state
