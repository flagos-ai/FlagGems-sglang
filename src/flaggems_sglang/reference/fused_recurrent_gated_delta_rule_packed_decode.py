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

"""Reference for the packed single-token gated delta-rule decode.

One decode step straight from the packed QKV projection into the paged SSM
state. ``mixed_qkv`` is ``[B, H*K + H*K + HV*V]`` holding q || k || v; ``a`` and
``b`` are ``[B, HV]``; ``A_log`` and ``dt_bias`` are ``[HV]``. ``initial_state``
is the paged state ``[num_slots, HV, V, K]``, indexed per batch row by
``ssm_state_indices``; a slot ``<= 0`` is a padding row (zero the output, leave
the state untouched).

Per value head ``hv`` (reading query/key head ``hv // (HV // H)``)::

    beta  = sigmoid(b)                      # round-tripped through b's dtype
    sp    = softplus(a + dt_bias)
    decay = exp(-exp(A_log) * sp)
    pred  = decay * (S . k)                 # S is [V, K]
    vcorr = (v - pred) * beta
    o     = decay * (S . (q * scale)) + vcorr * (k . (q * scale))
    S     = decay * S + outer(vcorr, k)

With ``use_qk_l2norm_in_kernel`` the q/k vectors are L2-normalized before the
``scale`` multiply. The op writes ``out`` and the paged state in place and
returns ``(out, initial_state)``.
"""

import torch
import torch.nn.functional as F


def reference(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    qk_dim = mixed_qkv.shape[1] - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K
    ratio = HV // H
    qk_half = H * K

    out_ref = torch.zeros(
        B, 1, HV, V, device=mixed_qkv.device, dtype=mixed_qkv.dtype
    )
    state = initial_state.clone().float()

    for n in range(B):
        slot = int(ssm_state_indices[n])
        if slot <= 0:
            continue
        row = mixed_qkv[n]
        for hv in range(HV):
            h = hv // ratio
            av = a[n, hv].float()
            beta = torch.sigmoid(b[n, hv].float()).to(b.dtype).float()
            alog = A_log[hv].float()
            dtb = dt_bias[hv].float()
            sp = F.softplus(av + dtb)
            decay = torch.exp(-torch.exp(alog) * sp)

            q = row[h * K : (h + 1) * K].float()
            kk = row[qk_half + h * K : qk_half + (h + 1) * K].float()
            vv = row[2 * qk_half + hv * V : 2 * qk_half + (hv + 1) * V].float()
            if use_qk_l2norm_in_kernel:
                q = q / q.pow(2).sum().add(1e-6).sqrt()
                kk = kk / kk.pow(2).sum().add(1e-6).sqrt()
            q = q * scale

            S = state[slot, hv]
            pred = decay * (S * kk[None, :]).sum(-1)
            vcorr = (vv - pred) * beta
            o = decay * (S * q[None, :]).sum(-1) + vcorr * (kk * q).sum()
            state[slot, hv] = decay * S + vcorr[:, None] * kk[None, :]
            out_ref[n, 0, hv] = o.to(mixed_qkv.dtype)

    return out_ref, state
