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

"""Reference implementation for fla/fused_recurrent_kda."""

import torch


def _sum_last(x):
    """Add adjacent pairs in fp32, carrying an odd final element unchanged."""
    while x.shape[-1] > 1:
        pairs = x.shape[-1] // 2
        reduced = x[..., : 2 * pairs : 2] + x[..., 1 : 2 * pairs : 2]
        if x.shape[-1] % 2:
            reduced = torch.cat((reduced, x[..., -1:]), dim=-1)
        x = reduced
    return x.squeeze(-1)


def _normalize(x):
    # Round each arithmetic boundary to fp32; evaluating the sqrt in fp64
    # avoids vendor reciprocal/sqrt approximations at this boundary.
    square_sum = _sum_last(x * x) + 1e-6
    denominator = square_sum.double().sqrt().float()
    return (x.double() / denominator[..., None].double()).float()


def reference(q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel):
    b, t, h, kdim = k.shape
    vdim = v.shape[-1]
    q32, k32, v32, beta32 = [x.float() for x in (q, k, v, beta)]
    # Only exp's internal evaluation is fp64; the stored decay and every
    # recurrent product and sum stay fp32, including their overflow behavior.
    decay = g.float().double().exp().float()
    state = (
        initial_state.float().contiguous().clone().reshape(b, h, vdim, kdim)
    )
    out = torch.empty((b, t, h, vdim), device=q.device, dtype=q.dtype)
    for i in range(t):
        qi, ki = q32[:, i], k32[:, i]
        if use_qk_l2norm_in_kernel:
            qi, ki = _normalize(qi), _normalize(ki)
        qi = qi * scale
        # Per-K-channel gate, broadcast along V.
        state = state * decay[:, i, :, None, :]
        vi = v32[:, i] - _sum_last(state * ki[:, :, None, :])
        bi = beta32[:, i]
        vi = vi * (bi if bi.dim() == 3 else bi[..., None])
        state = state + vi[..., None] * ki[:, :, None, :]
        out[:, i] = _sum_last(state * qi[:, :, None, :]).to(q.dtype)
    return out, state.reshape(initial_state.shape).to(initial_state.dtype)
