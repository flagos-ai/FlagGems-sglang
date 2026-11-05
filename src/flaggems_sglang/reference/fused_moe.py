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

"""Reference for SGLang fused MoE experts (unquantized path).

Mirrors SGLang/vLLM ``fused_experts``: for each token, for each of its top-k
experts, compute the gated projection ``act(x @ w1_gate) * (x @ w1_up)`` then
the down projection ``(.) @ w2``, scale by the routing weight, and sum over the
selected experts. ``w1`` packs gate and up as ``[E, 2 * inter, hidden]`` with
gate in ``[:inter]`` and up in ``[inter:]``; ``w2`` is ``[E, hidden, inter]``.

``topk_output`` is the ``(topk_weights, topk_ids, _)`` tuple produced by the
router.
"""

import torch
import torch.nn.functional as F


def reference(
    hidden_states,
    w1,
    w2,
    topk_output,
    activation="silu",
    routed_scaling_factor=1.0,
):
    topk_weights, topk_ids, _ = topk_output
    T, H = hidden_states.shape
    E, N, _ = w1.shape
    inter = N // 2
    topk = topk_ids.shape[1]

    x = hidden_states.float()
    w1f = w1.float()
    w2f = w2.float()
    tw = topk_weights.float()

    out = torch.zeros(T, H, device=hidden_states.device, dtype=torch.float32)
    for j in range(topk):
        eids = topk_ids[:, j].long()
        weights = tw[:, j]
        # Per-token expert weights gathered into [T, N, H] / [T, H, inter].
        gate_up = torch.einsum("th,tnh->tn", x, w1f[eids])
        gate = gate_up[:, :inter]
        up = gate_up[:, inter:]
        if activation == "silu":
            act = F.silu(gate)
        elif activation == "gelu":
            act = F.gelu(gate)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        hidden = act * up
        down = torch.einsum("ti,thi->th", hidden, w2f[eids])
        out += weights[:, None] * down

    out = out * routed_scaling_factor
    return out.to(hidden_states.dtype)
