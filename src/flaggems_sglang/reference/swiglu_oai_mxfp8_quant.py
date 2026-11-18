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

"""Reference implementation for moe/swiglu_oai_mxfp8_quant."""

import torch


def reference(gate_up, alpha, beta, limit):
    orig_shape = gate_up.shape
    two_i = orig_shape[-1]
    n = two_i // 2

    x = gate_up.reshape(-1, two_i).to(torch.float32)
    gate, up = x[:, :n], x[:, n:]
    if limit is not None:
        # Asymmetric clamp, as in moe/swiglu_oai_split: gate is bounded above
        # only, up on both sides.
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    # The activation stays fp32 all the way into the scale selection, which
    # is the point of the fusion.
    activated = gate * torch.sigmoid(alpha * gate) * (up + beta)

    groups = activated.reshape(-1, n // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp(min=1e-30)
    scale_biased = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(
        0.0, 254.0
    )
    descale = torch.exp2(scale_biased - 127.0)
    q = (
        (groups / descale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )

    # Plain row-major scales, unlike quant_store_kv_mxfp8's interleaving.
    return (
        q.reshape(*orig_shape[:-1], n),
        scale_biased.to(torch.uint8).reshape(*orig_shape[:-1], n // 32),
    )
